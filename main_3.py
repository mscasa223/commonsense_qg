# -*- coding: utf-8 -*-
"""
main.py  –  Full pipeline runner on CosmosQA dataset
=====================================================
Run from the parent folder with:
    python -m commonsense_dcqg
"""
from __future__ import annotations

import os
import csv
import json
import time
import xml.etree.ElementTree as ET
from typing import List, Dict

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from graph import (
    parse_amr, amr_to_graph, extract_entities,
    retrieve_kb, build_inferential_graph,
    retrieve_subgraph, PathSkeleton,
    encode_path, hop_count, features_to_vec128,
    LocalTSVBackend,
)
from skeletons import mine_skeletons
from hsmm import train_hsmm, HSMMInfer, ExpressivePool
from generatorr import (
    Vocabulary, ControllableGenerator,
    train_generator, generate_question,
    TORCH_AVAILABLE as GEN_TORCH,
)

BASE_PATH = r"C:\Users\sanke\OneDrive\Desktop\commonsense_qg"
# =============================================================================
# Step 1 — Load CSV
# =============================================================================

def load_cosmos_csv(path: str, limit: int = None) -> List[Dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            correct_idx           = int(row['label'])
            row['correct_answer'] = row[f'answer{correct_idx}']
            rows.append(row)
    return rows

def load_kqapro_json(path, limit=None):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for i, item in enumerate(data):
        if limit and i >= limit:
            break
        rows.append({
            'id': i,
            'context': item.get('question', ''),
            'question': item.get('question', ''),
            'correct_answer': item.get('answer', '')
        })
    return rows


def load_grailqa_json(path, limit=None):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for i, item in enumerate(data):
        if limit and i >= limit:
            break
        rows.append({
            'id': i,
            'context': item.get('question', ''),
            'question': item.get('question', ''),
            'correct_answer': ''
        })
    return rows


def load_mcscript_xml(path, limit=None):
    rows = []
    tree = ET.parse(path)
    root = tree.getroot()

    idx = 0

    for instance in root.findall("instance"):
        text_elem = instance.find("text")
        if text_elem is None or text_elem.text is None:
            continue

        context = text_elem.text.strip()

        questions_block = instance.find("questions")
        if questions_block is None:
            continue

        for q in questions_block.findall("question"):
            if limit and idx >= limit:
                return rows

            question_text = q.attrib.get("text", "").strip()

            correct_answer = ""
            for ans in q.findall("answer"):
                if ans.attrib.get("correct") == "True":
                    correct_answer = ans.attrib.get("text", "").strip()
                    break

            if question_text:
                rows.append({
                    "id": idx,
                    "context": context,
                    "question": question_text,
                    "correct_answer": correct_answer
                })
                idx += 1

    return rows

# =============================================================================
# Step 2 — Build Vocabulary from all contexts + questions
# =============================================================================

def prepare_vocab(rows: List[Dict],
                  vocab_path: str = 'data/cache/vocab.json') -> Vocabulary:
    vocab = Vocabulary()
    if os.path.isfile(vocab_path):
        vocab.load(vocab_path)
        return vocab
    # Build from all contexts and questions combined
    all_texts = [r['context'] for r in rows] + [r['question'] for r in rows]
    vocab.build(all_texts, min_freq=2, max_size=10000)
    vocab.save(vocab_path)
    return vocab


# =============================================================================
# Step 3 — Train HSMM on all questions
# =============================================================================

def prepare_hsmm(rows: List[Dict],
                 num_states:   int = 20,
                 max_duration: int = 5,
                 iters:        int = 10) -> tuple:
    print(f'[HSMM] Tokenizing {len(rows)} questions ...')
    questions_tok = [row['question'].lower().split() for row in rows]

    print(f'[HSMM] Training HSMM (states={num_states}, iters={iters}) ...')
    ckpt_path = train_hsmm(
        questions_tok,
        num_states=num_states,
        max_duration=max_duration,
        iters=iters,
        out_dir='data/cache/forms',
    )
    print(f'[HSMM] Checkpoint saved to: {ckpt_path}')

    infer    = HSMMInfer(ckpt_path)
    pool     = ExpressivePool()
    seg_ids  = []
    toks_all = []
    for toks in questions_tok:
        segs = infer.segment(toks)
        seg_ids.append([s for s, _ in segs])
        toks_all.append(toks)
    pool.add_segmented(seg_ids, toks_all)
    print(f'[HSMM] Pool built with {len(pool.form_freq)} unique forms.')
    return infer, pool


# =============================================================================
# Step 4 — Mine skeletons from all questions
# =============================================================================

def prepare_skeletons(rows: List[Dict]) -> str:
    questions = [row['question'] for row in rows]
    print(f'[Skeletons] Mining from {len(questions)} questions ...')
    sk_path = mine_skeletons(questions, top_k=50)
    print(f'[Skeletons] Saved to: {sk_path}')
    return sk_path


# =============================================================================
# Step 5 — Process one row (graph + subgraph + HSMM + generate question)
# =============================================================================

def process_row(row:        Dict,
                hsmm_infer: HSMMInfer,
                pool:       ExpressivePool,
                model,
                vocab:      Vocabulary,
                device,
                kb_backend) -> Dict:

    context        = row['context']
    question       = row['question']
    correct_answer = row['correct_answer']

    # --- Graph ---
    amr     = parse_amr(context)
    G_ctx   = amr_to_graph(amr)
    ents    = extract_entities(context)
    triples = retrieve_kb(ents, k=50, backend=kb_backend)
    G       = build_inferential_graph(G_ctx, triples)

    # --- Subgraph ---
    subg        = retrieve_subgraph(G, PathSkeleton(max_hops=2),
                                    answer_labels=[correct_answer], topk=3)
    feats       = encode_path(subg) if subg.paths else {'hop_count': 0}
    hops        = hop_count(subg)
    path_vec128 = features_to_vec128(feats) if np is not None else None

    # --- HSMM form ---
    q_toks      = question.lower().split()
    segs        = hsmm_infer.segment(q_toks)
    form_states = [s for s, _ in segs] or [0]

    # --- Generate real question ---
    generated_question = None
    if TORCH_AVAILABLE and model is not None and vocab is not None:
        path_vec_tensor = (
            torch.tensor(np.array([path_vec128]), dtype=torch.float32, device=device)
            if path_vec128 is not None
            else None
        )
        # generate_question returns a real decoded string now
        generated_question = generate_question(
            model       = model,
            context     = context,        # real context, not fake [10,11,12,13]
            vocab       = vocab,          # real vocabulary
            device      = device,
            path_vec    = path_vec_tensor,
            skeleton_id = 0,
            form_states = form_states,
            difficulty  = min(hops, 9),
        )

    return {
        'id':                 row['id'],
        'context':            context,
        'original_question':  question,       # ground truth
        'correct_answer':     correct_answer,
        'hop_count':          hops,
        'form_states':        form_states,
        'generated_question': generated_question,  # real words now
    }




def run_pipeline(rows, output_path, LIMIT,
                 HSMM_STATES=8, HSMM_ITERS=3,
                 TRAIN_EPOCHS=3, TRAIN_BATCH=32):

    print(f'\n[Data] Loaded {len(rows)} rows.')

    # Vocabulary
    vocab = prepare_vocab(rows, vocab_path='data/cache/vocab.json')

    # HSMM
    hsmm_infer, pool = prepare_hsmm(rows,
                                    num_states=HSMM_STATES,
                                    iters=HSMM_ITERS)

    # Skeletons
    prepare_skeletons(rows)

    # Generator
    model = None
    device = None
    if TORCH_AVAILABLE:
        device = torch.device('cpu')
        model = ControllableGenerator(vocab_size=len(vocab))
        model.to(device)

        gen_ckpt = 'data/cache/generator.pt'
        if os.path.isfile(gen_ckpt):
            model.load_state_dict(torch.load(gen_ckpt, map_location=device))
        else:
            train_generator(
                model=model,
                rows=rows,
                vocab=vocab,
                device=device,
                epochs=TRAIN_EPOCHS,
                batch_size=TRAIN_BATCH,
                save_path=gen_ckpt,
            )

    kb_backend = LocalTSVBackend()

    print(f'\n[Pipeline] Saving to {output_path}')
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for row in rows:
            try:
                result = process_row(row, hsmm_infer, pool,
                                     model, vocab, device, kb_backend)
                out_f.write(json.dumps(result) + '\n')
            except Exception as e:
                out_f.write(json.dumps({'id': row['id'], 'error': str(e)}) + '\n')

# =============================================================================
# Main
# =============================================================================

def main():

    LIMIT = 1000

    # ---- File Paths ----
    cosmos_path  = os.path.join(BASE_PATH, "Cosmostrain.csv")
    mcscript_path = os.path.join(BASE_PATH, "MCScript_train.xml")
    kqapro_path  = os.path.join(BASE_PATH, "KQAPro_train.json")
    grailqa_path = os.path.join(BASE_PATH, "grailqa_v1.0_train.json")

    # ---- Output Paths ----
    cosmos_out  = os.path.join(BASE_PATH, "results_cosmosqa.jsonl")
    mcscript_out = os.path.join(BASE_PATH, "results_mcscript.jsonl")
    kqapro_out  = os.path.join(BASE_PATH, "results_kqapro.jsonl")
    grailqa_out = os.path.join(BASE_PATH, "results_grailqa.jsonl")

    # ============================
    # CosmosQA
    # ============================
    print("\n=== Running CosmosQA ===")
    rows = load_cosmos_csv(cosmos_path, limit=LIMIT)
    run_pipeline(rows, cosmos_out, LIMIT)

    # ============================
    # MCScript
    # ============================
    print("\n=== Running MCScript ===")
    rows = load_mcscript_xml(mcscript_path, limit=LIMIT)
    run_pipeline(rows, mcscript_out, LIMIT)

    # ============================
    # KQA Pro
    # ============================
    print("\n=== Running KQA Pro ===")
    rows = load_kqapro_json(kqapro_path, limit=LIMIT)
    run_pipeline(rows, kqapro_out, LIMIT)

    # ============================
    # GrailQA
    # ============================
    print("\n=== Running GrailQA ===")
    rows = load_grailqa_json(grailqa_path, limit=LIMIT)
    run_pipeline(rows, grailqa_out, LIMIT)

if __name__ == '__main__':
    main()