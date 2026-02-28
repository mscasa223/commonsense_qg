# -*- coding: utf-8 -*-
"""
skeletons.py  –  Section 3: Reasoning Skeletons + Difficulty Control
=====================================================================
Covers:
  - Skeleton data structures (EdgeSpec, Skeleton)
  - Skeleton mining from unlabelled questions (proxy regex slots)
  - Skeleton-to-graph matching (placeholder assignment with constraints)
  - Difficulty controller (dl -> skeletons; hop-count check)
"""
from __future__ import annotations

import re
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter

from graph import InferentialGraph, Subgraph, hop_count


# =============================================================================
# 3.1  Skeleton data structures
# =============================================================================

@dataclass(frozen=True)
class EdgeSpec:
    src: str
    rel: str
    tgt: str


@dataclass
class Skeleton:
    id: str
    reasoning_type: str
    structure: List[EdgeSpec]
    size: int
    meta: Dict[str, str] = field(default_factory=dict)

    def placeholders(self) -> List[str]:
        ps: List[str] = []
        for e in self.structure:
            if e.src not in ps:
                ps.append(e.src)
            if e.tgt not in ps:
                ps.append(e.tgt)
        return ps

    def edges(self) -> List[Tuple[str, str, str]]:
        return [(e.src, e.rel, e.tgt) for e in self.structure]


# Convenience prototype skeletons
BRIDGE_2HOP = Skeleton(
    id='bridge_2hop',
    reasoning_type='bridge',
    structure=[EdgeSpec('X0', 'relatedTo', 'X1'),
               EdgeSpec('X1', 'relatedTo', 'X2')],
    size=3,
    meta={'desc': 'X0 -> X1 -> X2'},
)

CAUSAL_2HOP = Skeleton(
    id='causal_2hop',
    reasoning_type='causal',
    structure=[EdgeSpec('X0', 'causes', 'X1'),
               EdgeSpec('X1', 'causes', 'X2')],
    size=3,
    meta={'desc': 'X0 causes X1; X1 causes X2'},
)


# =============================================================================
# 3.2  Skeleton Mining (proxy regex)
# =============================================================================

SLOT_PATTERNS = [
    (r'\b(why)\b',                'WHY'),
    (r'\b(when)\b',               'WHEN'),
    (r'\b(where)\b',              'WHERE'),
    (r'\b(what|which)\b',         'WHAT'),
    (r'\b(how)\b',                'HOW'),
    (r'\b(because|due to|as a result)\b', 'CAUSE'),
    (r'\b(so that|in order to)\b', 'PURPOSE'),
    (r'\b(before|after|during)\b', 'TEMPORAL'),
]


def _slot_sequence(question: str) -> List[str]:
    q = question.lower()
    slots: List[str] = []
    for pat, tag in SLOT_PATTERNS:
        if re.search(pat, q):
            slots.append(tag)
    if not slots:
        if re.match(r'\s*(what|which|who|where|when|why|how)\b', q):
            slots.append('WH')
    return sorted(set(slots))


def _pattern_to_skeleton(pattern: Tuple[str, ...], sk_id: str) -> Skeleton:
    if 'CAUSE' in pattern:
        structure = [EdgeSpec('X0', 'causes', 'X1')]
        rtype = 'causal'
    elif 'TEMPORAL' in pattern:
        structure = [EdgeSpec('X0', 'temporal', 'X1')]
        rtype = 'temporal'
    elif 'WHERE' in pattern:
        structure = [EdgeSpec('X0', 'atLocation', 'X1')]
        rtype = 'spatial'
    else:
        structure = [EdgeSpec('X0', 'relatedTo', 'X1')]
        rtype = 'bridge'
    return Skeleton(id=sk_id, reasoning_type=rtype,
                    structure=structure, size=2)


def mine_skeletons(questions: List[str],
                   out_dir: str = 'data/cache/skeletons',
                   top_k: int = 50) -> str:
    os.makedirs(out_dir, exist_ok=True)
    signatures: List[Tuple[str, ...]] = []
    for q in questions:
        signatures.append(tuple(_slot_sequence(q)))
    freq = Counter(signatures)
    sk_list: List[Dict] = []
    for i, (sig, count) in enumerate(freq.most_common(top_k)):
        sk = _pattern_to_skeleton(sig, sk_id=f'skel_{i:04d}')
        sk.meta.update({'support': count, 'signature': list(sig)})
        sk_list.append({
            'id': sk.id,
            'reasoning_type': sk.reasoning_type,
            'structure': [{'src': e.src, 'rel': e.rel, 'tgt': e.tgt}
                          for e in sk.structure],
            'size': sk.size,
            'meta': sk.meta,
        })
    out_path = os.path.join(out_dir, 'skeleton_bank.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(sk_list, f, ensure_ascii=False, indent=2)
    return out_path


# =============================================================================
# 3.3  Skeleton Matcher
# =============================================================================

@dataclass
class Alignment:
    placeholder_to_node: Dict[str, str]
    score: float


def match_skeleton(G: InferentialGraph,
                   skeleton: Skeleton,
                   answer_labels: List[str],
                   max_alignments: int = 10) -> List[Alignment]:
    tgt_ids: List[str] = []
    for lab in answer_labels:
        tgt_ids.extend(G.find_nodes_by_label(lab))
    tgt_ids = list(dict.fromkeys(tgt_ids))
    if not tgt_ids:
        return []

    term = skeleton.structure[-1].tgt if skeleton.structure else 'X0'
    aligns: List[Alignment] = []
    for tgt in tgt_ids:
        partial = {term: tgt}
        used    = {tgt}
        ok, m   = _backtrack_assign(G, skeleton, partial, used)
        if ok:
            aligns.append(Alignment(placeholder_to_node=m, score=1.0))
            if len(aligns) >= max_alignments:
                break
    return aligns


def _backtrack_assign(G: InferentialGraph,
                      sk: Skeleton,
                      mapping: Dict[str, str],
                      used_nodes: Set[str]) -> Tuple[bool, Dict[str, str]]:
    if set(mapping.keys()) == set(sk.placeholders()):
        return (_check_edges(G, sk, mapping), mapping)

    for es in sk.structure:
        if es.src in mapping and es.tgt not in mapping:
            src_node = mapping[es.src]
            for e in G.edges:
                if (e.src == src_node and e.rel == es.rel
                        and e.tgt not in used_nodes):
                    m2 = dict(mapping); m2[es.tgt] = e.tgt
                    u2 = set(used_nodes); u2.add(e.tgt)
                    ok, mm = _backtrack_assign(G, sk, m2, u2)
                    if ok:
                        return True, mm

        if es.tgt in mapping and es.src not in mapping:
            tgt_node = mapping[es.tgt]
            for e in G.edges:
                if (e.tgt == tgt_node and e.rel == es.rel
                        and e.src not in used_nodes):
                    m2 = dict(mapping); m2[es.src] = e.src
                    u2 = set(used_nodes); u2.add(e.src)
                    ok, mm = _backtrack_assign(G, sk, m2, u2)
                    if ok:
                        return True, mm

    # Fallback: assign an unfilled placeholder arbitrarily
    remaining = [p for p in sk.placeholders() if p not in mapping]
    if not remaining:
        return False, mapping
    for nid in G.nodes.keys():
        if nid in used_nodes:
            continue
        m2 = dict(mapping); m2[remaining[0]] = nid
        u2 = set(used_nodes); u2.add(nid)
        ok, mm = _backtrack_assign(G, sk, m2, u2)
        if ok:
            return True, mm
    return False, mapping


def _check_edges(G: InferentialGraph,
                 sk: Skeleton,
                 m: Dict[str, str]) -> bool:
    edge_set = {(e.src, e.rel, e.tgt) for e in G.edges}
    for es in sk.structure:
        a, b = m.get(es.src), m.get(es.tgt)
        if a is None or b is None:
            return False
        if (a, es.rel, b) not in edge_set:
            return False
    return True


# =============================================================================
# 3.4  Difficulty Controller
# =============================================================================

class DifficultyController:
    def __init__(self, skeletons: List[Skeleton]):
        self.skeletons  = skeletons
        self.size_index: Dict[int, List[str]] = {}
        for sk in skeletons:
            self.size_index.setdefault(sk.size, []).append(sk.id)

    def difficulty_to_skeletons(self, dl: int) -> List[str]:
        return self.size_index.get(dl, [])

    def enforce_hop_count(self, subgraph: Subgraph,
                          dl: int, tolerance: int = 0) -> bool:
        return abs(hop_count(subgraph) - dl) <= tolerance
