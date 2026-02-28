import os
import json
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

BASE_PATH = r"C:\Users\sanke\OneDrive\Desktop\commonsense_qg"

RESULT_FILES = [
    "results_cosmosqa.jsonl",
    "results_mcscript.jsonl",
    "results_kqapro.jsonl",
    "results_grailqa.jsonl"
]


def evaluate_results(results_path):

    bleu_scores = []
    meteor_scores = []
    rouge_l_scores = []

    references = []
    hypotheses = []

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1

    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)

            ref = row.get('original_question')
            hyp = row.get('generated_question')

            # Skip invalid rows
            if not ref or not hyp:
                continue

            references.append(ref)
            hypotheses.append(hyp)

            ref_tokens = ref.lower().split()
            hyp_tokens = hyp.lower().split()

            # BLEU-4
            bleu = sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smooth
            )
            bleu_scores.append(bleu)

            # METEOR
            meteor = meteor_score([ref_tokens], hyp_tokens)
            meteor_scores.append(meteor)

            # ROUGE-L
            rouge = scorer.score(ref, hyp)
            rouge_l_scores.append(rouge['rougeL'].fmeasure)

    if len(bleu_scores) == 0:
        print("No valid reference-hypothesis pairs found.")
        return

    # ---- BERTScore ----
    print("Computing BERTScore...")
    P, R, F1 = bert_score(
        hypotheses,
        references,
        lang="en",
        model_type="roberta-large",
        batch_size=16,
        verbose=False
    )

    print(f"Samples evaluated: {len(bleu_scores)}")
    print(f"BLEU-4   : {sum(bleu_scores)/len(bleu_scores):.4f}")
    print(f"METEOR   : {sum(meteor_scores)/len(meteor_scores):.4f}")
    print(f"ROUGE-L  : {sum(rouge_l_scores)/len(rouge_l_scores):.4f}")
    print(f"BERTScore: {F1.mean().item():.4f}")


def evaluate_all():
    for file in RESULT_FILES:
        path = os.path.join(BASE_PATH, file)

        if not os.path.exists(path):
            print(f"\n{file} not found, skipping.")
            continue

        print("\n====================================")
        print(f"Evaluating: {file}")
        print("====================================")

        evaluate_results(path)


if __name__ == "__main__":
    evaluate_all()