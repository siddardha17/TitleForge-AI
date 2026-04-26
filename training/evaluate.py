"""
TitleForge AI — Evaluation Script
Metrics: BLEU, ROUGE-1/2/L, Exact Match %, Character Error Rate (CER)
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from rouge_score import rouge_scorer
import sacrebleu
from jiwer import cer as compute_cer

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED_DIR = os.path.join(BASE_DIR, "output", "merged-model")
TEST_GT    = os.path.join(BASE_DIR, "data", "processed", "test_ground_truth.json")

MAX_NEW_TOKENS = 80

INFERENCE_TEMPLATE = (
    "### Instruction:\n"
    "Normalize this raw product title into a clean, standardized catalog title. "
    "Fix capitalization, expand abbreviations, remove noise, and ensure consistent formatting.\n\n"
    "### Input:\n{raw_title}\n\n"
    "### Response:\n"
)


def load_model():
    print(f"🔄 Loading merged model from: {MERGED_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.1,
    )
    return pipe


def run_inference(pipe, raw_title: str) -> str:
    prompt = INFERENCE_TEMPLATE.format(raw_title=raw_title.strip())
    output = pipe(prompt)[0]["generated_text"]
    # Extract the response portion only
    response = output.split("### Response:\n")[-1].strip()
    # Take only the first line (the title)
    return response.split("\n")[0].strip()


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """Compute all evaluation metrics."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    bleu_scores = []
    exact_matches = 0
    cer_scores = []

    for pred, ref in zip(predictions, references):
        # ROUGE
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

        # Sentence BLEU
        bleu = sacrebleu.sentence_bleu(pred, [ref])
        bleu_scores.append(bleu.score)

        # Exact match (case-insensitive)
        if pred.strip().lower() == ref.strip().lower():
            exact_matches += 1

        # CER
        cer_scores.append(compute_cer(ref, pred))

    n = len(predictions)
    return {
        "bleu":         round(sum(bleu_scores) / n, 2),
        "rouge1":       round(sum(rouge1_scores) / n, 4),
        "rouge2":       round(sum(rouge2_scores) / n, 4),
        "rougeL":       round(sum(rougeL_scores) / n, 4),
        "exact_match":  round(exact_matches / n * 100, 2),
        "cer":          round(sum(cer_scores) / n, 4),
        "num_samples":  n,
    }


def print_results(metrics: dict, predictions: list[str], references: list[str], raw_titles: list[str]):
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated : {metrics['num_samples']}")
    print(f"  BLEU              : {metrics['bleu']:.2f}")
    print(f"  ROUGE-1           : {metrics['rouge1']:.4f}")
    print(f"  ROUGE-2           : {metrics['rouge2']:.4f}")
    print(f"  ROUGE-L           : {metrics['rougeL']:.4f}")
    print(f"  Exact Match       : {metrics['exact_match']:.2f}%")
    print(f"  Char Error Rate   : {metrics['cer']:.4f}")
    print("\n" + "─" * 60)
    print("🔍 SAMPLE PREDICTIONS")
    print("─" * 60)
    for i in range(min(5, len(predictions))):
        print(f"\n  Raw   : {raw_titles[i]}")
        print(f"  Pred  : {predictions[i]}")
        print(f"  Gold  : {references[i]}")


def save_results(metrics: dict, output_path: str = None):
    if output_path is None:
        output_path = os.path.join(BASE_DIR, "output", "eval_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Results saved → {output_path}")


def main():
    print("=" * 60)
    print("🧪 TitleForge AI — Model Evaluation")
    print("=" * 60)

    # 1. Load test ground truth
    with open(TEST_GT, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    raw_titles  = [d["raw_title"]   for d in test_data]
    gold_titles = [d["clean_title"] for d in test_data]

    print(f"📋 Evaluating on {len(test_data)} test samples...")

    # 2. Load model
    pipe = load_model()

    # 3. Run inference
    predictions = []
    for i, raw in enumerate(raw_titles):
        pred = run_inference(pipe, raw)
        predictions.append(pred)
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(raw_titles)}")

    # 4. Compute metrics
    metrics = compute_metrics(predictions, gold_titles)

    # 5. Display & save
    print_results(metrics, predictions, gold_titles, raw_titles)
    save_results(metrics)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
