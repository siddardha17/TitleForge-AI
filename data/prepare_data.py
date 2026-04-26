"""
TitleForge AI - Data Preparation Script
Loads the CSV dataset, formats it into instruction-style prompts,
and splits into train/validation/test sets.
"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(os.path.dirname(__file__))
OUTPUT_DIR    = os.path.join(DATA_DIR, "processed")
DATASET_FILE  = os.path.join(DATA_DIR, "sample_dataset.csv")

TRAIN_RATIO   = 0.80
VAL_RATIO     = 0.10
TEST_RATIO    = 0.10
RANDOM_SEED   = 42

# ── Prompt template ────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Normalize this raw product title into a clean, standardized catalog title. "
    "Fix capitalization, expand abbreviations, remove noise, and ensure consistent formatting.\n\n"
    "### Input:\n{raw_title}\n\n"
    "### Response:\n{clean_title}"
)

INFERENCE_TEMPLATE = (
    "### Instruction:\n"
    "Normalize this raw product title into a clean, standardized catalog title. "
    "Fix capitalization, expand abbreviations, remove noise, and ensure consistent formatting.\n\n"
    "### Input:\n{raw_title}\n\n"
    "### Response:\n"
)


def build_prompt(row: pd.Series, include_response: bool = True) -> str:
    """Build an instruction-style prompt from a dataset row."""
    if include_response:
        return PROMPT_TEMPLATE.format(
            raw_title=row["raw_title"].strip(),
            clean_title=row["clean_title"].strip(),
        )
    return INFERENCE_TEMPLATE.format(raw_title=row["raw_title"].strip())


def load_and_validate(filepath: str) -> pd.DataFrame:
    """Load CSV and perform basic validation."""
    df = pd.read_csv(filepath)

    # Ensure required columns exist
    required = {"raw_title", "clean_title"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns: {required}. Found: {set(df.columns)}"
        )

    # Drop rows with missing values
    before = len(df)
    df = df.dropna(subset=["raw_title", "clean_title"])
    df = df[df["raw_title"].str.strip() != ""]
    df = df[df["clean_title"].str.strip() != ""]
    after = len(df)

    if before != after:
        print(f"[WARN] Dropped {before - after} invalid rows.")

    print(f"[OK] Loaded {after} valid title pairs.")
    return df.reset_index(drop=True)


def split_dataset(df: pd.DataFrame):
    """Split dataframe into train / val / test splits."""
    train_df, temp_df = train_test_split(
        df, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED
    )
    print(
        f"[INFO] Split sizes -> train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}"
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def to_records(df: pd.DataFrame, include_response: bool = True) -> list[dict]:
    """Convert a DataFrame split into a list of prompt records."""
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "text": build_prompt(row, include_response=include_response),
                "raw_title": row["raw_title"].strip(),
                "clean_title": row["clean_title"].strip(),
            }
        )
    return records


def save_split(records: list[dict], name: str) -> None:
    """Save a split to a JSONL file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[SAVE] Saved {len(records)} records -> {out_path}")


def print_examples(records: list[dict], n: int = 2) -> None:
    """Print a few prompt examples for visual inspection."""
    print("\n" + "=" * 70)
    print("SAMPLE PROMPTS (Training format)")
    print("=" * 70)

    for i, rec in enumerate(records[:n]):
        print(f"\n[Example {i + 1}]\n{rec['text']}\n{'-' * 40}")


def main():
    print("TitleForge AI -- Data Preparation")
    print("=" * 40)

    # 1. Load & validate
    df = load_and_validate(DATASET_FILE)

    # 2. Split
    train_df, val_df, test_df = split_dataset(df)

    # 3. Format into prompt records
    train_records = to_records(train_df, include_response=True)
    val_records   = to_records(val_df,   include_response=True)
    test_records  = to_records(test_df,  include_response=False)  # No answer leaked

    # 4. Save
    save_split(train_records, "train")
    save_split(val_records,   "val")
    save_split(test_records,  "test")

    # 5. Save test ground truth separately for evaluation
    test_gt_path = os.path.join(OUTPUT_DIR, "test_ground_truth.json")
    gt = [{"raw_title": r["raw_title"], "clean_title": r["clean_title"]} for r in to_records(test_df, include_response=True)]
    with open(test_gt_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] Saved test ground truth -> {test_gt_path}")

    # 6. Preview
    print_examples(train_records)
    print("\n[DONE] Data preparation complete!")


if __name__ == "__main__":
    main()
