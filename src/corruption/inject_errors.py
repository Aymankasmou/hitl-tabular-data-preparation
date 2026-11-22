
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "clean" / "train_clean_reference.csv"
OUTPUT_DIR = ROOT / "data" / "corrupted"

TARGET = "y"

SEEDS = [42, 123, 2026]
CORRUPTION_RATE = 0.10

ERROR_TYPES = [
    "missing_value",
    "numerical_shift",
    "categorical_typo",
    "format_inconsistency"
]

# --------------------------------------------------
# 2. Load clean training data
# --------------------------------------------------

clean_df = pd.read_csv(INPUT_FILE)

feature_cols = [
    col for col in clean_df.columns if col != TARGET
]

numeric_cols = [
    "age", "balance", "day",
    "campaign", "pdays", "previous"
]

categorical_cols = [
    col for col in feature_cols
    if col not in numeric_cols
]

assert len(feature_cols) == 15
assert len(clean_df) == 4000

total_cells = len(clean_df) * len(feature_cols)
total_errors = int(total_cells * CORRUPTION_RATE)

errors_per_type = total_errors // len(ERROR_TYPES)

print("========== CORRUPTION CONFIGURATION ==========")
print("Training rows:", len(clean_df))
print("Feature columns:", len(feature_cols))
print("Total feature cells:", total_cells)
print("Target corrupted cells:", total_errors)
print("Errors per type:", errors_per_type)

# --------------------------------------------------
# 3. Select unique cells
# --------------------------------------------------

def select_cells(columns, count, rng, used_cells):

    candidates = [
        (row, col)
        for col in columns
        for row in range(len(clean_df))
        if (row, col) not in used_cells
    ]

    if count > len(candidates):
        raise ValueError("Not enough eligible cells.")

    indices = rng.choice(
        len(candidates),
        size=count,
        replace=False
    )

    selected = [
        candidates[int(i)] for i in indices
    ]

    used_cells.update(selected)

    return selected

# --------------------------------------------------
# 4. Corruption transformations
# --------------------------------------------------

def make_typo(value, rng):

    value = str(value)

    if len(value) < 4:
        return value + "x"

    position = int(rng.integers(0, len(value)))

    return value[:position] + value[position + 1:]


def shift_numeric(value, column, rng):

    std = clean_df[column].std()

    shift = max(10, round(3 * std))

    direction = int(rng.choice([-1, 1]))

    return float(value) + direction * shift


# --------------------------------------------------
# 5. Run experiments
# --------------------------------------------------

for seed in SEEDS:

    print(f"\nRunning corruption seed: {seed}")

    rng = np.random.default_rng(seed)

    corrupted_df = clean_df.copy(deep=True)

    used_cells = set()
    mask_records = []

    def record_and_change(row, col, error_type, new_value):

        original_value = clean_df.at[row, col]

        corrupted_df.at[row, col] = new_value

        mask_records.append({
            "row_index": row,
            "column": col,
            "error_type": error_type,
            "original_value": str(original_value),
            "corrupted_value": (
                "" if pd.isna(new_value) else str(new_value)
            )
        })

    # A. Missing values
    cells = select_cells(
        feature_cols,
        errors_per_type,
        rng,
        used_cells
    )

    for row, col in cells:
        record_and_change(
            row, col, "missing_value", np.nan
        )

    # B. Numerical shifts
    cells = select_cells(
        numeric_cols,
        errors_per_type,
        rng,
        used_cells
    )

    for row, col in cells:

        original = clean_df.at[row, col]

        new_value = shift_numeric(
            original, col, rng
        )

        record_and_change(
            row, col, "numerical_shift", new_value
        )

    # C. Categorical typos
    cells = select_cells(
        categorical_cols,
        errors_per_type,
        rng,
        used_cells
    )

    for row, col in cells:

        original = clean_df.at[row, col]

        new_value = make_typo(original, rng)

        record_and_change(
            row, col, "categorical_typo", new_value
        )

    # D. Format inconsistencies
    cells = select_cells(
        categorical_cols,
        errors_per_type,
        rng,
        used_cells
    )

    for row, col in cells:

        original = clean_df.at[row, col]

        new_value = f" {str(original).upper()} "

        record_and_change(
            row, col, "format_inconsistency", new_value
        )

    # --------------------------------------------------
    # 6. Save corrupted data and ground-truth mask
    # --------------------------------------------------

    seed_dir = OUTPUT_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    mask_df = pd.DataFrame(mask_records)

    corrupted_df.to_csv(
        seed_dir / "corrupted_train.csv",
        index=False
    )

    mask_df.to_csv(
        seed_dir / "ground_truth_mask.csv",
        index=False
    )

    # --------------------------------------------------
    # 7. Validate experiment
    # --------------------------------------------------

    assert len(mask_df) == total_errors

    assert not mask_df.duplicated(
        subset=["row_index", "column"]
    ).any()

    assert corrupted_df[TARGET].equals(
        clean_df[TARGET]
    )

    assert corrupted_df.shape == clean_df.shape

    print("Corrupted cells:", len(mask_df))

    print("\nError distribution:")
    print(mask_df["error_type"].value_counts())

    print("Saved:", seed_dir)

print("\nAll corruption experiments completed.")
