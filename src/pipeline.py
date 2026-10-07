"""King County Data Preprocessing Pipeline Orchestration Script.

This module orchestrates the deterministic linear sequence of data transformations,
chaining pure functions from the Data Cleaning module (data_cleaning) and the
Feature Engineering module (feature_engineering) over the raw dataset.
"""

from pathlib import Path
import pandas as pd

# Import sanitized and translated pure functions from src modules
from data_cleaning import (
    remove_outliers,
    rebuild_basement_sqft,
    impute_missing_values,
    consolidate_last_change_year,
)
from feature_engineering import (
    calculate_sqft_price,
    calculate_center_distance,
    calculate_water_distance,
)


def run_data_pipeline(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Executes the complete data preparation pipeline sequentially.

    Args:
        input_path (Path): Path to the raw input CSV file.
        output_path (Path): Path where the final processed CSV dataset will be saved.

    Returns:
        pd.DataFrame: Fully cleaned and feature-engineered DataFrame.
    """
    print(f"[1/4] Loading raw dataset from: {input_path}")
    df_raw = pd.read_csv(input_path)

    print("[2/4] Executing Data Cleaning Module...")
    df_clean_1 = remove_outliers(df_raw)
    df_clean_2 = rebuild_basement_sqft(df_clean_1)
    df_clean_3 = impute_missing_values(df_clean_2)
    df_cleaned = consolidate_last_change_year(df_clean_3)

    print("[3/4] Executing Feature Engineering Module...")
    df_feat_1 = calculate_sqft_price(df_cleaned)
    df_feat_2 = calculate_center_distance(df_feat_1)
    df_processed = calculate_water_distance(df_feat_2)

    print(f"[4/4] Exporting processed dataset to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)

    print("Pipeline execution completed successfully.")
    return df_processed


if __name__ == "__main__":
    # Project root directory resolved relative to this file
    # Path(__file__) -> src/pipeline.py
    # .parent        -> src/
    # .parent.parent -> king_county_project/ (Project Root)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Input path: Points to the raw dataset in the root data folder
    RAW_DATA_PATH = (
        PROJECT_ROOT.parent / "data" / "King_County_House_prices_dataset.csv"
    )
    
    # Output path: Points to the processed CSV inside the project data folder
    PROCESSED_DATA_PATH = (
        PROJECT_ROOT / "data" / "King_County_House_prices_pipeline_output.csv"
    )

    run_data_pipeline(input_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH)