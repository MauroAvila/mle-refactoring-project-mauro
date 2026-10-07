"""King County Data Cleaning and Preparation Module.

This module contains pure functions responsible for sanitizing the King County
real estate dataset by removing outliers, recalculating corrupted variables,
imputing missing values, and unifying temporal information.
"""

import os
import numpy as np
import pandas as pd

# Default input and output file paths
INPUT_DATASET_PATH = r"C:\SD-resp\Weiterbildung2026\AI_eng\Module_1\mle-refactoring-project\data\King_County_House_prices_dataset.csv"
OUTPUT_DIR_PATH = r"C:\SD-resp\Weiterbildung2026\AI_eng\Module_1\mle-refactoring-project\king_county_project\data"
OUTPUT_DATASET_PATH = os.path.join(
    OUTPUT_DIR_PATH, "King_County_House_prices_cleaned.csv"
)


def load_dataset(file_path: str = INPUT_DATASET_PATH) -> pd.DataFrame:
    """Loads the King County house prices dataset from a CSV file."""
    return pd.read_csv(file_path)


def remove_outliers(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Removes outlier records based on the bathroom-to-bedroom ratio.

    Business Rule: Properties with extreme bathroom-to-bedroom ratios are discarded:
    greater than or equal to 2.0 or less than or equal to 0.10.
    """
    cleaned_df = data_frame.copy()

    # Safely compute ratio avoiding DivisionByZero for 0-bedroom properties
    bedroom_series = cleaned_df["bedrooms"].replace(0, np.nan)
    bathroom_bedroom_ratio = cleaned_df["bathrooms"] / bedroom_series

    valid_ratio_mask = (bathroom_bedroom_ratio < 2.0) & (
        bathroom_bedroom_ratio > 0.10
    )
    return cleaned_df[valid_ratio_mask].reset_index(drop=True)


def rebuild_basement_sqft(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Mathematically recalculates the basement area column (sqft_basement).

    Business Rule: Basement area is recalculated by subtracting the above-ground living
    area (sqft_above) from total living area (sqft_living).
    """
    cleaned_df = data_frame.copy()
    cleaned_df["sqft_basement"] = (
        cleaned_df["sqft_living"] - cleaned_df["sqft_above"]
    )
    return cleaned_df


def impute_missing_values(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing values (NaN) in 'view' and 'waterfront' features."""
    cleaned_df = data_frame.copy()
    cleaned_df["view"] = cleaned_df["view"].fillna(0)
    cleaned_df["waterfront"] = cleaned_df["waterfront"].fillna(0)
    return cleaned_df


def consolidate_last_change_year(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Consolidates property modification year into 'last_known_change'."""
    cleaned_df = data_frame.copy()

    no_renovation_mask = (cleaned_df["yr_renovated"].isna()) | (
        cleaned_df["yr_renovated"] == 0.0
    )

    cleaned_df["last_known_change"] = np.where(
        no_renovation_mask, cleaned_df["yr_built"], cleaned_df["yr_renovated"]
    )

    # Convert safely using pandas nullable integer type
    cleaned_df["last_known_change"] = cleaned_df["last_known_change"].astype(
        "Int64"
    )

    cleaned_df = cleaned_df.drop(columns=["yr_renovated", "yr_built"])
    return cleaned_df


# --- ALIASES DE COMPATIBILIDAD CON LA SUITE DE PRUEBAS ---
clean_outliers = remove_outliers
rebuild_basement = rebuild_basement_sqft
consolidate_last_change = consolidate_last_change_year


def export_cleaned_data(
    data_frame: pd.DataFrame, output_path: str = OUTPUT_DATASET_PATH
) -> None:
    """Exports the cleaned DataFrame to a CSV file in the designated output folder."""
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    data_frame.to_csv(output_path, index=False)
    print(f"File successfully saved to: {output_path}")


def process_king_county_data(
    input_path: str = INPUT_DATASET_PATH,
    output_path: str = OUTPUT_DATASET_PATH,
) -> pd.DataFrame:
    """Executes the full cleaning pipeline on the King County dataset and saves the result."""
    df_raw = load_dataset(input_path)
    df_clean = remove_outliers(df_raw)
    df_clean = rebuild_basement_sqft(df_clean)
    df_clean = impute_missing_values(df_clean)
    df_clean = consolidate_last_change_year(df_clean)

    export_cleaned_data(df_clean, output_path)
    return df_clean


if __name__ == "__main__":
    df_processed = process_king_county_data()
    print(
        f"Pipeline completed successfully. Processed dataset shape: {df_processed.shape}"
    )