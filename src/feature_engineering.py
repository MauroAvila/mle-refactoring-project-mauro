"""King County Feature Engineering Module.

This module groups pure functions responsible for creating new predictive
variables from the real estate dataset, calculating relative value metrics,
and computing geographic distances adjusted for Earth's curvature.
"""

import os
import numpy as np
import pandas as pd

# Default input and output paths
INPUT_DATASET_PATH = r"C:\SD-resp\Weiterbildung2026\AI_eng\Module_1\mle-refactoring-project\king_county_project\data\King_County_House_prices_cleaned.csv"
OUTPUT_DIR_PATH = r"C:\SD-resp\Weiterbildung2026\AI_eng\Module_1\mle-refactoring-project\king_county_project\data"
OUTPUT_DATASET_PATH = os.path.join(
    OUTPUT_DIR_PATH, "King_County_House_prices_featured.csv"
)


def load_dataset(file_path: str = INPUT_DATASET_PATH) -> pd.DataFrame:
    """Loads the cleaned King County dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file. Defaults to INPUT_DATASET_PATH.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(file_path)


def calculate_sqft_price(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Calculates the relative price per square foot of total property size.

    Business Rule: Divides sale price ('price') by the sum of interior living area
    ('sqft_living') and total lot area ('sqft_lot').

    Args:
        data_frame (pd.DataFrame): Input DataFrame with 'price', 'sqft_living', and 'sqft_lot'.

    Returns:
        pd.DataFrame: DataFrame copy containing the new 'sqft_price' column.
    """
    featured_df = data_frame.copy()
    total_area = featured_df["sqft_living"] + featured_df["sqft_lot"]
    featured_df["sqft_price"] = (featured_df["price"] / total_area).round(2)
    return featured_df


def calculate_center_distance(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Calculates geographic distance in km to the wealth center in Medina.

    Business Rule: Uses Bill Gates' Medina residence coordinates (lat 47.62774, long -122.24194)
    as reference point. Applies a cosine latitude adjustment to longitude differential
    to compensate for Earth's curvature deformation over a mean radius of 6,378 km.

    Args:
        data_frame (pd.DataFrame): Input DataFrame containing 'lat' and 'long' columns.

    Returns:
        pd.DataFrame: DataFrame copy containing the new 'center_distance' column.
    """
    featured_df = data_frame.copy()

    # Reference geographic coordinates (Medina, WA)
    ref_lat = 47.62774
    ref_long = -122.24194
    earth_radius_km = 6378.0

    # Absolute coordinate differences
    delta_lat = np.abs(ref_lat - featured_df["lat"])
    delta_long = np.abs(ref_long - featured_df["long"])

    # Longitude correction using cosine of reference latitude in radians
    corrected_delta_long = delta_long * np.cos(np.radians(ref_lat))

    # Approximate geodesic distance calculation in kilometers
    distance_degrees = np.sqrt((corrected_delta_long**2) + (delta_lat**2))
    featured_df["center_distance"] = (
        distance_degrees * (2 * np.pi * earth_radius_km) / 360.0
    ).round(2)

    return featured_df


def calculate_water_distance(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Calculates minimum geodesic distance in km to the nearest waterfront property.

    Business Rule: Extracts coordinates of all verified waterfront properties (waterfront == 1)
    and computes minimum vector distance to the shoreline for each property in the dataset.

    Args:
        data_frame (pd.DataFrame): Input DataFrame containing 'lat', 'long', and 'waterfront'.

    Returns:
        pd.DataFrame: DataFrame copy containing the new 'water_distance' column.
    """
    featured_df = data_frame.copy()
    earth_radius_km = 6378.0

    # Extract coordinates of confirmed waterfront properties (waterfront == 1)
    waterfront_properties = featured_df[featured_df["waterfront"] == 1]

    if waterfront_properties.empty:
        featured_df["water_distance"] = 0.0
        return featured_df

    water_lats = np.radians(waterfront_properties["lat"].values)
    water_longs = np.radians(waterfront_properties["long"].values)

    df_lats = np.radians(featured_df["lat"].values)
    df_longs = np.radians(featured_df["long"].values)

    # Coordinate differences matrix between each property and waterfront points
    delta_lat = df_lats[:, np.newaxis] - water_lats
    delta_long = df_longs[:, np.newaxis] - water_longs

    # Apply cosine latitude correction for geodesic distance
    delta_long_corr = delta_long * np.cos(water_lats)
    distances_matrix = np.sqrt((delta_long_corr**2) + (delta_lat**2)) * earth_radius_km

    # Assign minimum distance found across waterfront points
    featured_df["water_distance"] = np.min(distances_matrix, axis=1).round(2)

    return featured_df


def export_featured_data(
    data_frame: pd.DataFrame, output_path: str = OUTPUT_DATASET_PATH
) -> None:
    """Exports the feature-engineered DataFrame to a CSV file in the designated folder.

    Args:
        data_frame (pd.DataFrame): Processed DataFrame to save.
        output_path (str): Destination file path. Defaults to OUTPUT_DATASET_PATH.
    """
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    data_frame.to_csv(output_path, index=False)
    print(f"File successfully saved to: {output_path}")


def process_feature_engineering(
    input_path: str = INPUT_DATASET_PATH,
    output_path: str = OUTPUT_DATASET_PATH,
) -> pd.DataFrame:
    """Executes full feature engineering pipeline on the King County dataset and saves output.

    Args:
        input_path (str): Input dataset path. Defaults to INPUT_DATASET_PATH.
        output_path (str): Output destination path. Defaults to OUTPUT_DATASET_PATH.

    Returns:
        pd.DataFrame: Feature-engineered DataFrame.
    """
    df_clean = load_dataset(input_path)
    df_featured = calculate_sqft_price(df_clean)
    df_featured = calculate_center_distance(df_featured)
    df_featured = calculate_water_distance(df_featured)

    export_featured_data(df_featured, output_path)
    return df_featured


if __name__ == "__main__":
    df_processed = process_feature_engineering()
    print(
        f"Feature engineering pipeline completed. Output shape: {df_processed.shape}"
    )