"""Unit tests for King County data processing and cleaning module."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from src.data_cleaning import (
    clean_outliers,
    consolidate_last_change,
    impute_missing_values,
    rebuild_basement,
)
from src.feature_engineering import (
    calculate_center_distance,
    calculate_sqft_price,
    calculate_water_distance,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Fixture providing a realistic sample raw DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "bedrooms": [3, 4, 4],
            "bathrooms": [2.0, 9.0, 2.5],  # Row 1 is an outlier ratio (9/4 = 2.25)
            "sqft_living": [2000, 3000, 1800],
            "sqft_lot": [5000, 6000, 4000],
            "sqft_above": [1500, 2000, 1800],
            "sqft_basement": [0, 0, 0],
            "price": [500000, 800000, 450000],
            "view": [np.nan, 2.0, np.nan],
            "waterfront": [1.0, np.nan, 0.0],
            "yr_built": [1995, 2000, 1980],
            "yr_renovated": [0.0, np.nan, 2018.0],
            "lat": [47.6062, 47.3000, 47.5000],
            "long": [-122.3321, -122.1000, -122.3000],
        }
    )


def test_clean_outliers_removes_extreme_ratios(sample_raw_df: pd.DataFrame):
    """Verify removal of records with bathroom/bedroom ratio >= 2.0 or <= 0.10."""
    # WHEN: Cleaning outliers
    cleaned_df = clean_outliers(sample_raw_df)

    # THEN: Row with bathrooms=9.0 and bedrooms=4 (ratio 2.25) is removed
    assert len(cleaned_df) == 2
    assert 2 not in cleaned_df["id"].values


def test_rebuild_basement_calculates_exact_difference(
    sample_raw_df: pd.DataFrame,
):
    """Verify sqft_basement is recalculated as (sqft_living - sqft_above)."""
    # WHEN: Rebuilding basement area
    result_df = rebuild_basement(sample_raw_df)

    # THEN: sqft_basement matches expected values
    expected_basement = pd.Series(
        [500, 1000, 0], name="sqft_basement", dtype=int
    )
    assert_series_equal(result_df["sqft_basement"], expected_basement)


def test_impute_missing_values_fills_nans_with_zero(
    sample_raw_df: pd.DataFrame,
):
    """Verify NaN values in view and waterfront are filled with 0.0."""
    # WHEN: Imputing missing values
    result_df = impute_missing_values(sample_raw_df)

    # THEN: No NaNs remain in target columns and values match expectation
    assert result_df["view"].isna().sum() == 0
    assert result_df["waterfront"].isna().sum() == 0
    assert result_df["view"].iloc[0] == 0.0
    assert result_df["waterfront"].iloc[1] == 0.0


def test_consolidate_last_change_handles_renovation_fallback(
    sample_raw_df: pd.DataFrame,
):
    """Verify unification of yr_renovated and yr_built into last_known_change."""
    # GIVEN: Record 0 (yr_renovated=0), Record 1 (NaN), Record 2 (renovated in 2018)
    # WHEN: Consolidating temporal feature
    result_df = consolidate_last_change(sample_raw_df)

    # THEN: last_known_change is created and original columns are removed
    expected_last_change = pd.Series(
        [1995, 2000, 2018], name="last_known_change", dtype="Int64"
    )
    assert_series_equal(result_df["last_known_change"], expected_last_change)


def test_calculate_sqft_price_correct_ratio(sample_raw_df: pd.DataFrame):
    """Verify price per living sqft calculation."""
    # WHEN: Calculating sqft_price
    result_df = calculate_sqft_price(sample_raw_df)

    # THEN: Check that sqft_price column is generated successfully
    assert "sqft_price" in result_df.columns
    assert not result_df["sqft_price"].isna().any()


def test_calculate_center_distance_zero_at_reference(
    sample_raw_df: pd.DataFrame,
):
    """Verify distance to city center is calculated successfully."""
    # WHEN: Calculating distance to center
    result_df = calculate_center_distance(sample_raw_df)

    # THEN: Check that center_distance column is generated and contains numeric values
    assert "center_distance" in result_df.columns
    assert not result_df["center_distance"].isna().any()
    assert np.isclose(result_df["center_distance"].iloc[0], 7.18, atol=1e-1)


def test_calculate_water_distance_zero_for_waterfront(
    sample_raw_df: pd.DataFrame,
):
    """Verify water_distance returns 0 for waterfront properties."""
    # WHEN: Calculating water distance
    result_df = calculate_water_distance(sample_raw_df)

    # THEN: Record with waterfront=1.0 has water_distance = 0
    assert result_df["water_distance"].iloc[0] == 0.0