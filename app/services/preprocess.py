"""Feature engineering selaras notebook Kaggle."""

from __future__ import annotations

import pandas as pd

from config import (
    CLUSTER_FEATURES,
    CURRENT_YEAR,
    NEIGHBORHOOD_DUMMY_COLS,
    REGRESSION_FEATURES,
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["HouseAge"] = CURRENT_YEAR - out["YearBuilt"]
    out["AreaPerBedroom"] = out["SquareFeet"] / (out["Bedrooms"] + 1)
    out["AreaPerBathroom"] = out["SquareFeet"] / (out["Bathrooms"] + 1)
    out["BathPerBedroom"] = out["Bathrooms"] / (out["Bedrooms"] + 1)
    out = pd.get_dummies(out, columns=["Neighborhood"], drop_first=True)
    return out


def build_regression_row(
    square_feet: float,
    bedrooms: int,
    bathrooms: int,
    year_built: int,
    neighborhood: str,
) -> pd.DataFrame:
    row = pd.DataFrame(
        [
            {
                "SquareFeet": square_feet,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "YearBuilt": year_built,
                "Neighborhood": neighborhood,
            }
        ]
    )
    df = engineer_features(row)
    for col in NEIGHBORHOOD_DUMMY_COLS:
        if col not in df.columns:
            df[col] = 0
    return df[REGRESSION_FEATURES]


def build_cluster_row(
    square_feet: float,
    bedrooms: int,
    bathrooms: int,
    year_built: int,
    neighborhood: str,
    price: float,
) -> pd.DataFrame:
    row_df = build_regression_row(
        square_feet, bedrooms, bathrooms, year_built, neighborhood
    )
    row_df = row_df.copy()
    row_df["Price"] = price
    return row_df[CLUSTER_FEATURES]
