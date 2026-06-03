"""Inferensi model dari folder model/ (hasil Kaggle)."""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

from config import CLASS_LABELS, MODEL_DIR
from services.preprocess import build_cluster_row, build_regression_row


def models_ready() -> bool:
    required = [
        "prediction.pkl",
        "classification.pkl",
        "clustering.pkl",
        "cluster_scaler.pkl",
    ]
    return all((MODEL_DIR / f).exists() for f in required)


@lru_cache(maxsize=1)
def _load(name: str):
    path = MODEL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Model tidak ada: {path}. Extract uas_artifacts.zip ke folder model/.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _regression_features(
    square_feet: float,
    bedrooms: int,
    bathrooms: int,
    year_built: int,
    neighborhood: str,
):
    return build_regression_row(
        square_feet, bedrooms, bathrooms, year_built, neighborhood
    )


def predict_regression(
    square_feet: float,
    bedrooms: int,
    bathrooms: int,
    year_built: int,
    neighborhood: str,
) -> dict:
    features = _regression_features(
        square_feet, bedrooms, bathrooms, year_built, neighborhood
    )
    reg = _load("prediction.pkl")
    price_pred = float(reg.predict(features)[0])
    return {"price_pred": price_pred}


def predict_classification(
    square_feet: float,
    bedrooms: int,
    bathrooms: int,
    year_built: int,
    neighborhood: str,
) -> dict:
    features = _regression_features(
        square_feet, bedrooms, bathrooms, year_built, neighborhood
    )
    clf = _load("classification.pkl")
    pred_cls = int(clf.predict(features)[0])
    label = CLASS_LABELS.get(pred_cls, str(pred_cls))
    proba = None
    if hasattr(clf, "predict_proba"):
        proba = float(clf.predict_proba(features)[0][pred_cls])
    return {"category": label, "proba": proba, "class_id": pred_cls}


def predict_cluster(
    square_feet: float,
    bedrooms: int,
    bathrooms: int,
    year_built: int,
    neighborhood: str,
    price: float,
) -> dict:
    if price is None or price <= 0:
        raise ValueError("Harga wajib diisi dan harus lebih dari 0.")
    cluster_price = float(price)
    kmeans = _load("clustering.pkl")
    scaler = _load("cluster_scaler.pkl")
    X = build_cluster_row(
        square_feet, bedrooms, bathrooms, year_built, neighborhood, cluster_price
    )
    cluster_id = int(kmeans.predict(scaler.transform(X.values))[0])
    return {
        "cluster_id": cluster_id,
        "cluster_price": cluster_price,
    }


def run_analysis(
    square_feet: float,
    bedrooms: int,
    bathrooms: int,
    year_built: int,
    neighborhood: str,
    use_predicted_price: bool = True,
    manual_price: float | None = None,
) -> dict:
    features = _regression_features(
        square_feet, bedrooms, bathrooms, year_built, neighborhood
    )
    reg_out = predict_regression(
        square_feet, bedrooms, bathrooms, year_built, neighborhood
    )
    clf_out = predict_classification(
        square_feet, bedrooms, bathrooms, year_built, neighborhood
    )
    cluster_price = (
        manual_price if manual_price and manual_price > 0 else reg_out["price_pred"]
    )
    cl_out = predict_cluster(
        square_feet,
        bedrooms,
        bathrooms,
        year_built,
        neighborhood,
        price=cluster_price,
    )
    return {
        "price_pred": reg_out["price_pred"],
        "category": clf_out["category"],
        "proba": clf_out["proba"],
        "cluster_id": cl_out["cluster_id"],
        "cluster_price": cl_out["cluster_price"],
        "features": features.to_dict(orient="records")[0],
    }
