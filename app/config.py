"""Konfigurasi proyek — sesuaikan identitas kelompok."""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "housing_price_dataset.csv"
MODEL_DIR = ROOT_DIR / "model"
ASSETS_DIR = Path(__file__).resolve().parent / "static" / "assets"
METRICS_JSON = ASSETS_DIR / "metrics_summary.json"

PROJECT_TITLE = "Housing Price Intelligence"
PROJECT_SUBTITLE = "Eksplorasi & Prediksi Harga Rumah"

PROJECT_DESCRIPTION = (
    "Dashboard interaktif dataset harga rumah dan integrasi model hasil training Kaggle: "
    "regresi harga, klasifikasi High/Low, serta clustering K-Means."
)

TEAM_MEMBERS = [
    {"nama": "MUHAMMAD ZHAFRAN AMMAR", "nim": "25051905009", "peran": "Modelling, Web App"},
    {"nama": "RIVALDI RAMADISYA PRAKOSO", "nim": "25051905017", "peran": "Preprocessing & EDA, Modelling"}
   ]

COURSE_INFO = {
    "mata_kuliah": "Data Mining",
    "program": "S2 — Informatika",
    "tahun_akademik": "2025/2026",
    "dosen": "Dr. Wiyli Yustanti, S.Si., M.Kom",
}

CURRENT_YEAR = 2026
LOCATION_MAP = {"Rural": 0, "Suburb": 1, "Urban": 2}

REGRESSION_FEATURE_BASE = [
    "SquareFeet",
    "Bedrooms",
    "Bathrooms",
    "YearBuilt",
    "HouseAge",
    "AreaPerBedroom",
    "AreaPerBathroom",
    "BathPerBedroom",
]
NEIGHBORHOOD_DUMMY_COLS = ["Neighborhood_Suburb", "Neighborhood_Urban"]
REGRESSION_FEATURES = REGRESSION_FEATURE_BASE + NEIGHBORHOOD_DUMMY_COLS

CLUSTER_FEATURE_BASE = [
    "SquareFeet",
    "Bedrooms",
    "Bathrooms",
    "HouseAge",
    "AreaPerBedroom",
    "AreaPerBathroom",
    "BathPerBedroom",
]
CLUSTER_FEATURES = CLUSTER_FEATURE_BASE + ["Price"] + NEIGHBORHOOD_DUMMY_COLS

CLASS_LABELS = {0: "High", 1: "Low"}


def load_metrics() -> dict:
    if METRICS_JSON.exists():
        with open(METRICS_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {
        "regression": {"model": "—", "r2": 0, "mae": 0, "rmse": 0},
        "classification": {"model": "—", "accuracy": 0, "f1": 0},
        "clustering": {"model": "K-Means", "n_clusters": 2, "silhouette": 0},
    }


METRICS = load_metrics()

MODEL_ASSETS = [
    "shap_regression.png",
    "shap_classification.png",
    "regression_coefficients.png",
    "cluster_scatter.png",
    "correlation_heatmap.png",
]
