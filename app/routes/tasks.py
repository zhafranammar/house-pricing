"""Halaman per task: regresi, klasifikasi, clustering (+ SHAP)."""

from flask import Blueprint, flash, render_template, request

from config import METRICS
from services.assets import asset_exists, load_csv_rows
from services.predict import (
    models_ready,
    predict_classification,
    predict_cluster,
    predict_regression,
)

bp = Blueprint("tasks", __name__)

NEIGHBORHOODS = ["Rural", "Suburb", "Urban"]

DEFAULT_FORM = {
    "square_feet": 2000,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 1995,
    "neighborhood": "Suburb",
    "manual_price": 250000.0,
}


def _parse_form() -> dict:
    return {
        "square_feet": float(request.form.get("square_feet", 2000)),
        "bedrooms": int(request.form.get("bedrooms", 3)),
        "bathrooms": int(request.form.get("bathrooms", 2)),
        "year_built": int(request.form.get("year_built", 1995)),
        "neighborhood": request.form.get("neighborhood", "Suburb"),
        "manual_price": float(request.form.get("manual_price") or 0),
    }


def _model_warning():
    if not models_ready():
        flash(
            "Model belum ada di folder model/. Extract uas_artifacts.zip (bagian model/) ke root proyek.",
            "warning",
        )


@bp.route("/regression", methods=["GET", "POST"])
def regression_page():
    form = dict(DEFAULT_FORM)
    result = None
    _model_warning()

    if request.method == "POST" and models_ready():
        try:
            form = _parse_form()
            result = predict_regression(
                form["square_feet"],
                form["bedrooms"],
                form["bathrooms"],
                form["year_built"],
                form["neighborhood"],
            )
        except Exception as exc:
            flash(f"Prediksi regresi gagal: {exc}", "danger")

    m = METRICS["regression"]
    return render_template(
        "regression.html",
        metrics=m,
        model_comparison=load_csv_rows("regression_results.csv"),
        shap_ready=asset_exists("shap_regression.png"),
        coef_ready=asset_exists("regression_coefficients.png"),
        form=form,
        result=result,
        neighborhoods=NEIGHBORHOODS,
        models_ready=models_ready(),
    )


@bp.route("/classification", methods=["GET", "POST"])
def classification_page():
    form = dict(DEFAULT_FORM)
    result = None
    _model_warning()

    if request.method == "POST" and models_ready():
        try:
            form = _parse_form()
            result = predict_classification(
                form["square_feet"],
                form["bedrooms"],
                form["bathrooms"],
                form["year_built"],
                form["neighborhood"],
            )
        except Exception as exc:
            flash(f"Prediksi klasifikasi gagal: {exc}", "danger")

    m = METRICS["classification"]
    return render_template(
        "classification.html",
        metrics=m,
        model_comparison=load_csv_rows("classification_results.csv"),
        shap_ready=asset_exists("shap_classification.png"),
        form=form,
        result=result,
        neighborhoods=NEIGHBORHOODS,
        models_ready=models_ready(),
    )


@bp.route("/clustering", methods=["GET", "POST"])
def clustering_page():
    form = dict(DEFAULT_FORM)
    result = None
    _model_warning()

    if request.method == "POST" and models_ready():
        try:
            form = _parse_form()
            if form["manual_price"] <= 0:
                raise ValueError("Harga wajib diisi untuk clustering.")
            result = predict_cluster(
                form["square_feet"],
                form["bedrooms"],
                form["bathrooms"],
                form["year_built"],
                form["neighborhood"],
                price=form["manual_price"],
            )
        except Exception as exc:
            flash(f"Prediksi clustering gagal: {exc}", "danger")

    m = METRICS["clustering"]
    profiles = load_csv_rows("cluster_profile.csv")
    return render_template(
        "clustering.html",
        metrics=m,
        cluster_profiles=profiles,
        scatter_ready=asset_exists("cluster_scatter.png"),
        form=form,
        result=result,
        neighborhoods=NEIGHBORHOODS,
        models_ready=models_ready(),
    )
