"""Halaman prediksi gabungan — 3 tab form + penjelasan fitur."""

from flask import Blueprint, flash, jsonify, render_template, request

from config import METRICS
from services.assets import asset_exists
from services.explain import build_explanations
from services.predict import (
    models_ready,
    predict_classification,
    predict_cluster,
    predict_regression,
)

bp = Blueprint("predict", __name__)

NEIGHBORHOODS = ["Rural", "Suburb", "Urban"]
VALID_TASKS = ("regression", "classification", "clustering")

DEFAULT_FORM = {
    "square_feet": 2000,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 1995,
    "neighborhood": "Suburb",
    "manual_price": 250000.0,
}


SHAP_BY_TASK = {
    "regression": "shap_regression.png",
    "classification": "shap_classification.png",
}
CLUSTER_VIZ = "cluster_scatter.png"


def _parse_form() -> dict:
    return {
        "square_feet": float(request.form.get("square_feet", 2000)),
        "bedrooms": int(request.form.get("bedrooms", 3)),
        "bathrooms": int(request.form.get("bathrooms", 2)),
        "year_built": int(request.form.get("year_built", 1995)),
        "neighborhood": request.form.get("neighborhood", "Suburb"),
        "manual_price": float(request.form.get("manual_price") or 0),
    }


def _parse_query_form() -> dict:
    return {
        "square_feet": float(request.args.get("square_feet", DEFAULT_FORM["square_feet"])),
        "bedrooms": int(request.args.get("bedrooms", DEFAULT_FORM["bedrooms"])),
        "bathrooms": int(request.args.get("bathrooms", DEFAULT_FORM["bathrooms"])),
        "year_built": int(request.args.get("year_built", DEFAULT_FORM["year_built"])),
        "neighborhood": request.args.get("neighborhood", DEFAULT_FORM["neighborhood"]),
        "manual_price": float(
            request.args.get("manual_price", DEFAULT_FORM["manual_price"])
        ),
    }


def _run_task(task: str, form: dict) -> dict:
    if task == "regression":
        return predict_regression(
            form["square_feet"],
            form["bedrooms"],
            form["bathrooms"],
            form["year_built"],
            form["neighborhood"],
        )
    if task == "classification":
        return predict_classification(
            form["square_feet"],
            form["bedrooms"],
            form["bathrooms"],
            form["year_built"],
            form["neighborhood"],
        )
    price = form.get("manual_price") or 0
    if price <= 0:
        raise ValueError("Harga wajib diisi untuk clustering.")
    return predict_cluster(
        form["square_feet"],
        form["bedrooms"],
        form["bathrooms"],
        form["year_built"],
        form["neighborhood"],
        price=price,
    )


@bp.route("/predict")
def predict_redirect():
    from flask import redirect, url_for

    return redirect(url_for("predict.prediction_page"), code=302)


@bp.route("/prediction", methods=["GET", "POST"])
def prediction_page():
    form = dict(DEFAULT_FORM)
    active_tab = request.args.get("tab", "regression")
    if active_tab not in VALID_TASKS:
        active_tab = "regression"

    result = None
    show_results = False
    shap_image = None
    viz_image = None
    explanations = None

    if not models_ready():
        flash(
            "Model belum ada di folder model/. Extract uas_artifacts.zip (bagian model/) ke root proyek.",
            "warning",
        )

    if request.method == "POST" and models_ready():
        active_tab = request.form.get("task", "regression")
        if active_tab not in VALID_TASKS:
            active_tab = "regression"
        try:
            form = _parse_form()
            if active_tab == "clustering" and form["manual_price"] <= 0:
                raise ValueError("Harga wajib diisi untuk clustering.")
            result = _run_task(active_tab, form)
            show_results = True
            explanations = build_explanations(active_tab, form, result)
            shap_file = SHAP_BY_TASK.get(active_tab)
            if shap_file and asset_exists(shap_file):
                shap_image = shap_file
            if active_tab == "clustering" and asset_exists(CLUSTER_VIZ):
                viz_image = CLUSTER_VIZ
        except Exception as exc:
            flash(f"Prediksi gagal: {exc}", "danger")

    return render_template(
        "prediction.html",
        neighborhoods=NEIGHBORHOODS,
        form=form,
        result=result,
        show_results=show_results,
        shap_image=shap_image,
        viz_image=viz_image,
        active_tab=active_tab,
        explanations=explanations,
        metrics=METRICS,
        models_ready=models_ready(),
    )


@bp.route("/api/prediction/explain")
def api_explain():
    task = request.args.get("task", "regression")
    if task not in VALID_TASKS:
        task = "regression"
    try:
        form = _parse_query_form()
        payload = build_explanations(task, form)
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
