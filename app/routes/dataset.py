"""Dataset dashboard + API chart."""

from flask import Blueprint, jsonify, render_template, request

from config import METRICS
from services.dashboard import (
    CHART_TYPES,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    SPLIT_MODES,
    build_feature_breakdown,
    build_feature_chart,
    build_main_chart,
    get_kpis,
)
from services.predict import models_ready

bp = Blueprint("dataset", __name__)


@bp.route("/dataset")
def dataset_dashboard():
    error = None
    kpis = features = None
    try:
        kpis = get_kpis()
        features = build_feature_breakdown()
    except FileNotFoundError as exc:
        error = str(exc)
    except Exception as exc:
        error = str(exc)

    return render_template(
        "dataset.html",
        kpis=kpis,
        features=features,
        numeric_cols=NUMERIC_COLS,
        categorical_cols=CATEGORICAL_COLS,
        chart_types=CHART_TYPES,
        split_modes=SPLIT_MODES,
        metrics=METRICS,
        models_ready=models_ready(),
        error=error,
    )


@bp.route("/api/chart")
def api_chart():
    try:
        color = request.args.get("color") or None
        fig = build_main_chart(
            x=request.args.get("x", "SquareFeet"),
            y=request.args.get("y", "Price"),
            color=color,
            chart_type=request.args.get("chart", "scatter"),
            neighborhood=request.args.get("neighborhood", "all"),
            sample=int(request.args.get("sample", 8000)),
            split=request.args.get("split", "none"),
        )
        return jsonify(fig)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@bp.route("/api/feature-chart/<col>")
def api_feature_chart(col: str):
    try:
        return jsonify(build_feature_chart(col))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
