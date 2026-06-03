"""Dashboard dataset: KPI, chart interaktif, breakdown per fitur."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from services.data import load_raw_data
from services.plotly_json import fig_to_web

NUMERIC_COLS = ["SquareFeet", "Bedrooms", "Bathrooms", "YearBuilt", "Price"]
CATEGORICAL_COLS = ["Neighborhood"]
CHART_TYPES = ("scatter", "bar", "histogram", "box", "violin", "line")
SPLIT_MODES = ("none", "neighborhood", "train_test")


def get_kpis() -> dict[str, Any]:
    df = load_raw_data()
    return {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "avg_price": float(df["Price"].mean()),
        "median_price": float(df["Price"].median()),
        "avg_sqft": float(df["SquareFeet"].mean()),
        "neighborhoods": int(df["Neighborhood"].nunique()),
        "year_min": int(df["YearBuilt"].min()),
        "year_max": int(df["YearBuilt"].max()),
        "missing_total": int(df.isnull().sum().sum()),
    }


def _filter_df(
    df: pd.DataFrame,
    neighborhood: str,
    sample: int,
    split: str,
) -> pd.DataFrame:
    out = df.copy()
    if neighborhood and neighborhood != "all":
        out = out[out["Neighborhood"] == neighborhood]
    if sample > 0 and len(out) > sample:
        out = out.sample(sample, random_state=42)
    if split == "train_test":
        n_train = int(len(out) * 0.8)
        out = out.copy()
        out["_split"] = ["Train"] * n_train + ["Test"] * (len(out) - n_train)
        out = out.sample(frac=1, random_state=42).reset_index(drop=True)
    elif split == "neighborhood":
        out["_split"] = out["Neighborhood"]
    return out


def _style_fig(fig: go.Figure, height: int = 480, *, compact: bool = False) -> go.Figure:
    """Layout agar judul, legenda, dan label sumbu tidak saling menutupi."""
    n_traces = len(fig.data)
    show_legend = n_traces > 1 and any(getattr(t, "showlegend", True) for t in fig.data)

    if compact:
        # Judul di HTML (nama fitur di kartu), bukan di dalam plot
        margins = dict(l=52, r=16, t=28, b=44)
        legend_cfg = dict(visible=False)
    else:
        # Judul di HTML (#main-chart-title); margin atas kecil — hindari tabrakan modebar
        bottom = 100 if show_legend else 72
        margins = dict(l=72, r=16, t=28, b=bottom)
        legend_cfg = (
            dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#d6e4f5",
                borderwidth=1,
                font=dict(size=11),
                tracegroupgap=8,
            )
            if show_legend
            else dict(visible=False)
        )

    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=margins,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(color="#0f172a", size=12),
        colorway=["#2563eb", "#3b82f6", "#60a5fa", "#1d4ed8", "#93c5fd"],
        title=None,
        legend=legend_cfg,
        xaxis=dict(
            automargin=True,
            title_standoff=14,
            tickangle=-35 if not compact else 0,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            automargin=True,
            title_standoff=14,
            tickfont=dict(size=10),
        ),
    )

    # Satu seri → tanpa legenda
    if not show_legend:
        fig.update_traces(showlegend=False)

    fig.update_traces(
        marker=dict(opacity=0.7, line=dict(width=0.4, color="#ffffff")),
        selector=dict(type="scatter"),
    )
    fig.update_traces(showlegend=show_legend)

    return fig


def build_main_chart(
    x: str,
    y: str,
    color: str | None,
    chart_type: str,
    neighborhood: str = "all",
    sample: int = 8000,
    split: str = "none",
) -> dict:
    df = load_raw_data()
    x = x if x in df.columns else "SquareFeet"
    y = y if y in df.columns else "Price"
    color = color if color and color in df.columns else None
    chart_type = chart_type if chart_type in CHART_TYPES else "scatter"
    split = split if split in SPLIT_MODES else "none"

    df = _filter_df(df, neighborhood, sample, split)
    if df.empty:
        raise ValueError("Tidak ada data setelah filter.")

    color_arg = color or ("_split" if split != "none" and "_split" in df.columns else None)

    if chart_type == "scatter":
        fig = px.scatter(df, x=x, y=y, color=color_arg)
    elif chart_type == "histogram":
        fig = px.histogram(df, x=x, color=color_arg, nbins=40)
    elif chart_type == "box":
        fig = px.box(df, x=color_arg or "Neighborhood", y=y)
    elif chart_type == "violin":
        fig = px.violin(df, x=color_arg or "Neighborhood", y=y, box=True)
    elif chart_type == "bar":
        if color_arg:
            agg = df.groupby(color_arg, as_index=False)[y].mean()
            fig = px.bar(agg, x=color_arg, y=y)
        else:
            agg = df.groupby(x, as_index=False)[y].mean()
            fig = px.bar(agg, x=x, y=y)
    else:
        if pd.api.types.is_numeric_dtype(df[x]):
            df = df.sort_values(x)
        fig = px.line(df, x=x, y=y, color=color_arg)

    return fig_to_web(_style_fig(fig, height=520))


def build_feature_chart(col: str) -> dict:
    df = load_raw_data()
    if col not in df.columns:
        raise ValueError(f"Kolom tidak ada: {col}")

    if col in CATEGORICAL_COLS:
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        fig = px.bar(
            counts,
            x=col,
            y="count",
            color_discrete_sequence=["#2563eb", "#3b82f6", "#60a5fa"],
        )
    else:
        fig = px.histogram(
            df,
            x=col,
            nbins=35,
            color_discrete_sequence=["#2563eb"],
        )

    fig.update_traces(showlegend=False)
    return fig_to_web(_style_fig(fig, height=240, compact=True))


def build_feature_breakdown() -> list[dict]:
    df = load_raw_data()
    items = []

    for col in NUMERIC_COLS:
        s = df[col].dropna()
        items.append(
            {
                "name": col,
                "type": "numeric",
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std()), 2),
                "min": round(float(s.min()), 2),
                "max": round(float(s.max()), 2),
            }
        )

    for col in CATEGORICAL_COLS:
        counts = df[col].value_counts()
        items.append(
            {
                "name": col,
                "type": "categorical",
                "top_category": str(counts.index[0]),
                "top_count": int(counts.iloc[0]),
                "unique": int(df[col].nunique()),
            }
        )

    return items
