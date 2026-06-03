"""Serialisasi Plotly ke JSON yang kompatibel dengan Plotly.js di browser."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import plotly.graph_objects as go


def _tolist(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return val.tolist()
    if hasattr(val, "tolist"):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    return val


def fig_to_web(fig: go.Figure) -> dict:
    """Figure dict dengan array biasa (bukan bdata) agar Plotly.js merender."""
    data = []
    for trace in fig.data:
        t = trace.to_plotly_json()
        for key in list(t.keys()):
            if key in ("x", "y", "z", "text", "customdata", "marker", "line"):
                if key in ("marker", "line") and isinstance(t[key], dict):
                    for sub in ("color", "size"):
                        if sub in t[key]:
                            t[key][sub] = _tolist(t[key][sub])
                else:
                    t[key] = _tolist(t.get(key))
        data.append(t)

    layout = fig.layout.to_plotly_json()
    return {"data": data, "layout": layout}


def dumps_fig(fig: go.Figure) -> str:
    return json.dumps(fig_to_web(fig))
