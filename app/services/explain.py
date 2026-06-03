"""Penjelasan mengapa setiap fitur memengaruhi prediksi."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from config import CURRENT_YEAR
from services.data import load_raw_data
from services.preprocess import engineer_features


@lru_cache(maxsize=1)
def _benchmarks() -> dict[str, Any]:
    df = load_raw_data()
    med_price = float(df["Price"].median())
    return {
        "median_price": med_price,
        "mean_price": float(df["Price"].mean()),
        "median_sqft": float(df["SquareFeet"].median()),
        "median_bedrooms": float(df["Bedrooms"].median()),
        "median_bathrooms": float(df["Bathrooms"].median()),
        "median_year_built": float(df["YearBuilt"].median()),
        "median_house_age": float(CURRENT_YEAR - df["YearBuilt"].median()),
        "urban_share": float((df["Neighborhood"] == "Urban").mean()),
        "price_by_neighborhood": {
            n: float(df.loc[df["Neighborhood"] == n, "Price"].median())
            for n in df["Neighborhood"].unique()
        },
    }


def _level(value: float, ref: float, pct: float = 0.1) -> str:
    if value > ref * (1 + pct):
        return "high"
    if value < ref * (1 - pct):
        return "low"
    return "mid"


FEATURE_META: dict[str, dict[str, str]] = {
    "square_feet": {
        "label": "Luas bangunan (SquareFeet)",
        "general": (
            "Luas adalah proxy langsung kapasitas hunian dan material bangunan. "
            "Di dataset ini korelasi dengan harga sangat kuat."
        ),
        "regression": "Koefisien regresi & SHAP: SquareFeet konsisten sebagai pendorong utama nilai prediksi harga.",
        "classification": "Rumah lebih luas cenderung masuk kelas High (di atas median harga).",
        "clustering": "Membedakan cluster 'premium/luas' vs cluster lebih kompak.",
    },
    "bedrooms": {
        "label": "Jumlah kamar tidur",
        "general": "Lebih banyak kamar → target pasar keluarga lebih besar, tetapi rasio luas per kamar ikut masuk model.",
        "regression": "Efek non-linear lewat fitur turunan AreaPerBedroom = luas / (bedrooms + 1).",
        "classification": "Kombinasi bedrooms + luas membedakan segment mid vs entry-level.",
        "clustering": "Profil cluster 0 vs 1 di notebook punya rata-rata bedrooms berbeda.",
    },
    "bathrooms": {
        "label": "Jumlah kamar mandi",
        "general": "Kamar mandi menandai kenyamanan & kelas properti; berkorelasi positif dengan harga.",
        "regression": "AreaPerBathroom dan BathPerBedroom menangkap efisiensi tata ruang.",
        "classification": "Lebih banyak bathroom sering mengangkat probabilitas kelas High.",
        "clustering": "Pemisah kuat antara cluster dengan kualitas fasilitas berbeda.",
    },
    "year_built": {
        "label": "Tahun dibangun",
        "general": "Tahun bangun menentukan usia rumah (HouseAge) — bangunan lebih baru sering dihargai lebih tinggi.",
        "regression": "Model memakai HouseAge = tahun sekarang − YearBuilt, bukan tahun mentah.",
        "classification": "Usia ekstrem (sangat tua / sangat baru) bisa menggeser kategori.",
        "clustering": "HouseAge masuk vektor cluster setelah scaling.",
    },
    "neighborhood": {
        "label": "Neighborhood (lokasi)",
        "general": "Lokasi menangkap faktor eksternal: akses, permintaan pasar, dan segmentasi Rural / Suburb / Urban.",
        "regression": "Di-encode one-hot (Suburb, Urban); Rural menjadi baseline.",
        "classification": "Urban/Suburb biasanya terkait median harga lebih tinggi daripada Rural.",
        "clustering": "Dummy neighborhood ikut membentuk jarak Euclidean setelah StandardScaler.",
    },
    "house_age": {
        "label": "Usia rumah (HouseAge)",
        "general": "Fitur hasil engineering — menggantikan YearBuilt di model.",
        "regression": "Rumah lebih tua dapat menurunkan prediksi jika pasar mengutamakan kondisi baru.",
        "classification": "Membantu memisahkan properti renovasi vs butuh perbaikan.",
        "clustering": "Salah satu sumbu perilaku di profil cluster.",
    },
    "area_per_bedroom": {
        "label": "Luas per kamar (AreaPerBedroom)",
        "general": "Mengukur 'kemewahan' ruang per tidur — kamar kecil di rumah besar vs sebaliknya.",
        "regression": "Nilai tinggi → tiap kamar lebih lapang → bias positif pada harga.",
        "classification": "Membedakan rumah keluarga besar vs over-built kecil.",
        "clustering": "Memisahkan cluster premium vs padat.",
    },
    "price_for_cluster": {
        "label": "Harga (untuk clustering)",
        "general": "K-Means memakai Price (prediksi regresi atau manual) sebagai fitur skala harga pasar.",
        "regression": "Tidak dipakai di regresi.",
        "classification": "Tidak dipakai langsung di klasifikasi (label High/Low dari median training).",
        "clustering": "Menggeser posisi cluster: harga tinggi → cenderung cluster dengan centroid Price lebih besar.",
    },
}


def _engineered(form: dict) -> dict[str, float]:
    row = engineer_features(
        pd.DataFrame(
            [
                {
                    "SquareFeet": form["square_feet"],
                    "Bedrooms": form["bedrooms"],
                    "Bathrooms": form["bathrooms"],
                    "YearBuilt": form["year_built"],
                    "Neighborhood": form["neighborhood"],
                }
            ]
        )
    )
    r = row.iloc[0]
    return {
        "house_age": float(r.get("HouseAge", CURRENT_YEAR - form["year_built"])),
        "area_per_bedroom": float(r["AreaPerBedroom"]),
        "area_per_bathroom": float(r["AreaPerBathroom"]),
        "bath_per_bedroom": float(r["BathPerBedroom"]),
    }


def _dynamic_insight(key: str, form: dict, bench: dict, eng: dict) -> str:
    if key == "square_feet":
        lv = _level(form["square_feet"], bench["median_sqft"])
        if lv == "high":
            return f"Luas {form['square_feet']:,.0f} sq ft di atas median dataset ({bench['median_sqft']:,.0f}) → cenderung mendorong harga & kelas High."
        if lv == "low":
            return f"Luas di bawah median ({bench['median_sqft']:,.0f} sq ft) → tekanan ke harga lebih rendah / cluster kompak."
        return "Luas mendekati median pasar — kontribusi moderat terhadap prediksi."

    if key == "bedrooms":
        lv = _level(form["bedrooms"], bench["median_bedrooms"], 0.15)
        apb = eng["area_per_bedroom"]
        if lv == "high":
            return f"Banyak kamar ({form['bedrooms']}) dengan AreaPerBedroom ≈ {apb:.0f} — perhatikan apakah luas 'menopang' jumlah kamar."
        return f"AreaPerBedroom ≈ {apb:.0f} sq ft/kamar — rasio ini yang dibaca model, bukan bedrooms saja."

    if key == "bathrooms":
        lv = _level(form["bathrooms"], bench["median_bathrooms"], 0.15)
        if lv == "high":
            return "Kamar mandi di atas median → sinyal fasilitas lebih baik, mendukung harga & klasifikasi High."
        return "Jumlah bathroom mendekati tipikal pasar; efek lewat AreaPerBathroom & BathPerBedroom."

    if key == "year_built":
        age = eng["house_age"]
        if age < bench["median_house_age"] - 5:
            return f"Rumah relatif baru (HouseAge ≈ {age:.0f} tahun) → bias positif pada nilai."
        if age > bench["median_house_age"] + 10:
            return f"Rumah lebih tua (HouseAge ≈ {age:.0f} tahun) → risiko penurunan nilai di model linear/tree."
        return f"Usia bangunan ≈ {age:.0f} tahun, dekat median pasar."

    if key == "neighborhood":
        nb = form["neighborhood"]
        med_nb = bench["price_by_neighborhood"].get(nb, bench["median_price"])
        diff = (med_nb - bench["median_price"]) / bench["median_price"] * 100
        if diff > 5:
            return f"{nb}: median harga historis ~${med_nb:,.0f} ({diff:+.0f}% vs global median) → lokasi premium."
        if diff < -5:
            return f"{nb}: median historis lebih rendah (~${med_nb:,.0f}) → menekan prediksi relatif."
        return f"{nb}: median harga sejajar dengan keseluruhan dataset."

    if key == "house_age":
        return _dynamic_insight("year_built", form, bench, eng)

    if key == "area_per_bedroom":
        apb = eng["area_per_bedroom"]
        if apb > 700:
            return f"AreaPerBedroom tinggi ({apb:.0f}) → ruang per kamar lapang, mendukung segment premium."
        if apb < 450:
            return f"AreaPerBedroom rendah ({apb:.0f}) → hunian padat, sering cluster/ harga entry."
        return f"Rasio luas/kamar moderat ({apb:.0f})."

    if key == "price_for_cluster":
        price = form.get("cluster_price") or form.get("manual_price") or 0
        if price <= 0:
            return "Isi harga (USD) — wajib untuk menentukan cluster."
        lv = _level(price, bench["median_price"])
        if lv == "high":
            return f"Harga ${price:,.0f} di atas median (${bench['median_price']:,.0f}) → cenderung cluster dengan centroid harga lebih tinggi."
        if lv == "low":
            return f"Harga ${price:,.0f} di bawah median → mendekati cluster entry-level."
        return f"Harga ${price:,.0f} mendekati median pasar — posisi cluster moderat."

    return ""


def build_explanations(
    task: str,
    form: dict,
    result: dict | None = None,
) -> dict[str, Any]:
    """task: regression | classification | clustering"""
    bench = _benchmarks()
    eng = _engineered(form)
    if result and "cluster_price" in result:
        form = {**form, "cluster_price": result["cluster_price"]}

    keys = ["square_feet", "bedrooms", "bathrooms", "year_built", "neighborhood"]
    if task == "clustering":
        keys.append("price_for_cluster")

    features = []
    for key in keys:
        meta = FEATURE_META[key]
        features.append(
            {
                "key": key,
                "label": meta["label"],
                "general": meta["general"],
                "task_impact": meta.get(task, meta["general"]),
                "insight": _dynamic_insight(key, form, bench, eng),
            }
        )

    features.append(
        {
            "key": "house_age",
            "label": FEATURE_META["house_age"]["label"],
            "general": FEATURE_META["house_age"]["general"],
            "task_impact": FEATURE_META["house_age"][task],
            "insight": _dynamic_insight("house_age", form, bench, eng),
            "computed": f"{eng['house_age']:.0f} tahun",
        }
    )
    features.append(
        {
            "key": "area_per_bedroom",
            "label": FEATURE_META["area_per_bedroom"]["label"],
            "general": FEATURE_META["area_per_bedroom"]["general"],
            "task_impact": FEATURE_META["area_per_bedroom"][task],
            "insight": _dynamic_insight("area_per_bedroom", form, bench, eng),
            "computed": f"{eng['area_per_bedroom']:.0f} sq ft/kamar",
        }
    )

    summary = _result_summary(task, form, result, bench, eng)
    return {
        "task": task,
        "features": features,
        "engineered": eng,
        "summary": summary,
        "shap_hint": (
            "Penjelasan disesuaikan dengan input Anda. "
            + (
                "Plot SHAP ditampilkan di atas untuk task ini."
                if task in ("regression", "classification")
                else "Clustering memakai harga yang Anda masukkan sebagai fitur Price."
            )
        ),
    }


def _result_summary(
    task: str,
    form: dict,
    result: dict | None,
    bench: dict,
    eng: dict,
) -> str | None:
    if not result:
        return None
    if task == "regression":
        p = result["price_pred"]
        diff = (p - bench["median_price"]) / bench["median_price"] * 100
        return (
            f"Prediksi ${p:,.0f} — {'di atas' if diff > 0 else 'di bawah'} median dataset "
            f"(${bench['median_price']:,.0f}, {diff:+.1f}%). "
            f"Kontributor utama pada input ini: luas {form['square_feet']:,.0f} sq ft, "
            f"lokasi {form['neighborhood']}, HouseAge {eng['house_age']:.0f} tahun."
        )
    if task == "classification":
        cat = result["category"]
        proba = result.get("proba")
        base = (
            f"Model memprediksi {cat} (dibanding median harga training). "
            f"Kombinasi luas, bathroom, dan neighborhood {form['neighborhood']} "
            f"menjelaskan arah klasifikasi."
        )
        if proba is not None:
            base += f" Confidence ≈ {proba * 100:.1f}%."
        return base
    cid = result["cluster_id"]
    cp = result["cluster_price"]
    return (
        f"Anda masuk Cluster {cid}. Harga fitur clustering: ${cp:,.0f}. "
        f"Cluster memisahkan pola luas–harga–lokasi; lihat profil di halaman Clustering."
    )
