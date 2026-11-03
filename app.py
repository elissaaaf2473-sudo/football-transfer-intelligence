
from pathlib import Path
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from catboost import CatBoostRegressor, Pool

# ==========================================
# 01 - PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Football Transfer Intelligence 2.0",
    page_icon="⚽",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent

FEATURES = [
    "year",
    "age_approx",
    "left_club_id",
    "joined_club_id"
]

CATS = ["left_club_id", "joined_club_id"]
TEAL = "#2DD4BF"

st.markdown("""
<style>
.stApp {
    background-color: #0B1220;
}
[data-testid="stSidebar"] {
    background-color: #111B2A;
}
.block-container {
    padding-top: 2rem;
}
div[data-testid="stMetric"] {
    background: #132238;
    border: 1px solid #263951;
    border-radius: 14px;
    padding: 16px;
}
div[data-testid="stMetricValue"] {
    color: #2DD4BF;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 02 - VERIFY REQUIRED FILES
# ==========================================

required = [
    "final_test_results.csv",
    "shap_importance.csv",
    "coverage.csv",
    "club_catalog.csv"
]

missing = [
    f for f in required
    if not (ROOT / f).is_file()
]

parts = sorted(
    ROOT.glob("catboost_final.cbm.part*")
)

if (
    not (ROOT / "catboost_final.cbm").is_file()
    and not parts
):
    missing.append(
        "catboost_final.cbm.part001, part002, dst."
    )

if missing:
    st.error(
        "File belum lengkap: " + ", ".join(missing)
    )
    st.stop()


# ==========================================
# 03 - LOAD SPLIT CATBOOST MODEL
# ==========================================

@st.cache_resource(
    show_spinner="Memuat model CatBoost..."
)
def load_model():

    model = CatBoostRegressor()
    original = ROOT / "catboost_final.cbm"

    # Jika model tersedia dalam satu file
    if original.is_file():
        model.load_model(str(original))
        return model

    # Jika model dipecah menjadi beberapa part
    chunks = sorted(
        ROOT.glob("catboost_final.cbm.part*")
    )

    numbers = []

    for path in chunks:

        part_number = path.name.rsplit(
            ".part", 1
        )[-1]

        if not part_number.isdigit():
            st.error(
                f"Nama file part tidak valid: {path.name}"
            )
            st.stop()

        numbers.append(int(part_number))

    expected = list(
        range(1, len(numbers) + 1)
    )

    if numbers != expected:
        st.error(
            "Nomor part tidak berurutan. "
            "Upload semua bagian model!"
        )
        st.stop()

    # Gabungkan part sementara
    with tempfile.NamedTemporaryFile(
        suffix=".cbm",
        delete=False
    ) as temp:

        temporary_path = Path(temp.name)

        for part in chunks:
            with part.open("rb") as source:
                shutil.copyfileobj(
                    source, temp
                )

    try:
        model.load_model(
            str(temporary_path)
        )

    except Exception as exc:
        st.error(
            "Model tidak bisa dimuat. "
            "Pastikan semua part berasal "
            "dari ZIP yang sama."
        )
        st.exception(exc)
        st.stop()

    finally:
        temporary_path.unlink(
            missing_ok=True
        )

    return model


# ==========================================
# 04 - LOAD RESEARCH DATA
# ==========================================

@st.cache_data(show_spinner=False)
def read_csv(name, **kwargs):
    return pd.read_csv(
        ROOT / name,
        **kwargs
    )


model = load_model()

results = read_csv(
    "final_test_results.csv"
)

coverage = read_csv(
    "coverage.csv"
)

shap_global = read_csv(
    "shap_importance.csv"
)

clubs = read_csv(
    "club_catalog.csv",
    dtype={"club_id": str}
)


# ==========================================
# 05 - DATA VALIDATION
# ==========================================

expected = [
    (
        results,
        {"Model", "MAE", "RMSE", "R2"},
        "final_test_results.csv"
    ),
    (
        coverage,
        {
            "year",
            "positive_fee_pct",
            "positive_fee_records",
            "total_records"
        },
        "coverage.csv"
    ),
    (
        shap_global,
        {"Feature", "Mean_Absolute_SHAP"},
        "shap_importance.csv"
    ),
    (
        clubs,
        {"club_id", "record_count"},
        "club_catalog.csv"
    )
]

for table, cols, filename in expected:

    if not cols.issubset(table.columns):
        st.error(
            f"Kolom {filename} tidak sesuai. "
            f"Dibutuhkan: {sorted(cols)}"
        )
        st.stop()


clubs = clubs.dropna(
    subset=["club_id"]
).copy()

clubs["record_count"] = pd.to_numeric(
    clubs["record_count"],
    errors="coerce"
).fillna(0)

clubs = clubs.sort_values(
    "record_count",
    ascending=False
)

club_ids = clubs["club_id"].tolist()

club_count = dict(
    zip(
        club_ids,
        clubs["record_count"].astype(int)
    )
)

if not club_ids:
    st.error("Daftar klub kosong.")
    st.stop()


# ==========================================
# 06 - SIDEBAR INPUT
# ==========================================

st.sidebar.title(
    "⚽ Transfer Intelligence"
)

st.sidebar.caption(
    "Simulator skenario transfer historis"
)

with st.sidebar.form("fee_form"):

    year = st.slider(
        "Tahun transfer",
        2010,
        2021,
        2021
    )

    age = st.number_input(
        "Usia pemain",
        min_value=15,
        max_value=45,
        value=24
    )

    origin = st.selectbox(
        "Klub asal",
        club_ids,
        format_func=lambda x: (
            f"Club ID {x} · "
            f"{club_count.get(x, 0):,} catatan"
        )
    )

    destination = st.selectbox(
        "Klub tujuan",
        club_ids,
        index=1 if len(club_ids) > 1 else 0,
        format_func=lambda x: (
            f"Club ID {x} · "
            f"{club_count.get(x, 0):,} catatan"
        )
    )

    submit = st.form_submit_button(
        "⚡ Estimate Transfer Fee",
        use_container_width=True
    )


if submit:
    st.session_state["scenario"] = {
        "year": int(year),
        "age_approx": float(age),
        "left_club_id": str(origin),
        "joined_club_id": str(destination)
    }


st.sidebar.divider()

st.sidebar.caption(
    "Model: CatBoost, depth 8, "
    "1.000 iterasi, target log1p"
)

st.sidebar.caption(
    "Simulasi historis, bukan "
    "harga pasar terkini."
)


# ==========================================
# 07 - MAIN DASHBOARD
# ==========================================

st.title(
    "⚽ Football Transfer Intelligence 2.0"
)

st.caption(
    "Interactive ML Research · CatBoost · "
    "Temporal Validation · Explainable AI"
)

cat = results.loc[
    results["Model"] == "CatBoost"
]

base = results.loc[
    results["Model"] == "Median Baseline"
]

if cat.empty or base.empty:
    st.error(
        "Baris CatBoost / Median Baseline "
        "tidak ditemukan di hasil test."
    )
    st.stop()

cat_mae = float(cat.iloc[0]["MAE"])
base_mae = float(base.iloc[0]["MAE"])
score = float(cat.iloc[0]["R2"])

improvement = (
    (base_mae - cat_mae)
    / base_mae * 100
)

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Final test MAE",
    f"{cat_mae:,.0f}"
)

k2.metric(
    "Final test R²",
    f"{score:.4f}"
)

k3.metric(
    "Perbaikan vs baseline",
    f"{improvement:.2f}%"
)

k4.metric(
    "Test transactions",
    "5,332"
)


market, simulator, lab, explain = st.tabs([
    "📈 Market Explorer",
    "🧮 Fee Simulator",
    "🧪 Model Laboratory",
    "🔍 Explainable AI"
])


# ==========================================
# 08 - CHART STYLING
# ==========================================

def style(fig, height=450):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#132238",
        height=height,
        margin=dict(
            l=30,
            r=25,
            t=45,
            b=35
        )
    )

    return fig


# ==========================================
# 09 - MARKET EXPLORER
# ==========================================

with market:

    st.subheader(
        "Recorded Transfer Fee Coverage"
    )

    coverage["year"] = pd.to_numeric(
        coverage["year"],
        errors="coerce"
    )

    years = (
        int(coverage["year"].min()),
        int(coverage["year"].max())
    )

    chosen = st.slider(
        "Rentang tahun",
        years[0],
        years[1],
        years
    )

    metric_label = st.selectbox(
        "Tampilkan",
        [
            "Persentase fee positif",
            "Jumlah fee positif",
            "Semua catatan non-loan"
        ]
    )

    cols = {
        "Persentase fee positif":
            "positive_fee_pct",
        "Jumlah fee positif":
            "positive_fee_records",
        "Semua catatan non-loan":
            "total_records"
    }

    filtered = coverage[
        coverage["year"].between(*chosen)
    ]

    fig = px.line(
        filtered,
        x="year",
        y=cols[metric_label],
        markers=True,
        labels={
            "year": "Tahun",
            cols[metric_label]: metric_label
        }
    )

    fig.update_traces(
        line_color=TEAL,
        line_width=3,
        marker_size=9
    )

    fig.update_xaxes(dtick=1)

    st.plotly_chart(
        style(fig),
        use_container_width=True
    )

    with st.expander(
        "📖 Cara membaca grafik"
    ):

        st.write(
            "Grafik menunjukkan catatan "
            "transfer non-loan dengan fee positif. "
            "Fee nol atau tak tercatat tidak "
            "otomatis berarti transfer gratis. "
            "Angka ini bukan cakupan seluruh "
            "transfer dunia."
        )


# ==========================================
# 10 - FEE SIMULATOR
# ==========================================

with simulator:

    st.subheader(
        "Simulasi Estimasi Biaya Transfer"
    )

    st.caption(
        "Ubah input pada sidebar, lalu klik "
        "**Estimate Transfer Fee**."
    )

    scenario = st.session_state.get(
        "scenario"
    )

    if scenario is None:

        st.info(
            "Belum ada skenario. Pilih tahun, "
            "usia, klub asal, dan klub tujuan "
            "di sidebar."
        )

    else:

        frame = pd.DataFrame(
            [scenario],
            columns=FEATURES
        )

        for col in CATS:
            frame[col] = frame[col].astype(str)

        prediction_log = float(
            model.predict(frame)[0]
        )

        estimate = max(
            float(np.expm1(prediction_log)),
            0
        )

        st.metric(
            "Estimated Recorded Fee",
            f"{estimate:,.0f}"
        )

        st.dataframe(
            frame,
            hide_index=True,
            use_container_width=True
        )

        if (
            scenario["left_club_id"]
            == scenario["joined_club_id"]
        ):
            st.warning(
                "Klub asal sama dengan tujuan. "
                "Skenario ini mungkin tidak realistis."
            )

        st.info(
            "Angka mengikuti satuan dalam dataset, "
            "bukan valuasi pemain saat ini. "
            "Model hanya dilatih untuk transfer "
            "permanen dengan biaya positif tercatat."
        )


# ==========================================
# 11 - MODEL LABORATORY
# ==========================================

with lab:

    st.subheader(
        "Evaluasi Final Test 2022-2023"
    )

    metric = st.selectbox(
        "Pilih metrik",
        ["MAE", "RMSE", "R2"]
    )

    ranking = results.sort_values(
        metric,
        ascending=(metric != "R2")
    )

    fig = px.bar(
        ranking,
        x="Model",
        y=metric,
        color="Model",
        text=metric
    )

    fig.update_layout(
        showlegend=False
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        style(fig),
        use_container_width=True
    )

    st.dataframe(
        ranking,
        hide_index=True,
        use_container_width=True
    )

    with st.expander(
        "📖 Cara membaca metrik"
    ):

        st.markdown(
            "**MAE:** Rata-rata selisih absolut. "
            "Makin kecil makin baik.\n\n"
            "**RMSE:** Lebih sensitif terhadap "
            "kesalahan ekstrem.\n\n"
            "**R²:** Kemampuan menjelaskan "
            "variasi target dibanding prediksi "
            "konstan. R² bukan persentase akurasi."
        )

    # Grafik tambahan jika CSV tersedia
    if (ROOT / "final_predictions.csv").exists():

        pred = read_csv(
            "final_predictions.csv"
        )

        needed = {
            "actual_fee",
            "predicted_fee",
            "year"
        }

        if needed.issubset(pred.columns):

            st.markdown(
                "#### Actual vs Predicted"
            )

            show = pred[
                (pred["actual_fee"] > 0)
                & (pred["predicted_fee"] > 0)
            ].copy()

            show["year"] = (
                show["year"].astype(str)
            )

            fig = px.scatter(
                show,
                x="actual_fee",
                y="predicted_fee",
                color="year",
                log_x=True,
                log_y=True,
                opacity=0.5,
                render_mode="webgl"
            )

            lo = float(min(
                show["actual_fee"].min(),
                show["predicted_fee"].min()
            ))

            hi = float(max(
                show["actual_fee"].max(),
                show["predicted_fee"].max()
            ))

            fig.add_trace(
                go.Scatter(
                    x=[lo, hi],
                    y=[lo, hi],
                    mode="lines",
                    name="Perfect prediction",
                    line=dict(
                        color="#F59E0B",
                        dash="dash"
                    )
                )
            )

            st.plotly_chart(
                style(fig, height=550),
                use_container_width=True
            )


# ==========================================
# 12 - EXPLAINABLE AI (SHAP)
# ==========================================

with explain:

    st.subheader(
        "SHAP - Faktor yang Memengaruhi Prediksi"
    )

    global_chart = shap_global.sort_values(
        "Mean_Absolute_SHAP"
    )

    fig = px.bar(
        global_chart,
        x="Mean_Absolute_SHAP",
        y="Feature",
        orientation="h",
        color_discrete_sequence=[TEAL]
    )

    st.plotly_chart(
        style(fig),
        use_container_width=True
    )

    with st.expander(
        "📖 Cara membaca SHAP"
    ):

        st.write(
            "Mean absolute SHAP menunjukkan "
            "kontribusi fitur pada skala log1p "
            "target, bukan uang dan bukan "
            "pengaruh sebab-akibat."
        )

    scenario = st.session_state.get(
        "scenario"
    )

    if scenario:

        st.markdown(
            "#### SHAP untuk skenario yang dipilih"
        )

        frame = pd.DataFrame(
            [scenario],
            columns=FEATURES
        )

        for col in CATS:
            frame[col] = frame[col].astype(str)

        pool = Pool(
            frame,
            cat_features=CATS
        )

        values = model.get_feature_importance(
            data=pool,
            type="ShapValues"
        )[0]

        local = pd.DataFrame({
            "Feature": FEATURES,
            "Contribution": values[:-1]
        })

        local["Direction"] = np.where(
            local["Contribution"] >= 0,
            "Menaikkan prediksi log",
            "Menurunkan prediksi log"
        )

        fig = px.bar(
            local.sort_values("Contribution"),
            x="Contribution",
            y="Feature",
            color="Direction",
            orientation="h"
        )

        st.plotly_chart(
            style(fig),
            use_container_width=True
        )

        st.caption(
            f"Base value (log): {values[-1]:.3f}. "
            "Jumlah SHAP + base = "
            "output model (log)."
        )

    else:

        st.info(
            "Jalankan simulasi di sidebar "
            "untuk melihat SHAP individual."
        )


# ==========================================
# 13 - FOOTER
# ==========================================

st.divider()

st.caption(
    "Research demo · Transfer permanen "
    "dengan fee positif tercatat · "
    "Bukan harga real-time · "
    "Model final tidak dilatih ulang "
    "di aplikasi."
)
