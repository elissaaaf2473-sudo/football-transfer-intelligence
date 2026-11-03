"""Football Transfer Intelligence 2.0 — historical scenario dashboard.

Run with: streamlit run app.py
The model was trained on recorded positive-fee, non-loan transfers.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from catboost import CatBoostRegressor, Pool

st.set_page_config(
    page_title="Football Transfer Intelligence 2.0",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
FEATURES = ["year", "age_approx", "left_club_id", "joined_club_id"]
CATEGORICAL = ["left_club_id", "joined_club_id"]
COLORS = {
    "navy": "#0B1220",
    "panel": "#111E30",
    "teal": "#2DD4BF",
    "blue": "#60A5FA",
    "purple": "#A78BFA",
    "orange": "#F59E0B",
}

st.markdown(
    """
<style>
.stApp { background-color: #0B1220; }
[data-testid="stSidebar"] { background-color: #111B2A; }
.block-container { padding-top: 2rem; }
h1, h2, h3 { letter-spacing: -.025em; }
div[data-testid="stMetric"] {
  background: #132238;
  border: 1px solid #243750;
  border-radius: 14px;
  padding: 16px 18px;
}
div[data-testid="stMetricValue"] { color: #2DD4BF; }
</style>
""",
    unsafe_allow_html=True,
)

REQUIRED = [
    "catboost_final.cbm",
    "final_test_results.csv",
    "shap_importance.csv",
    "coverage.csv",
    "club_catalog.csv",
]
missing = [filename for filename in REQUIRED if not (ROOT / filename).is_file()]
if missing:
    st.error("Berkas aplikasi belum lengkap: " + ", ".join(missing))
    st.info("Upload semua file yang diperlukan ke folder utama repository GitHub, bersama app.py.")
    st.stop()


@st.cache_resource(show_spinner="Memuat model CatBoost ...")
def load_model() -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(ROOT / "catboost_final.cbm"))
    return model


@st.cache_data(show_spinner=False)
def load_csv(filename: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(ROOT / filename, **kwargs)


model = load_model()
results = load_csv("final_test_results.csv")
shap_global = load_csv("shap_importance.csv")
coverage = load_csv("coverage.csv")
club_catalog = load_csv("club_catalog.csv", dtype={"club_id": str})

if not set(["Model", "MAE", "RMSE", "R2"]).issubset(results.columns):
    st.error("Kolom final_test_results.csv tidak sesuai.")
    st.stop()
if not set(["club_id", "record_count"]).issubset(club_catalog.columns):
    st.error("Kolom club_catalog.csv tidak sesuai.")
    st.stop()
if not set(["year", "positive_fee_pct", "positive_fee_records", "total_records"]).issubset(coverage.columns):
    st.error("Kolom coverage.csv tidak sesuai.")
    st.stop()
if not set(["Feature", "Mean_Absolute_SHAP"]).issubset(shap_global.columns):
    st.error("Kolom shap_importance.csv tidak sesuai.")
    st.stop()

club_catalog = club_catalog.dropna(subset=["club_id"]).copy()
club_catalog["record_count"] = pd.to_numeric(club_catalog["record_count"], errors="coerce").fillna(0)
club_catalog = club_catalog.sort_values("record_count", ascending=False)
club_ids = club_catalog["club_id"].astype(str).tolist()
club_counts = dict(zip(club_ids, club_catalog["record_count"].astype(int)))
if not club_ids:
    st.error("Katalog klub kosong.")
    st.stop()

st.sidebar.markdown("## ⚽ Football Transfer Intelligence")
st.sidebar.caption("Interactive historical transfer-fee research")
st.sidebar.markdown("### 🎛️ Scenario inputs")
st.sidebar.caption("Input untuk CatBoost (2010–2021). Pilih ID klub dari data development.")

with st.sidebar.form("scenario_form"):
    year = st.slider("Historical transfer year", min_value=2010, max_value=2021, value=2021)
    age = st.number_input("Player age at transfer", min_value=15, max_value=45, value=24, step=1)
    origin = st.selectbox(
        "Origin club",
        club_ids,
        index=0,
        format_func=lambda c: f"Club ID {c} · {club_counts.get(c, 0):,} records",
    )
    destination = st.selectbox(
        "Destination club",
        club_ids,
        index=(1 if len(club_ids) > 1 else 0),
        format_func=lambda c: f"Club ID {c} · {club_counts.get(c, 0):,} records",
    )
    run = st.form_submit_button("⚡ Estimate transfer fee", use_container_width=True)

if run:
    st.session_state["scenario"] = {
        "year": int(year),
        "age_approx": float(age),
        "left_club_id": str(origin),
        "joined_club_id": str(destination),
    }

st.sidebar.divider()
st.sidebar.caption("Model: CatBoost depth 8 · 1,000 trees · log1p target")
st.sidebar.caption("Important: fee units follow the original dataset and are not independently verified here.")

st.markdown("# ⚽ Football Transfer Intelligence 2.0")
st.caption("Historical transfer-fee prediction · Temporal validation · Explainable AI")

cat_row = results.loc[results["Model"] == "CatBoost"]
base_row = results.loc[results["Model"] == "Median Baseline"]
if cat_row.empty or base_row.empty:
    st.error("Baris CatBoost atau Median Baseline tidak tersedia.")
    st.stop()
cat_mae = float(cat_row.iloc[0]["MAE"])
cat_r2 = float(cat_row.iloc[0]["R2"])
base_mae = float(base_row.iloc[0]["MAE"])
improvement = 100 * (base_mae - cat_mae) / base_mae

kpi = st.columns(4)
kpi[0].metric("Final test MAE", f"{cat_mae:,.0f}")
kpi[1].metric("Final test R²", f"{cat_r2:.4f}")
kpi[2].metric("MAE vs baseline", f"{improvement:.2f}% better")
kpi[3].metric("Final test transactions", "5,332")

market, simulator, laboratory, explain = st.tabs([
    "📈 Market Explorer",
    "🧮 Fee Simulator",
    "🧪 Model Laboratory",
    "🔍 Explainable AI",
])


def style(fig: go.Figure, height: int = 450) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["navy"],
        plot_bgcolor=COLORS["panel"],
        font={"color": "#DCE8F5"},
        height=height,
        margin={"l": 30, "r": 25, "t": 45, "b": 40},
    )
    return fig


with market:
    st.subheader("How does reported transfer-fee coverage change over time?")
    available_years = pd.to_numeric(coverage["year"], errors="coerce").dropna().astype(int)
    min_year, max_year = int(available_years.min()), int(available_years.max())
    years = st.slider("Filter years", min_year, max_year, (min_year, max_year), key="market_years")
    selected_metric = st.radio(
        "Show metric",
        ["Positive-fee share (%)", "Positive-fee records", "All non-loan records"],
        horizontal=True,
    )
    col_map = {
        "Positive-fee share (%)": "positive_fee_pct",
        "Positive-fee records": "positive_fee_records",
        "All non-loan records": "total_records",
    }
    shown = coverage.loc[coverage["year"].between(*years)].copy()
    fig = px.line(
        shown,
        x="year",
        y=col_map[selected_metric],
        markers=True,
        custom_data=["positive_fee_records", "total_records", "positive_fee_pct"],
        labels={"year": "Year", col_map[selected_metric]: selected_metric},
    )
    fig.update_traces(
        line={"color": COLORS["teal"], "width": 3},
        marker={"size": 9},
        hovertemplate=(
            "Year: %{x}<br>Positive-fee records: %{customdata[0]:,.0f}"
            "<br>Non-loan records: %{customdata[1]:,.0f}"
            "<br>Positive-fee share: %{customdata[2]:.2f}%<extra></extra>"
        ),
    )
    fig.update_xaxes(dtick=1)
    st.plotly_chart(style(fig), use_container_width=True)
    with st.expander("📖 Cara membaca grafik ini"):
        st.write(
            "Grafik mengukur persentase atau jumlah **catatan** non-loan yang memiliki fee positif, "
            "bukan persentase semua transfer dunia yang berbayar. Fee bernilai nol tidak otomatis berarti gratis; "
            "cakupan pencatatan dataset bisa berubah menurut tahun."
        )

with simulator:
    st.subheader("Estimate the fee of a hypothetical paid transfer")
    st.caption("Isi empat input di sidebar, kemudian tekan **Estimate transfer fee**.")
    scenario = st.session_state.get("scenario")
    if scenario:
        frame = pd.DataFrame([scenario], columns=FEATURES)
        for cat_col in CATEGORICAL:
            frame[cat_col] = frame[cat_col].astype(str)
        prediction_log = float(model.predict(frame)[0])
        estimated_fee = max(float(np.expm1(prediction_log)), 0.0)
        st.metric("Estimated recorded transfer fee", f"{estimated_fee:,.0f}")
        st.caption("Estimated numeric fee in the dataset's original units — not a live valuation or guaranteed market price.")
        if scenario["left_club_id"] == scenario["joined_club_id"]:
            st.warning("Origin and destination club IDs are the same; this scenario may be unrealistic.")
        st.write("**Scenario used by the model**")
        st.dataframe(frame.rename(columns={
            "year": "Year", "age_approx": "Age", "left_club_id": "Origin club ID", "joined_club_id": "Destination club ID"
        }), hide_index=True, use_container_width=True)
        st.info(
            "This model was trained only on **recorded positive-fee, non-loan transfers**. "
            "It does not predict whether a transfer will occur, whether it will be free, or a player's current market value. "
            "Club IDs are shown until a trustworthy ID-to-name table is added."
        )
    else:
        st.info("Choose a scenario in the sidebar to enable prediction. The app runs the saved CatBoost model, not a canned answer.")

with laboratory:
    st.subheader("Final holdout comparison · transfers from 2022–2023")
    metric = st.selectbox("Choose evaluation metric", ["MAE", "RMSE", "R2"])
    ordered = results.sort_values(metric, ascending=(metric != "R2")).copy()
    fig = px.bar(
        ordered,
        x="Model", y=metric, color="Model", text=metric,
        color_discrete_map={
            "CatBoost": COLORS["teal"],
            "Ridge Regression": COLORS["blue"],
            "Random Forest": COLORS["purple"],
            "Median Baseline": COLORS["orange"],
        },
    )
    fig.update_traces(texttemplate="%{text:,.0f}" if metric != "R2" else "%{text:.3f}", textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False)
    if metric != "R2":
        fig.update_yaxes(range=[0, float(ordered[metric].max()) * 1.2])
    st.plotly_chart(style(fig), use_container_width=True)
    st.dataframe(
        ordered.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "R2": "{:.4f}"}),
        hide_index=True, use_container_width=True,
    )
    with st.expander("📖 Bagaimana membaca MAE, RMSE, dan R²?"):
        st.markdown(
            "**MAE:** rata-rata kesalahan absolut pada satuan biaya asli — makin kecil makin baik.  \n"
            "**RMSE:** lebih sensitif terhadap kesalahan ekstrem — makin kecil makin baik.  \n"
            "**R²:** ukuran kemampuan menjelaskan variasi target relatif terhadap baseline berbasis rata-rata; "
            "bukan persentase akurasi."
        )
    st.caption("Final test digunakan satu kali setelah konfigurasi model ditentukan melalui temporal validation.")

with explain:
    st.subheader("Which inputs influence CatBoost predictions?")
    show = shap_global.sort_values("Mean_Absolute_SHAP", ascending=True).copy()
    fig = px.bar(
        show, x="Mean_Absolute_SHAP", y="Feature", orientation="h",
        color_discrete_sequence=[COLORS["teal"]],
        labels={"Mean_Absolute_SHAP": "Mean absolute SHAP (log-target scale)"},
    )
    st.plotly_chart(style(fig), use_container_width=True)
    with st.expander("📖 Interpretasi SHAP global"):
        st.write(
            "Semakin tinggi mean absolute SHAP, semakin besar kontribusi fitur dalam mengubah output model "
            "pada sampel yang dianalisis. Nilai berada pada skala **log1p fee**, bukan dalam satuan uang. "
            "SHAP tidak menunjukkan sebab-akibat."
        )

    scenario = st.session_state.get("scenario")
    if scenario:
        st.markdown("#### SHAP for your selected scenario")
        frame = pd.DataFrame([scenario], columns=FEATURES)
        for cat_col in CATEGORICAL:
            frame[cat_col] = frame[cat_col].astype(str)
        pool = Pool(frame, cat_features=CATEGORICAL)
        contrib = model.get_feature_importance(data=pool, type="ShapValues")[0]
        local_df = pd.DataFrame({
            "Feature": FEATURES,
            "SHAP contribution": contrib[:-1],
        })
        local_df["Direction"] = np.where(local_df["SHAP contribution"] >= 0, "Raises log prediction", "Lowers log prediction")
        local_df = local_df.sort_values("SHAP contribution")
        fig = px.bar(
            local_df, x="SHAP contribution", y="Feature", orientation="h", color="Direction",
            color_discrete_map={"Raises log prediction": COLORS["teal"], "Lowers log prediction": COLORS["orange"]},
        )
        st.plotly_chart(style(fig), use_container_width=True)
        predicted_log = float(model.predict(pool)[0])
        st.caption(
            f"Base log prediction: {contrib[-1]:.3f} · Sum of contributions: {sum(contrib[:-1]):.3f} "
            f"· Final log prediction: {predicted_log:.3f}."
        )
    else:
        st.info("Run a scenario from the sidebar to view local SHAP contributions for that prediction.")

st.divider()
st.caption(
    "Educational research demo · Only historically recorded positive-fee non-loan transfers · "
    "No real-time prices · Do not interpret predictions as a live market quote."
)
