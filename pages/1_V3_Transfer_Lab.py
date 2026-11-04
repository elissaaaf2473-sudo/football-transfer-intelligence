
# ============================================================
# FOOTBALL TRANSFER INTELLIGENCE V3
# STREAMLIT MULTIPAGE APPLICATION
# File: pages/1_V3_Transfer_Lab.py
# ============================================================

from pathlib import Path
import html
import json
import re
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from catboost import CatBoostRegressor, Pool


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Football Transfer Intelligence V3",
    page_icon="⚽",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = ROOT / "model_v3.cbm"
META_FILE = ROOT / "model_metadata_v3.json"
DATA_FILE = ROOT / "player_scenarios_v3.csv"

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        max-width: 1300px;
    }

    .big-title {
        font-size: 2.05rem;
        font-weight: 750;
        line-height: 1.2;
    }

    .muted-note {
        font-size: 0.88rem;
        opacity: 0.75;
    }

    .player-face {
        width: 144px;
        height: 176px;
        object-fit: cover;
        border: 1px solid #64748b44;
        border-radius: 16px;
        background: #101827;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 2. DATA AND MODEL LOADING
# ============================================================

@st.cache_resource(
    show_spinner="Memuat model CatBoost V3..."
)
def load_model(path):
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


@st.cache_data(
    show_spinner="Memuat katalog transfer..."
)
def load_catalog(path):

    df = pd.read_csv(
        path,
        dtype={
            "player_id": "string",
            "from_club_id": "string",
            "to_club_id": "string"
        },
        low_memory=False
    )

    df["transfer_date"] = pd.to_datetime(
        df["transfer_date"],
        errors="coerce"
    )

    df["player_name"] = (
        df["player_name"]
        .fillna("Unknown player")
        .astype(str)
    )

    for field in [
        "player_id",
        "from_club_id",
        "to_club_id"
    ]:
        df[field] = df[field].str.replace(
            r"\.0$",
            "",
            regex=True
        )

    return (
        df.sort_values(
            "transfer_date",
            na_position="first"
        )
        .reset_index(drop=True)
    )


@st.cache_data
def load_metadata(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


required_files = [
    MODEL_FILE,
    META_FILE,
    DATA_FILE
]

missing_files = [
    p.name
    for p in required_files
    if not p.is_file()
]

if missing_files:
    st.error(
        "File V3 belum ditemukan di root repository: "
        + ", ".join(missing_files)
    )
    st.stop()


try:
    model = load_model(MODEL_FILE)
    metadata = load_metadata(META_FILE)
    catalog = load_catalog(DATA_FILE)

except Exception as exc:
    st.error(
        f"Gagal membaca file V3: {exc}"
    )
    st.stop()


# ============================================================
# 3. FEATURE VALIDATION
# ============================================================

FEATURES = list(model.feature_names_)

CAT_COLS = [
    "from_club_id",
    "to_club_id"
]

missing_fields = [
    column
    for column in FEATURES
    if column not in catalog.columns
]

if missing_fields:
    st.error(
        "Kolom fitur hilang dari katalog: "
        + ", ".join(missing_fields)
    )
    st.stop()


def feature_frame(df):

    # Mengikuti preprocessing yang terverifikasi
    # pada Cell 53 Google Colab.

    x = df[FEATURES].copy()

    for col in CAT_COLS:
        x[col] = (
            pd.to_numeric(
                x[col],
                errors="coerce"
            )
            .astype("Int64")
            .astype(str)
        )

    for col in FEATURES:

        if col in CAT_COLS:
            continue

        values = x[col]

        if (
            values.dtype == "object"
            or pd.api.types.is_string_dtype(values)
        ):
            values = values.replace({
                "True": 1,
                "False": 0,
                "true": 1,
                "false": 0
            })

        x[col] = pd.to_numeric(
            values,
            errors="coerce"
        ).astype(float)

    return x


def predict_fee(df):

    x = feature_frame(df)

    raw = np.asarray(
        model.predict(x),
        dtype=float
    )

    prediction = np.maximum(raw, 0)

    return prediction, raw, x


# ============================================================
# 4. DISPLAY HELPERS
# ============================================================

def euros(value):

    if pd.isna(value):
        return "Tidak tersedia"

    value = float(value)

    if abs(value) >= 1_000_000:
        return (
            f"€{value / 1_000_000:,.2f} juta"
        )

    return f"€{value:,.0f}"


def num(value, decimals=0):

    if pd.isna(value):
        return "Tidak tersedia"

    return f"{float(value):,.{decimals}f}"


def club_text(value):

    if (
        pd.isna(value)
        or not str(value).strip()
    ):
        return "Klub tidak diketahui"

    return str(value)


def show_face(player_id, name):

    # Foto dari host eksternal.
    # Jika gagal, gunakan ilustrasi cadangan.

    number = re.sub(
        r"\D",
        "",
        str(player_id)
    )

    placeholder = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="144" height="176" '
        'viewBox="0 0 144 176">'
        '<rect width="144" height="176" rx="16" '
        'fill="#18243a"/>'
        '<circle cx="72" cy="59" r="27" '
        'fill="#8b9db4"/>'
        '<path d="M24 164q4-58 48-58t48 58" '
        'fill="#8b9db4"/>'
        '</svg>'
    )

    fallback = (
        "data:image/svg+xml;charset=utf-8,"
        + quote(placeholder)
    )

    image_url = (
        "https://img.a.transfermarkt.technology/"
        f"portrait/medium/{number}.jpg?lm=1"
        if number
        else fallback
    )

    st.markdown(
        '<img class="player-face" src="'
        + html.escape(image_url, quote=True)
        + '" alt="Foto '
        + html.escape(name, quote=True)
        + '" onerror="this.onerror=null;this.src=\''
        + fallback
        + '\';">',
        unsafe_allow_html=True
    )

    st.caption(
        "Foto eksternal; jika tidak tersedia "
        "akan ditampilkan ilustrasi."
    )


# ============================================================
# 5. DASHBOARD HEADER
# ============================================================

st.markdown(
    '<div class="big-title">'
    '⚽ Football Transfer Intelligence · V3 Lab'
    '</div>',
    unsafe_allow_html=True
)

st.caption(
    "Historical transfer fee estimator · CatBoost · "
    "Data pengembangan 2013–2023 · "
    "Evaluasi 2024–2025"
)

st.info(
    "Prediksi adalah estimasi berdasarkan rekam "
    "historis sebelum tanggal transfer pada katalog, "
    "**bukan valuasi terkini 2026**, tawaran resmi, "
    "atau biaya transfer yang pasti."
)


metrics = metadata.get(
    "final_test_metrics",
    {}
)

a, b, c, d = st.columns(4)

a.metric(
    "Final Test MAE",
    euros(metrics.get("MAE_EUR", float("nan")))
)

b.metric(
    "Final Test R²",
    num(metrics.get("R2", float("nan")), 4)
)

c.metric(
    "Final Test WAPE",
    num(
        metrics.get("WAPE_pct", float("nan")),
        2
    ) + "%"
)

d.metric(
    "Skenario historis",
    f"{len(catalog):,}"
)


# ============================================================
# 6. APPLICATION TABS
# ============================================================

tab_search, tab_shap, tab_eval = st.tabs([
    "🔎 Jelajahi & Simulasikan",
    "🧠 Explainable AI",
    "📊 Evaluasi Model"
])


# ============================================================
# 7. PLAYER SEARCH SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🔍 Cari Pemain")

    query = st.text_input(
        "Nama pemain",
        placeholder="Contoh: Haaland",
        key="v3_query"
    ).strip()

    candidates = catalog

    if query:
        candidates = candidates[
            candidates["player_name"].str.contains(
                re.escape(query),
                case=False,
                na=False
            )
        ]
    else:
        st.caption(
            "Ketik nama pemain untuk menyaring "
            "katalog historis."
        )

    players = candidates.drop_duplicates(
        subset=["player_id", "player_name"],
        keep="last"
    )

    players = (
        players.sort_values("player_name")
        .head(100)
        .reset_index(drop=True)
    )

    if players.empty:
        st.warning(
            "Nama pemain tidak ditemukan "
            "dalam katalog historis."
        )
        st.stop()

    options = list(players.index)

    selected_index = st.selectbox(
        "Hasil pencarian (maksimal 100)",
        options,
        format_func=lambda i: (
            f"{players.loc[i, 'player_name']} "
            f"· ID {players.loc[i, 'player_id']}"
        ),
        key="v3_player"
    )

    selected_player = players.loc[
        selected_index
    ]

    player_records = catalog[
        (
            catalog["player_id"]
            == selected_player["player_id"]
        )
        &
        (
            catalog["player_name"]
            == selected_player["player_name"]
        )
    ].copy()

    player_records = (
        player_records.sort_values(
            "transfer_date",
            ascending=False
        )
        .reset_index(drop=True)
    )

    row_index = st.selectbox(
        "Transfer historis",
        list(player_records.index),
        format_func=lambda i: (
            f"{player_records.loc[i, 'transfer_date']:%Y-%m-%d}"
            f" · {club_text(player_records.loc[i, 'from_club_name'])}"
            f" → {club_text(player_records.loc[i, 'to_club_name'])}"
        )
        if pd.notna(
            player_records.loc[i, "transfer_date"]
        )
        else "Tanggal tak tersedia",
        key="v3_scenario"
    )

    row = player_records.iloc[row_index]


original_df = (
    player_records.iloc[[row_index]].copy()
)

original_pred, original_raw, original_x = (
    predict_fee(original_df)
)


# ============================================================
# 8. PLAYER PROFILE AND PREDICTION
# ============================================================

with tab_search:

    photo, description = st.columns(
        [1, 4],
        vertical_alignment="center"
    )

    with photo:
        show_face(
            row["player_id"],
            row["player_name"]
        )

    with description:

        st.subheader(
            str(row["player_name"])
        )

        if pd.notna(row["transfer_date"]):
            date_text = row[
                "transfer_date"
            ].strftime("%d %B %Y")
        else:
            date_text = "Tidak tersedia"

        st.write(
            "**Transfer:** "
            f"{club_text(row['from_club_name'])}"
            " → "
            f"{club_text(row['to_club_name'])}"
        )

        st.caption(
            f"Tanggal transfer historis: {date_text}"
            f" · ID pemain: {row['player_id']}"
        )

        st.metric(
            "Estimasi Biaya Transfer Historis",
            euros(original_pred[0])
        )

        if original_raw[0] < 0:
            st.warning(
                "Prediksi mentah bernilai negatif. "
                "Estimasi ditampilkan sebagai €0 "
                "sesuai metode evaluasi."
            )

    st.divider()

    st.subheader(
        "📋 Kondisi Pemain Sebelum Transfer"
    )

    stat1, stat2, stat3, stat4 = st.columns(4)

    age = row["age_at_transfer"]

    stat1.metric(
        "Usia",
        (
            num(age, 1) + " tahun"
            if pd.notna(age)
            else "Tidak tersedia"
        )
    )

    stat2.metric(
        "Market Value Historis",
        euros(
            row["mv_pretransfer_strict_eur"]
        )
    )

    stat3.metric(
        "Menit Bermain 365 Hari",
        num(row["minutes_365d"])
    )

    if (
        pd.notna(row["goals_365d"])
        and pd.notna(row["assists_365d"])
    ):
        goal_assist = (
            row["goals_365d"]
            + row["assists_365d"]
        )
    else:
        goal_assist = np.nan

    stat4.metric(
        "Gol + Assist 365 Hari",
        num(goal_assist)
    )

    availability = pd.to_numeric(
        pd.Series([
            row["mv_strict_available"]
        ]),
        errors="coerce"
    ).iloc[0]

    if (
        pd.isna(availability)
        or availability != 1
    ):
        st.warning(
            "Market value yang memenuhi aturan "
            "kecocokan klub dan waktu tidak tersedia."
        )

    if pd.isna(row["minutes_365d"]):
        st.caption(
            "Statistik performa tidak tersedia. "
            "Nilai kosong tidak dianggap nol."
        )


    # ========================================================
    # 9. WHAT-IF SIMULATION
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Simulasi Klub Tujuan Alternatif"
    )

    st.caption(
        "Mengganti klub tujuan pada tanggal transfer "
        "historis yang sama. "
        "Usia, klub asal, market value, dan "
        "performa pemain tetap. "
        "Ini bukan proyeksi transfer tahun 2026."
    )

    clubs_from = catalog[
        [
            "from_club_id",
            "from_club_name"
        ]
    ].rename(
        columns={
            "from_club_id": "id",
            "from_club_name": "name"
        }
    )

    clubs_to = catalog[
        [
            "to_club_id",
            "to_club_name"
        ]
    ].rename(
        columns={
            "to_club_id": "id",
            "to_club_name": "name"
        }
    )

    clubs = pd.concat(
        [clubs_from, clubs_to],
        ignore_index=True
    ).dropna()

    clubs = (
        clubs.drop_duplicates("id")
        .sort_values("name")
        .reset_index(drop=True)
    )

    club_options = list(clubs.index)

    destination_i = st.selectbox(
        "Pilih klub tujuan alternatif",
        club_options,
        format_func=lambda i: (
            f"{clubs.loc[i, 'name']}"
            f" · ID {clubs.loc[i, 'id']}"
        )
    )

    alternative = original_df.copy()

    alternative.loc[
        :,
        "to_club_id"
    ] = clubs.loc[destination_i, "id"]

    alternative_pred, _, _ = predict_fee(
        alternative
    )

    sim1, sim2, sim3 = st.columns(3)

    sim1.metric(
        "Skenario Historis",
        euros(original_pred[0])
    )

    sim2.metric(
        "Klub Alternatif",
        euros(alternative_pred[0])
    )

    sim3.metric(
        "Selisih Estimasi",
        euros(
            alternative_pred[0]
            - original_pred[0]
        )
    )

    st.caption(
        "Perbedaan prediksi menunjukkan perubahan "
        "output model ketika fitur klub tujuan "
        "diganti. Ini bukan bukti efek kausal."
    )


    # ========================================================
    # 10. HISTORICAL TRANSFER SCENARIOS
    # ========================================================

    st.divider()

    st.subheader(
        "📈 Riwayat Skenario Transfer"
    )

    history_pred, _, _ = predict_fee(
        player_records
    )

    history = player_records[
        [
            "transfer_date",
            "from_club_name",
            "to_club_name"
        ]
    ].copy()

    history["Estimasi EUR"] = history_pred

    history["transfer_date"] = (
        history["transfer_date"]
        .dt.strftime("%Y-%m-%d")
    )

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )

    if len(history) > 1:

        graph = history.sort_values(
            "transfer_date"
        )

        fig = px.line(
            graph,
            x="transfer_date",
            y="Estimasi EUR",
            markers=True,
            title=(
                "Estimasi Model Menurut "
                "Skenario Historis Pemain"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# 11. EXPLAINABLE AI - SHAP
# ============================================================

with tab_shap:

    st.subheader(
        "🧠 Explainable AI — SHAP"
    )

    st.caption(
        "SHAP menunjukkan kontribusi fitur "
        "terhadap output CatBoost. "
        "Ini bukan bukti hubungan sebab-akibat."
    )

    if st.button(
        "Hitung SHAP untuk Transfer Ini",
        type="primary"
    ):

        try:

            shap_values = (
                model.get_feature_importance(
                    Pool(
                        original_x,
                        cat_features=CAT_COLS
                    ),
                    type="ShapValues"
                )
            )

            vec = np.asarray(
                shap_values
            )[0]

            baseline = float(vec[-1])
            contributions = vec[:-1]

            explanation = pd.DataFrame({
                "Fitur": FEATURES,
                "Kontribusi EUR": contributions
            })

            explanation = explanation.reindex(
                explanation[
                    "Kontribusi EUR"
                ].abs().sort_values(
                    ascending=False
                ).index
            )

            st.metric(
                "Output Model Sebelum Clipping",
                euros(original_raw[0])
            )

            st.caption(
                f"Nilai acuan SHAP: {euros(baseline)}"
                " · "
                f"Jumlah kontribusi: "
                f"{euros(contributions.sum())}"
            )

            fig = px.bar(
                explanation,
                x="Kontribusi EUR",
                y="Fitur",
                orientation="h",
                color="Kontribusi EUR",
                color_continuous_scale="RdBu_r",
                title=(
                    "Kontribusi Fitur "
                    "pada Satu Prediksi"
                )
            )

            fig.update_layout(
                yaxis={
                    "categoryorder":
                    "total ascending"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                explanation,
                use_container_width=True,
                hide_index=True
            )

        except Exception as exc:
            st.error(
                "SHAP tidak berhasil dihitung: "
                f"{exc}"
            )

    else:

        st.info(
            "Klik tombol di atas untuk menghitung "
            "kontribusi fitur pada skenario "
            "transfer yang sedang dipilih."
        )


# ============================================================
# 12. MODEL EVALUATION
# ============================================================

with tab_eval:

    st.subheader(
        "📊 Evaluasi Final Model V3"
    )

    final_metrics = metadata.get(
        "final_test_metrics",
        {}
    )

    evaluation = pd.DataFrame([
        {
            "Model": "CatBoost B_Raw",
            "Periode Test": "2024–2025",
            "Data Uji": metadata.get(
                "test_records"
            ),
            "MAE EUR": final_metrics.get(
                "MAE_EUR"
            ),
            "RMSE EUR": final_metrics.get(
                "RMSE_EUR"
            ),
            "R²": final_metrics.get(
                "R2"
            ),
            "WAPE %": final_metrics.get(
                "WAPE_pct"
            )
        }
    ])

    st.dataframe(
        evaluation,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "🔬 Metodologi Penelitian"
    )

    st.markdown(
        """
        **Pembagian data temporal**

        - Training awal: 2013–2021.
        - Validasi dan pemilihan model: 2022–2023.
        - Model final dilatih ulang memakai data 2013–2023.
        - Final test: 4.582 transaksi pada 2024–2025.

        **Feature engineering**

        - Market value memakai informasi sebelum tanggal transfer.
        - Statistik pertandingan dihitung dari periode 365 hari sebelum transfer.
        - Data performa yang tidak tersedia dipertahankan sebagai missing value.
        - Model menggunakan 13 fitur.
        - Identitas pemain tidak digunakan sebagai fitur prediksi.

        **Model**

        - Algoritma: CatBoost Regressor.
        - Target: biaya transfer dalam EUR.
        - Model utama: B_Raw.
        - Prediksi negatif dipotong menjadi EUR 0.
        - Prediksi merupakan estimasi historis, bukan harga transfer terkini.
        """
    )

    st.warning(
        "Keterbatasan model: MAE sekitar €1,91 juta "
        "dan WAPE sekitar 40%. "
        "Prediksi individu masih dapat meleset jauh, "
        "terutama pada transfer bernilai tinggi. "
        "R² sebesar 0,7995 bukan berarti "
        "akurasi 79,95%."
    )

    st.caption(
        "Sumber evaluasi: model dan metadata V3 "
        "yang dihasilkan dari notebook penelitian."
    )

# ============================================================
# END OF V3 STREAMLIT PAGE
# ============================================================
