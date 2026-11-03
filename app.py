
from pathlib import Path
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

COLORS = {
    "teal": "#2DD4BF",
    "blue": "#60A5FA",
    "orange": "#F59E0B",
    "purple": "#A78BFA"
}

st.markdown("""
<style>
.stApp {
    background-color: #0B1220;
}
[data-testid="stSidebar"] {
    background-color: #111B2A;
}
.block-container {
    padding-top: 1.6rem;
    max-width: 1350px;
}
[data-testid="stMetric"] {
    background: #132238;
    border: 1px solid #263951;
    border-radius: 14px;
    padding: 15px;
}
[data-testid="stMetricValue"] {
    color: #2DD4BF;
}
h1, h2, h3 {
    letter-spacing: -.02em;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 02 - REQUIRED DATA FILES
# ==========================================

required_files = [
    "club_catalog_named.csv",
    "player_scenarios.csv",
    "coverage.csv",
    "final_test_results.csv",
    "shap_importance.csv"
]

missing = [
    name for name in required_files
    if not (ROOT / name).is_file()
]

if missing:
    st.error(
        "File belum tersedia di GitHub: "
        + ", ".join(missing)
    )
    st.info(
        "Upload CSV dari Google Drive → "
        "Football_ML → research_outputs."
    )
    st.stop()


# ==========================================
# 03 - LOAD DATA
# ==========================================

@st.cache_data(show_spinner=False)
def load_tables():

    clubs = pd.read_csv(
        ROOT / "club_catalog_named.csv",
        dtype={"club_id": "string"}
    )

    scenarios = pd.read_csv(
        ROOT / "player_scenarios.csv",
        dtype={
            "player_id": "string",
            "left_club_id": "string",
            "joined_club_id": "string"
        }
    )

    coverage = pd.read_csv(
        ROOT / "coverage.csv"
    )

    results = pd.read_csv(
        ROOT / "final_test_results.csv"
    )

    shap = pd.read_csv(
        ROOT / "shap_importance.csv"
    )

    return clubs, scenarios, coverage, results, shap


clubs, scenarios, coverage, results, shap_global = (
    load_tables()
)

required_columns = [
    (
        clubs,
        {"club_id", "club_name", "record_count"},
        "club_catalog_named.csv"
    ),
    (
        scenarios,
        {
            "player_id",
            "player_name",
            "year",
            "age_approx",
            "left_club_id",
            "joined_club_id",
            "origin_name",
            "destination_name"
        },
        "player_scenarios.csv"
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
        results,
        {"Model", "MAE", "RMSE", "R2"},
        "final_test_results.csv"
    ),
    (
        shap_global,
        {"Feature", "Mean_Absolute_SHAP"},
        "shap_importance.csv"
    )
]

for table, needed, filename in required_columns:
    if not needed.issubset(table.columns):
        st.error(
            f"Kolom {filename} tidak sesuai: "
            f"{sorted(needed - set(table.columns))}"
        )
        st.stop()


# ==========================================
# 04 - PREPROCESS IDENTIFIERS
# ==========================================

def clean_ids(series):
    return (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .astype("Int64")
        .astype("string")
    )


clubs["club_id"] = clean_ids(
    clubs["club_id"]
)

clubs["club_name"] = (
    clubs["club_name"]
    .fillna("Klub tidak diketahui")
    .astype(str)
)

clubs["record_count"] = pd.to_numeric(
    clubs["record_count"],
    errors="coerce"
).fillna(0)

clubs = (
    clubs.dropna(subset=["club_id"])
    .drop_duplicates("club_id")
    .sort_values(
        "record_count",
        ascending=False
    )
    .reset_index(drop=True)
)

for col in [
    "player_id",
    "left_club_id",
    "joined_club_id"
]:
    scenarios[col] = clean_ids(
        scenarios[col]
    )

for col in ["year", "age_approx"]:
    scenarios[col] = pd.to_numeric(
        scenarios[col],
        errors="coerce"
    )

for col in [
    "player_name",
    "origin_name",
    "destination_name"
]:
    scenarios[col] = (
        scenarios[col]
        .fillna("Tidak diketahui")
        .astype(str)
    )

scenarios = scenarios.dropna(
    subset=[
        "player_id",
        "year",
        "age_approx",
        "left_club_id",
        "joined_club_id"
    ]
)

scenarios = scenarios[
    scenarios["year"].between(2010, 2021)
].copy()

if clubs.empty or scenarios.empty:
    st.error(
        "Katalog nama klub atau "
        "skenario pemain kosong."
    )
    st.stop()


# ==========================================
# 05 - HUMAN READABLE CLUB NAMES
# ==========================================

names = dict(
    zip(
        clubs["club_id"].astype(str),
        clubs["club_name"]
    )
)

name_frequency = (
    clubs["club_name"].value_counts()
)

def club_label(club_id):

    text = names.get(
        str(club_id),
        "Klub tidak diketahui"
    )

    if text in [
        "Unknown Club",
        "Klub tidak diketahui"
    ]:
        return (
            f"Klub belum teridentifikasi "
            f"(ID {club_id})"
        )

    if name_frequency.get(text, 0) > 1:
        return f"{text} (ID {club_id})"

    return text


club_ids = (
    clubs["club_id"]
    .astype(str)
    .tolist()
)


# ==========================================
# 06 - LOAD SPLIT CATBOOST MODEL
# ==========================================

@st.cache_resource(
    show_spinner="Memuat model CatBoost..."
)
def load_model():

    original = (
        ROOT / "catboost_final.cbm"
    )

    model = CatBoostRegressor()

    # Jika model utuh tersedia
    if original.is_file():
        model.load_model(str(original))
        return model

    # Jika model dibagi menjadi part
    parts = sorted(
        ROOT.glob(
            "catboost_final.cbm.part*"
        )
    )

    nums = []

    for path in parts:

        suffix = path.name.rsplit(
            ".part", 1
        )[-1]

        if suffix.isdigit():
            nums.append(int(suffix))

    if (
        not parts
        or len(nums) != len(parts)
        or nums != list(
            range(1, len(parts) + 1)
        )
    ):
        raise RuntimeError(
            "File part model tidak lengkap "
            "atau tidak berurutan."
        )

    # Gabungkan semua bagian sementara
    with tempfile.NamedTemporaryFile(
        suffix=".cbm"
    ) as temporary:

        for part in parts:

            with part.open("rb") as source:
                shutil.copyfileobj(
                    source,
                    temporary
                )

        temporary.flush()

        model.load_model(
            temporary.name
        )

    return model


# ==========================================
# 07 - PREDICTION FUNCTION
# ==========================================

def predict_scenario(item, title):

    inputs = pd.DataFrame(
        [{
            "year": int(item["year"]),
            "age_approx": float(
                item["age_approx"]
            ),
            "left_club_id": str(
                item["left_club_id"]
            ),
            "joined_club_id": str(
                item["joined_club_id"]
            )
        }],
        columns=FEATURES
    )

    try:

        model = load_model()

        value_log = float(
            model.predict(inputs)[0]
        )

        value = float(
            np.expm1(value_log)
        )

        if not np.isfinite(value):
            raise ValueError(
                "Hasil prediksi tidak valid."
            )

    except Exception as exc:

        st.error(
            "Model belum bisa dijalankan. "
            "Pastikan seluruh file .part "
            "sudah terunggah."
        )

        st.exception(exc)
        return

    st.session_state[
        "last_prediction"
    ] = {
        "inputs": inputs.iloc[0].to_dict(),
        "title": title,
        "fee": max(value, 0.0)
    }


def show_prediction():

    result = st.session_state.get(
        "last_prediction"
    )

    if not result:
        return

    st.divider()

    st.subheader(
        "Hasil estimasi terakhir"
    )

    st.caption(
        result["title"]
    )

    st.metric(
        "Estimasi biaya transfer "
        "tercatat (satuan dataset)",
        f"{result['fee']:,.0f}"
    )

    st.info(
        "Ini simulasi berdasarkan tahun, "
        "usia, klub asal dan tujuan; "
        "bukan harga pasar aktual "
        "atau harga pasti seorang pemain."
    )

    st.caption(
        "Nama pemain hanya digunakan "
        "untuk mencari skenario historis. "
        "Nama dan statistik performanya "
        "bukan fitur model."
    )


# ==========================================
# 08 - PLOTLY STYLING
# ==========================================

def plot_style(fig, height=440):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#132238",
        height=height,
        margin=dict(
            l=30,
            r=25,
            t=40,
            b=40
        )
    )

    return fig


# ==========================================
# 09 - NAVIGATION
# ==========================================

st.sidebar.title(
    "⚽ Football Transfer Intelligence"
)

st.sidebar.caption(
    "Interactive historical transfer analytics"
)

page = st.sidebar.radio(
    "Menu",
    [
        "🏠 Beranda",
        "🔎 Cari Pemain",
        "⚽ Simulasi Transfer",
        "📊 Analisis & Model",
        "🧠 Explainable AI"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Penelitian: transfer permanen "
    "berbiaya positif, 2010–2023."
)

st.sidebar.caption(
    "Rentang input simulasi "
    "dibatasi pada 2010–2021."
)

cat = results.loc[
    results["Model"] == "CatBoost"
]

base = results.loc[
    results["Model"] == "Median Baseline"
]

if cat.empty or base.empty:
    st.error(
        "Hasil CatBoost atau Median "
        "Baseline tidak ditemukan."
    )
    st.stop()

cat_mae = float(
    cat.iloc[0]["MAE"]
)

cat_r2 = float(
    cat.iloc[0]["R2"]
)

base_mae = float(
    base.iloc[0]["MAE"]
)


# ==========================================
# 10 - HOME PAGE
# ==========================================

if page == "🏠 Beranda":

    st.title(
        "⚽ Football Transfer Intelligence 2.0"
    )

    st.write(
        "Jelajahi perpindahan pemain historis "
        "dan simulasikan biaya transfer "
        "menggunakan CatBoost."
    )

    a, b, c = st.columns(3)

    a.metric(
        "Final test MAE",
        f"{cat_mae:,.0f}"
    )

    b.metric(
        "Final test R²",
        f"{cat_r2:.4f}"
    )

    c.metric(
        "MAE lebih rendah dari baseline",
        f"{(1-cat_mae/base_mae)*100:.2f}%"
    )

    st.markdown(
        "### Mulai dari mana?"
    )

    st.markdown(
        "**🔎 Cari Pemain:** ketik nama "
        "lalu pilih transaksi historis.  \n"
        "**⚽ Simulasi Transfer:** pilih "
        "dua nama klub, tahun dan usia.  \n"
        "**📊 Analisis & Model:** lihat "
        "grafik serta panduan membacanya.  \n"
        "**🧠 Explainable AI:** pahami "
        "fitur yang memengaruhi prediksi."
    )

    st.warning(
        "Model tidak mempelajari performa "
        "individual pemain. Prediksi di sini "
        "bukan nilai pasar terkini."
    )


# ==========================================
# 11 - SEARCH HISTORICAL PLAYER
# ==========================================

elif page == "🔎 Cari Pemain":

    st.title(
        "🔎 Cari Pemain Historis"
    )

    st.write(
        "Cari nama pemain, pilih catatan "
        "perpindahannya, lalu jalankan "
        "simulasi model."
    )

    player_list = (
        scenarios.groupby(
            "player_id",
            as_index=False
        )
        .agg(
            player_name=(
                "player_name",
                "first"
            ),
            jumlah=(
                "player_id",
                "size"
            )
        )
        .sort_values(
            ["jumlah", "player_name"],
            ascending=[False, True]
        )
    )

    search = st.text_input(
        "Cari nama pemain",
        placeholder=(
            "Contoh: Berbatov, Ronaldo, "
            "Ziegler..."
        )
    )

    if search.strip():

        found = player_list[
            player_list[
                "player_name"
            ].str.contains(
                search.strip(),
                case=False,
                regex=False,
                na=False
            )
        ].head(100)

    else:

        found = player_list.head(100)

        st.caption(
            "Menampilkan hingga 100 pemain "
            "teratas. Ketik nama untuk "
            "mencari pemain lainnya."
        )

    if found.empty:

        st.warning(
            "Nama tidak ditemukan dalam "
            "skenario development 2010–2021."
        )

    else:

        players_map = found.set_index(
            "player_id"
        )

        player_id = st.selectbox(
            "Pilih pemain",
            found[
                "player_id"
            ].astype(str).tolist(),
            format_func=lambda pid: (
                f"{players_map.loc[pid, 'player_name']} "
                f"· {int(players_map.loc[pid, 'jumlah'])} "
                "catatan"
            )
        )

        history = scenarios[
            scenarios[
                "player_id"
            ].astype(str) == player_id
        ].copy()

        history = (
            history.sort_values(
                "year",
                ascending=False
            )
            .reset_index(drop=True)
        )

        chosen_idx = st.selectbox(
            "Pilih riwayat transfer",
            list(range(len(history))),
            format_func=lambda i: (
                f"{int(history.loc[i, 'year'])}"
                f" · {history.loc[i, 'origin_name']}"
                f" → {history.loc[i, 'destination_name']}"
                f" · usia "
                f"{history.loc[i, 'age_approx']:.0f}"
            )
        )

        selected = history.loc[
            chosen_idx
        ]

        st.subheader(
            selected["player_name"]
        )

        st.write(
            f"**{selected['origin_name']} "
            f"→ {selected['destination_name']}**"
        )

        st.caption(
            f"Tahun {int(selected['year'])} "
            f"· Usia pada data: "
            f"{selected['age_approx']:.0f} tahun"
        )

        if st.button(
            "⚡ Prediksi skenario ini",
            type="primary"
        ):

            predict_scenario(
                selected,
                (
                    f"{selected['player_name']} "
                    f"· {int(selected['year'])}: "
                    f"{selected['origin_name']} "
                    f"→ {selected['destination_name']}"
                )
            )

        show_prediction()


# ==========================================
# 12 - CUSTOM TRANSFER SIMULATOR
# ==========================================

elif page == "⚽ Simulasi Transfer":

    st.title(
        "⚽ Simulasi Transfer Sendiri"
    )

    st.write(
        "Pilih nama klub, bukan ID. "
        "Data klub berasal dari katalog "
        "historis penelitian."
    )

    with st.form("custom_prediction"):

        x, y = st.columns(2)

        with x:

            origin = st.selectbox(
                "Klub asal",
                club_ids,
                format_func=club_label
            )

            year = st.slider(
                "Tahun skenario historis",
                2010,
                2021,
                2021
            )

        with y:

            destination = st.selectbox(
                "Klub tujuan",
                club_ids,
                index=(
                    1 if len(club_ids) > 1
                    else 0
                ),
                format_func=club_label
            )

            age = st.number_input(
                "Usia pemain",
                min_value=15,
                max_value=45,
                value=24,
                step=1
            )

        submit = st.form_submit_button(
            "⚡ Estimate Transfer Fee",
            type="primary",
            use_container_width=True
        )

    if submit:

        if origin == destination:

            st.warning(
                "Klub asal dan tujuan harus "
                "berbeda agar skenario masuk akal."
            )

        else:

            predict_scenario(
                {
                    "year": year,
                    "age_approx": age,
                    "left_club_id": origin,
                    "joined_club_id": destination
                },
                (
                    f"{club_label(origin)} "
                    f"→ {club_label(destination)} "
                    f"· {year} · usia {age}"
                )
            )

    show_prediction()


# ==========================================
# 13 - ANALYTICS AND MODEL LABORATORY
# ==========================================

elif page == "📊 Analisis & Model":

    st.title(
        "📊 Analisis Pasar & Evaluasi Model"
    )

    st.subheader(
        "1. Data coverage berdasarkan tahun"
    )

    coverage["year"] = pd.to_numeric(
        coverage["year"],
        errors="coerce"
    )

    coverage = (
        coverage.dropna(
            subset=["year"]
        )
        .sort_values("year")
    )

    min_y = int(
        coverage["year"].min()
    )

    max_y = int(
        coverage["year"].max()
    )

    year_range = st.slider(
        "Pilih rentang tahun",
        min_y,
        max_y,
        (min_y, max_y)
    )

    metric_options = {
        "Persentase fee positif":
            "positive_fee_pct",
        "Jumlah fee positif":
            "positive_fee_records",
        "Semua transfer non-loan tercatat":
            "total_records"
    }

    chosen = st.selectbox(
        "Tampilkan",
        list(metric_options)
    )

    shown = coverage[
        coverage[
            "year"
        ].between(*year_range)
    ]

    fig = px.line(
        shown,
        x="year",
        y=metric_options[chosen],
        markers=True,
        labels={
            "year": "Tahun",
            metric_options[chosen]: chosen
        }
    )

    fig.update_traces(
        line_color=COLORS["teal"],
        line_width=3,
        marker_size=8
    )

    fig.update_xaxes(dtick=1)

    st.plotly_chart(
        plot_style(fig),
        use_container_width=True
    )

    with st.expander(
        "📖 Cara membaca grafik ini"
    ):

        st.write(
            "Grafik menunjukkan banyaknya "
            "catatan transfer non-loan atau "
            "bagian yang memiliki fee positif "
            "tercatat. Data tanpa fee positif "
            "tidak otomatis berarti gratis."
        )

    st.subheader(
        "2. Perbandingan model "
        "pada final test 2022–2023"
    )

    metric = st.selectbox(
        "Pilih metrik evaluasi",
        ["MAE", "RMSE", "R2"]
    )

    ordered = results.sort_values(
        metric,
        ascending=(metric != "R2")
    )

    fig2 = px.bar(
        ordered,
        x="Model",
        y=metric,
        text=metric,
        color="Model"
    )

    fig2.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        plot_style(fig2),
        use_container_width=True
    )

    with st.expander(
        "📖 Arti MAE, RMSE, dan R²"
    ):

        st.markdown(
            "**MAE:** rata-rata selisih "
            "absolut prediksi dengan "
            "nilai sebenarnya. "
            "Lebih kecil lebih baik.  \n"
            "**RMSE:** lebih menghukum "
            "kesalahan besar. "
            "Lebih kecil lebih baik.  \n"
            "**R²:** ukuran variasi target "
            "yang dijelaskan dibanding "
            "prediksi konstan rata-rata; "
            "**bukan akurasi persen**."
        )

    # Grafik optional dari prediksi final
    if (
        ROOT / "final_predictions.csv"
    ).is_file():

        st.subheader(
            "3. Actual vs Predicted"
        )

        predicted = pd.read_csv(
            ROOT / "final_predictions.csv"
        )

        if {
            "actual_fee",
            "predicted_fee",
            "year"
        }.issubset(predicted.columns):

            predicted = predicted[
                (predicted["actual_fee"] > 0)
                &
                (predicted["predicted_fee"] > 0)
            ].copy()

            predicted["year"] = (
                predicted["year"].astype(str)
            )

            fig3 = px.scatter(
                predicted,
                x="actual_fee",
                y="predicted_fee",
                color="year",
                opacity=0.45,
                log_x=True,
                log_y=True,
                render_mode="webgl"
            )

            edge_low = min(
                predicted["actual_fee"].min(),
                predicted["predicted_fee"].min()
            )

            edge_high = max(
                predicted["actual_fee"].max(),
                predicted["predicted_fee"].max()
            )

            fig3.add_trace(
                go.Scatter(
                    x=[edge_low, edge_high],
                    y=[edge_low, edge_high],
                    mode="lines",
                    name="Prediksi sempurna",
                    line=dict(
                        color=COLORS["orange"],
                        dash="dash"
                    )
                )
            )

            st.plotly_chart(
                plot_style(fig3, 520),
                use_container_width=True
            )

            with st.expander(
                "📖 Cara membaca scatter plot"
            ):

                st.write(
                    "Tiap titik adalah satu transfer. "
                    "Semakin dekat garis diagonal, "
                    "semakin mendekati nilai aktual. "
                    "Di atas garis berarti prediksi "
                    "terlalu tinggi; di bawah berarti "
                    "terlalu rendah. Kedua sumbu "
                    "memakai skala logaritmik."
                )


# ==========================================
# 14 - EXPLAINABLE AI
# ==========================================

elif page == "🧠 Explainable AI":

    st.title(
        "🧠 Explainable AI — SHAP"
    )

    fig = px.bar(
        shap_global.sort_values(
            "Mean_Absolute_SHAP"
        ),
        x="Mean_Absolute_SHAP",
        y="Feature",
        orientation="h",
        color_discrete_sequence=[
            COLORS["teal"]
        ],
        labels={
            "Mean_Absolute_SHAP":
                "Rata-rata |SHAP| (skala log)"
        }
    )

    st.plotly_chart(
        plot_style(fig),
        use_container_width=True
    )

    st.info(
        "SHAP menjelaskan kontribusi fitur "
        "terhadap prediksi model, bukan "
        "pengaruh sebab-akibat. Nilainya "
        "berada pada skala log1p biaya."
    )

    prediction = st.session_state.get(
        "last_prediction"
    )

    if prediction:

        st.subheader(
            "Penjelasan prediksi terakhir"
        )

        st.caption(
            prediction["title"]
        )

        frame = pd.DataFrame(
            [prediction["inputs"]],
            columns=FEATURES
        )

        try:

            model = load_model()

            pool = Pool(
                frame,
                cat_features=CATS
            )

            shap_values = (
                model.get_feature_importance(
                    data=pool,
                    type="ShapValues"
                )[0]
            )

            local = pd.DataFrame({
                "Fitur": FEATURES,
                "Kontribusi": shap_values[:-1]
            })

            local["Arah"] = np.where(
                local["Kontribusi"] >= 0,
                "Menaikkan",
                "Menurunkan"
            )

            chart = px.bar(
                local.sort_values(
                    "Kontribusi"
                ),
                x="Kontribusi",
                y="Fitur",
                color="Arah",
                orientation="h",
                color_discrete_map={
                    "Menaikkan": COLORS["teal"],
                    "Menurunkan": COLORS["orange"]
                }
            )

            st.plotly_chart(
                plot_style(chart),
                use_container_width=True
            )

            st.caption(
                f"Nilai dasar log: "
                f"{shap_values[-1]:.3f}. "
                "Total kontribusi + nilai dasar "
                "= output log model."
            )

        except Exception as exc:

            st.error(
                "SHAP untuk skenario ini "
                "belum dapat dihitung."
            )

            st.exception(exc)

    else:

        st.info(
            "Lakukan prediksi lewat menu "
            "Cari Pemain atau Simulasi Transfer "
            "untuk melihat SHAP individual."
        )


# ==========================================
# 15 - FOOTER
# ==========================================

st.divider()

st.caption(
    "Dataset historis | Hanya transfer "
    "non-loan berbiaya positif tercatat | "
    "Model final tidak dilatih ulang "
    "di aplikasi | "
    "Bukan harga pasar real-time"
)
