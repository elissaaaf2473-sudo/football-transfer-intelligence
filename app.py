
from pathlib import Path
from urllib.parse import quote, urlparse
import html
import re
import shutil
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from catboost import CatBoostRegressor, Pool

# ==========================================
# 01 - PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Football Transfer Intelligence 2.1",
    page_icon="⚽",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent

FEATURES = [
    "year", "age_approx",
    "left_club_id", "joined_club_id"
]

CAT_FEATURES = [
    "left_club_id", "joined_club_id"
]

TEAL = "#2DD4BF"
BLUE = "#60A5FA"
ORANGE = "#F59E0B"

st.markdown("""
<style>
.stApp {
    background: #0B1220;
}
[data-testid="stSidebar"] {
    background: #111B2A;
}
.block-container {
    padding-top: 1.7rem;
    max-width: 1350px;
}
[data-testid="stMetric"] {
    background: #132238;
    border: 1px solid #263951;
    border-radius: 15px;
    padding: 16px;
}
[data-testid="stMetricValue"] {
    color: #2DD4BF;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 02 - LOAD RESEARCH DATA
# ==========================================

NEEDED = [
    "player_scenarios.csv",
    "club_catalog_named.csv",
    "coverage.csv",
    "final_test_results.csv",
    "shap_importance.csv"
]

missing = [
    f for f in NEEDED
    if not (ROOT / f).is_file()
]

if missing:
    st.error(
        "File belum ada di GitHub: "
        + ", ".join(missing)
    )
    st.stop()


@st.cache_data(show_spinner=False)
def load_tables():

    scenarios = pd.read_csv(
        ROOT / "player_scenarios.csv",
        dtype={
            "player_id": "string",
            "left_club_id": "string",
            "joined_club_id": "string"
        }
    )

    clubs = pd.read_csv(
        ROOT / "club_catalog_named.csv",
        dtype={"club_id": "string"}
    )

    coverage = pd.read_csv(
        ROOT / "coverage.csv"
    )

    results = pd.read_csv(
        ROOT / "final_test_results.csv"
    )

    shap_global = pd.read_csv(
        ROOT / "shap_importance.csv"
    )

    return (
        scenarios, clubs,
        coverage, results, shap_global
    )


scenarios, clubs, coverage, results, shap_global = (
    load_tables()
)

expected_columns = [
    (
        scenarios,
        {
            "player_id", "player_name",
            "year", "age_approx",
            "left_club_id", "joined_club_id",
            "origin_name", "destination_name"
        },
        "player_scenarios.csv"
    ),
    (
        clubs,
        {"club_id", "club_name", "record_count"},
        "club_catalog_named.csv"
    ),
    (
        coverage,
        {
            "year", "positive_fee_pct",
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

for table, required, filename in expected_columns:

    if not required.issubset(table.columns):
        st.error(
            f"Kolom {filename} kurang: "
            f"{sorted(required - set(table.columns))}"
        )
        st.stop()


def normal_id(series):
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype("string")
    )


clubs["club_id"] = normal_id(
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
    scenarios[col] = normal_id(
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
        "player_id", "year", "age_approx",
        "left_club_id", "joined_club_id"
    ]
)

scenarios = scenarios[
    scenarios["year"].between(2010, 2021)
].copy()

if clubs.empty or scenarios.empty:
    st.error("Katalog klub atau skenario kosong.")
    st.stop()

club_names = dict(
    zip(
        clubs["club_id"].astype(str),
        clubs["club_name"]
    )
)

club_ids = clubs["club_id"].astype(str).tolist()
name_freq = clubs["club_name"].value_counts()


def club_label(club_id):

    name = club_names.get(
        str(club_id),
        "Klub tidak diketahui"
    )

    if name.lower() in [
        "unknown club",
        "klub tidak diketahui"
    ]:
        return (
            f"Klub belum teridentifikasi "
            f"(ID {club_id})"
        )

    if name_freq.get(name, 0) > 1:
        return f"{name} (ID {club_id})"

    return name


# ==========================================
# 03 - LOAD ORIGINAL CATBOOST MODEL
# ==========================================

@st.cache_resource(
    show_spinner="Memuat CatBoost..."
)
def load_model():

    model = CatBoostRegressor()

    full_path = ROOT / "catboost_final.cbm"

    if full_path.exists():
        model.load_model(str(full_path))
        return model

    parts = sorted(
        ROOT.glob("catboost_final.cbm.part*")
    )

    numbers = [
        int(p.name.rsplit(".part", 1)[-1])
        for p in parts
        if p.name.rsplit(".part", 1)[-1].isdigit()
    ]

    if (
        not parts
        or len(numbers) != len(parts)
        or numbers != list(range(1, len(parts) + 1))
    ):
        raise FileNotFoundError(
            "Part model tidak lengkap "
            "atau tidak berurutan."
        )

    with tempfile.NamedTemporaryFile(
        suffix=".cbm"
    ) as tmp:

        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, tmp)

        tmp.flush()
        model.load_model(tmp.name)

    return model


# ==========================================
# 04 - PLAYER PHOTO SEARCH
# 5 Wikipedia language sources
# + Wikimedia Commons
# ==========================================

UA = (
    "FootballTransferIntelligence/2.1 "
    "(https://github.com/elissaaaf2473-sudo/"
    "football-transfer-intelligence) requests"
)

LANGS = ["en", "de", "es", "fr", "id"]

HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json"
}


def api_json(url, params):

    response = requests.get(
        url,
        params=params,
        headers=HEADERS,
        timeout=4
    )

    response.raise_for_status()

    return response.json()


def image_bytes(url):

    host = urlparse(url).hostname

    if host not in [
        "upload.wikimedia.org",
        "commons.wikimedia.org"
    ]:
        return None

    response = requests.get(
        url,
        headers={"User-Agent": UA},
        timeout=6
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type", ""
    ).lower()

    if not content_type.startswith("image/"):
        return None

    if len(response.content) > 6 * 1024 * 1024:
        return None

    return response.content


def plain_text(value):

    return " ".join(
        html.unescape(
            re.sub(r"<[^>]*>", " ", value)
        ).split()
    )


def photo_credit(filename):

    source_url = (
        "https://commons.wikimedia.org/wiki/File:"
        + quote(filename.replace(" ", "_"))
    )

    credit = (
        "Wikimedia Commons · "
        "cek lisensi pada halaman berkas"
    )

    try:

        data = api_json(
            "https://commons.wikimedia.org/w/api.php",
            {
                "action": "query",
                "format": "json",
                "prop": "imageinfo",
                "iiprop": "extmetadata|url",
                "titles": "File:" + filename
            }
        )

        item = next(
            iter(
                data.get("query", {})
                .get("pages", {})
                .values()
            ),
            {}
        )

        info = (
            item.get("imageinfo") or [{}]
        )[0]

        meta = info.get("extmetadata", {})

        artist = plain_text(
            meta.get("Artist", {}).get("value", "")
        )

        license_name = plain_text(
            meta.get(
                "LicenseShortName", {}
            ).get("value", "")
        )

        credit = " · ".join(
            x for x in [
                artist[:100], license_name
            ] if x
        ) or credit

        source_url = (
            info.get("descriptionurl")
            or source_url
        )

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        StopIteration
    ):
        pass

    return credit, source_url


def wiki_candidate(name, lang, search=False):

    api = (
        f"https://{lang}.wikipedia.org/w/api.php"
    )

    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "prop": "pageimages|info",
        "inprop": "url",
        "piprop": "thumbnail|name",
        "pithumbsize": 550,
        "pilicense": "free",
        "redirects": 1
    }

    if search:
        params.update({
            "generator": "search",
            "gsrsearch": f'"{name}" footballer',
            "gsrlimit": 4
        })

    else:
        params["titles"] = name

    data = api_json(api, params)

    pages = data.get(
        "query", {}
    ).get("pages", [])

    surname = name.strip().split()[-1].casefold()

    for page in pages:

        title = page.get(
            "title", ""
        ).casefold()

        if (
            page.get("missing")
            or not title
            or surname not in title
        ):
            continue

        if any(
            word in title
            for word in [
                "list of ",
                "category:",
                "daftar "
            ]
        ):
            continue

        thumbnail = page.get(
            "thumbnail", {}
        ).get("source")

        if thumbnail and page.get("pageimage"):
            return (
                thumbnail,
                page["pageimage"],
                page.get("fullurl", "")
            )

    return None


@st.cache_data(
    ttl=7 * 24 * 3600,
    max_entries=250,
    show_spinner=False
)
def get_player_photo(player_id, player_name):

    # Cari foto bebas pakai di Wikipedia
    for lang in LANGS:

        try:

            result = wiki_candidate(
                player_name, lang
            )

            if not result and lang == "en":
                result = wiki_candidate(
                    player_name, lang,
                    search=True
                )

            if not result:
                continue

            thumbnail, filename, article_url = result

            picture = image_bytes(thumbnail)

            if not picture:
                continue

            credit, source = photo_credit(
                filename
            )

            return {
                "bytes": picture,
                "credit": credit,
                "source": source,
                "caption": (
                    "Foto pemain, bisa berasal "
                    "dari musim berbeda"
                )
            }

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            TypeError
        ):
            continue

    # Fallback khusus Haaland
    # Foto CC BY 4.0 Wikimedia Commons
    if (
        player_name.strip().casefold()
        == "erling haaland"
    ):

        file_name = (
            "Erling Haaland June 2025.jpg"
        )

        file_url = (
            "https://commons.wikimedia.org/"
            "wiki/Special:FilePath/"
            + quote(file_name)
            + "?width=500"
        )

        try:

            picture = image_bytes(file_url)

            if picture:

                return {
                    "bytes": picture,
                    "credit": (
                        "MichaelEmilio · CC BY 4.0 "
                        "· Wikimedia Commons"
                    ),
                    "source": (
                        "https://commons.wikimedia.org/"
                        "wiki/File:Erling_Haaland_June_2025.jpg"
                    ),
                    "caption": (
                        "Foto tahun 2025, "
                        "bukan foto transfer 2019"
                    )
                }

        except requests.RequestException:
            pass

    return None


def show_player_photo(player_id, player_name):

    with st.spinner(
        "Mencari foto berlisensi bebas..."
    ):
        photo = get_player_photo(
            str(player_id),
            str(player_name)
        )

    if photo:

        st.image(
            photo["bytes"],
            width=230,
            alt=f"Foto {player_name}"
        )

        st.caption(photo["caption"])
        st.caption(
            "Kredit: " + photo["credit"]
        )

        st.markdown(
            f"[📷 Sumber foto & lisensi]"
            f"({photo['source']})"
        )

    elif (
        str(player_name).strip().casefold()
        == "erling haaland"
    ):

        # Fallback browser jika server
        # tidak bisa mengambil gambar.
        url = (
            "https://commons.wikimedia.org/"
            "wiki/Special:FilePath/"
            "Erling_Haaland_June_2025.jpg?width=500"
        )

        st.markdown(
            f'<img src="{html.escape(url, quote=True)}" '
            'alt="Erling Haaland" '
            'style="max-width:230px;width:100%;'
            'border-radius:14px;" />',
            unsafe_allow_html=True
        )

        st.caption(
            "Foto 2025 · MichaelEmilio · "
            "CC BY 4.0. Bukan foto transfer 2019."
        )

        st.markdown(
            "[📷 Sumber & lisensi]"
            "(https://commons.wikimedia.org/"
            "wiki/File:Erling_Haaland_June_2025.jpg)"
        )

    else:

        st.markdown("### 👤")

        st.caption(
            "Foto yang sesuai belum ditemukan. "
            "Tidak menampilkan foto orang lain."
        )


# ==========================================
# 05 - PREDICTION HELPERS
# ==========================================

def frame_from(values):

    return pd.DataFrame(
        [{
            "year": int(values["year"]),
            "age_approx": float(
                values["age_approx"]
            ),
            "left_club_id": str(
                values["left_club_id"]
            ),
            "joined_club_id": str(
                values["joined_club_id"]
            )
        }],
        columns=FEATURES
    )


def scenario_key(values, player_id=None):

    return (
        str(player_id or "custom"),
        str(int(values["year"])),
        str(values["left_club_id"]),
        str(values["joined_club_id"]),
        str(float(values["age_approx"]))
    )


def run_prediction(
    values,
    label,
    player_id=None
):

    try:

        model = load_model()

        frame = frame_from(values)

        result = max(
            float(
                np.expm1(
                    model.predict(frame)[0]
                )
            ),
            0.0
        )

        if not np.isfinite(result):
            raise ValueError(
                "Hasil prediksi tidak valid."
            )

        st.session_state[
            "last_prediction"
        ] = {
            "key": scenario_key(
                values, player_id
            ),
            "inputs": frame.iloc[0].to_dict(),
            "title": label,
            "amount": result,
            "player_id": player_id
        }

    except Exception as exc:

        st.error(
            "Model gagal memuat atau memprediksi. "
            "Periksa seluruh file .part."
        )

        st.exception(exc)


# ==========================================
# 06 - CHART STYLING
# ==========================================

def style_chart(fig, height=440):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#132238",
        height=height,
        margin=dict(
            l=35, r=25, t=40, b=45
        )
    )

    return fig


# ==========================================
# 07 - INDIVIDUAL SHAP
# ==========================================

def local_explanation(pred):

    try:

        model = load_model()
        frame = frame_from(pred["inputs"])

        pool = Pool(
            frame,
            cat_features=CAT_FEATURES
        )

        shap_values = (
            model.get_feature_importance(
                pool,
                type="ShapValues"
            )[0]
        )

    except Exception as exc:

        st.warning(
            f"SHAP individual belum tersedia: {exc}"
        )
        return

    labels = {
        "year": "Tahun transfer",
        "age_approx": "Usia pemain",
        "left_club_id": "Klub asal",
        "joined_club_id": "Klub tujuan"
    }

    explanation = pd.DataFrame({
        "Fitur": [
            labels[x] for x in FEATURES
        ],
        "Kontribusi": shap_values[:-1]
    }).sort_values("Kontribusi")

    explanation["Arah"] = np.where(
        explanation["Kontribusi"] >= 0,
        "Menaikkan output log",
        "Menurunkan output log"
    )

    fig = px.bar(
        explanation,
        x="Kontribusi",
        y="Fitur",
        orientation="h",
        color="Arah",
        color_discrete_map={
            "Menaikkan output log": TEAL,
            "Menurunkan output log": ORANGE
        }
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    biggest = explanation.iloc[
        explanation["Kontribusi"].abs().argmax()
    ]

    direction = (
        "menaikkan"
        if biggest["Kontribusi"] >= 0
        else "menurunkan"
    )

    st.write(
        f"**Kontribusi paling besar:** "
        f"{biggest['Fitur']} cenderung "
        f"{direction} output model relatif "
        "terhadap nilai acuannya."
    )

    st.caption(
        f"Nilai dasar output log: "
        f"{shap_values[-1]:.3f}. "
        "SHAP menjelaskan output log1p, "
        "bukan besaran uang atau sebab-akibat."
    )


# ==========================================
# 08 - PREDICTION RESULT COMPONENT
# ==========================================

def show_prediction(expected_key=None):

    pred = st.session_state.get(
        "last_prediction"
    )

    if not pred:
        return

    if (
        expected_key is not None
        and pred["key"] != expected_key
    ):
        return

    st.divider()

    st.subheader(
        "✨ AI Transfer Valuation"
    )

    st.caption(pred["title"])

    st.metric(
        "Estimasi biaya transfer "
        "(satuan dataset)",
        f"{pred['amount']:,.0f}"
    )

    st.info(
        "Estimasi berdasarkan tahun, usia, "
        "klub asal dan klub tujuan. "
        "Ini bukan nilai pasar saat ini "
        "atau harga pasti seorang pemain."
    )

    st.markdown(
        "#### 🧠 Mengapa model "
        "memberi estimasi ini?"
    )

    local_explanation(pred)

    st.caption(
        "Nama pemain digunakan untuk "
        "mencari skenario historis. "
        "Nama dan performa individualnya "
        "bukan fitur CatBoost."
    )

    st.markdown(
        "#### ⚖️ Prediksi vs biaya tercatat"
    )

    st.write(
        "File `player_scenarios.csv` saat ini "
        "belum memuat biaya transfer aktual. "
        "Karena itu, aplikasi tidak membuat "
        "angka pembanding yang tidak tersedia. "
        "Jika data aktual ditambahkan nanti, "
        "hasil dari transaksi training harus "
        "diberi label **in-sample**."
    )


# ==========================================
# 09 - SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title(
    "⚽ Football Transfer Intelligence"
)

page = st.sidebar.radio(
    "Navigasi",
    [
        "🏠 Beranda",
        "🔎 Cari Pemain",
        "⚽ Simulasi Transfer",
        "📊 Market & Model",
        "🧠 Explainable AI"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Dataset historis 2010–2023. "
    "Model final CatBoost tidak dilatih "
    "ulang di aplikasi."
)

cat = results.loc[
    results["Model"] == "CatBoost"
]

base = results.loc[
    results["Model"] == "Median Baseline"
]

if cat.empty or base.empty:
    st.error(
        "Hasil CatBoost / Median Baseline "
        "tidak ditemukan."
    )
    st.stop()

mae = float(cat.iloc[0]["MAE"])
base_mae = float(base.iloc[0]["MAE"])


# ==========================================
# 10 - HOME PAGE
# ==========================================

if page == "🏠 Beranda":

    st.title(
        "⚽ Football Transfer Intelligence 2.1"
    )

    st.write(
        "Jelajahi profil pemain, simulasi "
        "transfer historis, dan penjelasan "
        "prediksi Machine Learning."
    )

    a, b, c = st.columns(3)

    a.metric(
        "Final test MAE",
        f"{mae:,.0f}"
    )

    b.metric(
        "Final test R²",
        f"{float(cat.iloc[0]['R2']):.4f}"
    )

    c.metric(
        "MAE turun vs baseline",
        f"{(1 - mae/base_mae)*100:.2f}%"
    )

    st.markdown("### Mulai dari mana?")

    st.write(
        "**🔎 Cari Pemain** — Cari nama, "
        "lihat foto dan riwayat transfer, "
        "jalankan prediksi, dan pahami SHAP."
    )

    st.write(
        "**⚽ Simulasi Transfer** — "
        "Pilih klub asal, klub tujuan, "
        "tahun dan usia."
    )

    st.write(
        "**📊 Market & Model** — "
        "Eksplorasi grafik serta "
        "perbandingan model."
    )

    st.write(
        "**🧠 Explainable AI** — "
        "SHAP global dan penjelasan "
        "prediksi terakhir."
    )

    st.warning(
        "Model hanya memakai empat fitur: "
        "tahun, usia, ID klub asal, "
        "dan ID klub tujuan. Foto dan "
        "nama pemain bukan input CatBoost."
    )


# ==========================================
# 11 - PLAYER PROFILE & TRANSFER JOURNEY
# ==========================================

elif page == "🔎 Cari Pemain":

    st.title(
        "🔎 Football Player Profile "
        "& Transfer Journey"
    )

    st.caption(
        "Cari pemain dalam transaksi historis "
        "2010–2021. Foto mungkin berasal "
        "dari musim lain."
    )

    players = (
        scenarios.groupby(
            "player_id",
            as_index=False
        )
        .agg(
            player_name=("player_name", "first"),
            records=("player_id", "size")
        )
        .sort_values(
            ["records", "player_name"],
            ascending=[False, True]
        )
    )

    search = st.text_input(
        "Cari nama pemain",
        placeholder=(
            "Misalnya Haaland, "
            "Berbatov, Ronaldo..."
        )
    )

    if search.strip():

        found = players.loc[
            players["player_name"].str.contains(
                search.strip(),
                case=False,
                regex=False,
                na=False
            )
        ].head(100)

    else:

        found = players.head(100)

        st.caption(
            "Menampilkan 100 pemain teratas. "
            "Ketik nama untuk mencari lainnya."
        )

    if found.empty:

        st.warning(
            "Pemain tidak ditemukan dalam "
            "katalog skenario development."
        )

    else:

        labels = found.set_index("player_id")

        player_id = st.selectbox(
            "Pilih pemain",
            found["player_id"].astype(str).tolist(),
            format_func=lambda pid: (
                f"{labels.loc[pid, 'player_name']} "
                f"· {int(labels.loc[pid, 'records'])} "
                "catatan"
            )
        )

        history = (
            scenarios.loc[
                scenarios["player_id"].astype(str)
                == player_id
            ]
            .sort_values("year", ascending=False)
            .reset_index(drop=True)
        )

        player_name = str(
            history.loc[0, "player_name"]
        )

        photo_col, info_col = st.columns(
            [1, 3],
            gap="large"
        )

        with photo_col:

            show_player_photo(
                player_id,
                player_name
            )

        with info_col:

            st.header(player_name)

            st.markdown(
                "**Riwayat transfer historis "
                "yang tercatat pada dataset:**"
            )

            history_table = history[
                [
                    "year",
                    "origin_name",
                    "destination_name",
                    "age_approx"
                ]
            ].rename(
                columns={
                    "year": "Tahun",
                    "origin_name": "Klub asal",
                    "destination_name": "Klub tujuan",
                    "age_approx": "Usia"
                }
            )

            st.dataframe(
                history_table,
                hide_index=True,
                use_container_width=True
            )

            st.caption(
                "Riwayat mengikuti catatan dataset, "
                "bukan daftar lengkap karier pemain."
            )

        choice = st.selectbox(
            "Pilih transaksi untuk diprediksi",
            list(range(len(history))),
            format_func=lambda i: (
                f"{int(history.loc[i, 'year'])}: "
                f"{history.loc[i, 'origin_name']} "
                f"→ {history.loc[i, 'destination_name']} "
                f"(usia {history.loc[i, 'age_approx']:.0f})"
            )
        )

        chosen = history.loc[choice]

        key = scenario_key(
            chosen,
            player_id
        )

        if st.button(
            "⚡ Prediksi biaya transfer "
            "skenario ini",
            type="primary"
        ):

            label = (
                f"{player_name} · "
                f"{int(chosen['year'])}: "
                f"{chosen['origin_name']} "
                f"→ {chosen['destination_name']}"
            )

            run_prediction(
                chosen,
                label,
                player_id
            )

        show_prediction(key)


# ==========================================
# 12 - CUSTOM TRANSFER SIMULATOR
# ==========================================

elif page == "⚽ Simulasi Transfer":

    st.title(
        "⚽ Simulasi Transfer Sendiri"
    )

    st.write(
        "Pilih klub berdasarkan nama. "
        "ID hanya digunakan di balik layar "
        "oleh model CatBoost."
    )

    with st.form("custom"):

        a, b = st.columns(2)

        with a:

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

        with b:

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
                15,
                45,
                24
            )

        submit = st.form_submit_button(
            "⚡ Estimate Transfer Fee",
            type="primary",
            use_container_width=True
        )

    candidate = {
        "year": year,
        "age_approx": age,
        "left_club_id": origin,
        "joined_club_id": destination
    }

    if submit:

        if origin == destination:

            st.warning(
                "Klub asal dan tujuan "
                "harus berbeda."
            )

        else:

            label = (
                f"{club_label(origin)} "
                f"→ {club_label(destination)} "
                f"· {year} · usia {age}"
            )

            run_prediction(
                candidate,
                label
            )

    show_prediction(
        scenario_key(candidate)
    )


# ==========================================
# 13 - MARKET & MODEL LABORATORY
# ==========================================

elif page == "📊 Market & Model":

    st.title(
        "📊 Market Explorer "
        "& Model Laboratory"
    )

    coverage["year"] = pd.to_numeric(
        coverage["year"],
        errors="coerce"
    )

    coverage = (
        coverage.dropna(subset=["year"])
        .sort_values("year")
    )

    period = st.slider(
        "Pilih rentang tahun",
        int(coverage.year.min()),
        int(coverage.year.max()),
        (
            int(coverage.year.min()),
            int(coverage.year.max())
        )
    )

    metric_map = {
        "Persentase fee positif":
            "positive_fee_pct",
        "Jumlah fee positif":
            "positive_fee_records",
        "Semua catatan non-loan":
            "total_records"
    }

    market_choice = st.selectbox(
        "Metrik pasar",
        list(metric_map)
    )

    filtered = coverage[
        coverage["year"].between(*period)
    ]

    fig = px.line(
        filtered,
        x="year",
        y=metric_map[market_choice],
        markers=True
    )

    fig.update_traces(
        line_color=TEAL,
        line_width=3,
        marker_size=8
    )

    fig.update_xaxes(dtick=1)

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    with st.expander(
        "📖 Cara membaca grafik coverage"
    ):

        st.write(
            "Grafik menunjukkan catatan "
            "transfer non-loan dan bagian "
            "yang memiliki fee positif. "
            "Fee nol atau tidak tercatat "
            "bukan otomatis transfer gratis."
        )

    st.subheader(
        "Perbandingan model "
        "final test 2022–2023"
    )

    metric = st.selectbox(
        "Pilih metrik model",
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
        color="Model",
        text=metric
    )

    fig2.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        style_chart(fig2),
        use_container_width=True
    )

    with st.expander(
        "📖 Penjelasan MAE, RMSE, R²"
    ):

        st.write(
            "MAE: rata-rata kesalahan absolut, "
            "lebih kecil lebih baik."
        )

        st.write(
            "RMSE: lebih sensitif terhadap "
            "kesalahan prediksi yang besar."
        )

        st.write(
            "R²: ukuran kemampuan model "
            "menjelaskan variasi target, "
            "bukan persentase akurasi."
        )

    prediction_file = (
        ROOT / "final_predictions.csv"
    )

    if prediction_file.exists():

        pred = pd.read_csv(prediction_file)

        required_pred = {
            "year",
            "actual_fee",
            "predicted_fee"
        }

        if required_pred.issubset(
            pred.columns
        ):

            data = pred[
                (pred.actual_fee > 0)
                &
                (pred.predicted_fee > 0)
            ].copy()

            if not data.empty:

                data["year"] = (
                    data["year"].astype(str)
                )

                scatter = px.scatter(
                    data,
                    x="actual_fee",
                    y="predicted_fee",
                    color="year",
                    log_x=True,
                    log_y=True,
                    opacity=0.5,
                    render_mode="webgl"
                )

                lo = min(
                    data.actual_fee.min(),
                    data.predicted_fee.min()
                )

                hi = max(
                    data.actual_fee.max(),
                    data.predicted_fee.max()
                )

                scatter.add_trace(
                    go.Scatter(
                        x=[lo, hi],
                        y=[lo, hi],
                        mode="lines",
                        name="Perfect prediction",
                        line=dict(
                            color=ORANGE,
                            dash="dash"
                        )
                    )
                )

                st.subheader(
                    "Final test: Actual vs Predicted"
                )

                st.plotly_chart(
                    style_chart(scatter, 540),
                    use_container_width=True
                )

                st.caption(
                    "Semakin dekat garis diagonal, "
                    "semakin tepat. Kedua sumbu "
                    "menggunakan skala logaritmik."
                )


# ==========================================
# 14 - EXPLAINABLE AI PAGE
# ==========================================

else:

    st.title(
        "🧠 Explainable AI"
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
        style_chart(fig),
        use_container_width=True
    )

    st.info(
        "Global SHAP menunjukkan besarnya "
        "kontribusi fitur pada skala log-target. "
        "Bukan hubungan sebab-akibat."
    )

    pred = st.session_state.get(
        "last_prediction"
    )

    if pred:

        st.subheader(
            "SHAP untuk prediksi terakhir"
        )

        st.caption(
            pred["title"]
        )

        local_explanation(pred)

    else:

        st.caption(
            "Jalankan prediksi dari halaman "
            "Cari Pemain atau Simulasi Transfer "
            "terlebih dahulu."
        )


# ==========================================
# 15 - FOOTER
# ==========================================

st.divider()

st.caption(
    "Research demo · Data transfer historis "
    "positif non-loan · Foto mungkin berbeda "
    "periode · Bukan harga pemain real-time"
)
