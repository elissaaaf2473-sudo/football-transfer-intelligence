"""Streamlit explorer and holdout gap analysis for the original CatBoost V3."""

from base64 import b64encode
from html import escape
import json
from pathlib import Path
import unicodedata

from catboost import CatBoostError, CatBoostRegressor
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "transfer_fee_lab_assets"
MODEL_PATH = ROOT / "model_v3.cbm"
KEYS = ["player_id", "transfer_date", "from_club_id", "to_club_id"]
CLUB_FEATURES = ["from_club_id", "to_club_id"]
KEY_DTYPES = dict.fromkeys(KEYS, "string")
SPLITS = {"Train": "Training", "Validation": "Validasi", "Test": "Holdout"}
SYMBOLS = {"EUR": "€", "USD": "US$", "IDR": "Rp "}
EXPLORER = "Transfer Explorer"
GAPS = "Gap Analysis"
FX_URL = "https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html"
STADIUM_SOURCE = "https://commons.wikimedia.org/wiki/File:Inside_the_Emirates_Stadium_-_Panorama.jpg"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def football_theme():
    css = (ASSETS / "football_theme.css").read_text(encoding="utf-8")
    stadium = b64encode((ASSETS / "stadium.webp").read_bytes()).decode("ascii")
    return css, stadium


def render_cover(css, stadium):
    st.html(f"<style>{css}</style>")
    st.markdown(
        f"""
<section class="v3-hero" aria-label="Football Transfer Lab V3"
         style="--v3-stadium: url('data:image/webp;base64,{stadium}')">
    <span class="v3-hero-frame" aria-hidden="true"></span>
    <div>
        <div class="v3-hero-kicker">Football Transfer Intelligence</div>
        <div class="v3-hero-title">FOOTBALL<strong>TRANSFER LAB</strong></div>
        <div class="v3-edition">Edisi V3 &nbsp;·&nbsp; CatBoost &nbsp;·&nbsp; Transfer historis</div>
    </div>
</section>
""",
        unsafe_allow_html=True,
    )


def feature_frame(rows, features):
    frame = rows[features].copy()
    for column in features:
        if column in CLUB_FEATURES:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64").astype(str)
        else:
            frame[column] = pd.to_numeric(
                frame[column].replace({"True": 1, "False": 0, "true": 1, "false": 0}),
                errors="coerce",
            ).astype(float)
    return frame


@st.cache_resource
def load_model():
    return CatBoostRegressor().load_model(str(MODEL_PATH))


@st.cache_data(show_spinner="Menyiapkan skenario dan prediksi V3…")
def load_catalog():
    metadata = read_json(ROOT / "model_metadata_v3.json")
    rows = pd.read_csv(ROOT / "player_scenarios_v3.csv", dtype=KEY_DTYPES, low_memory=False)
    fees = pd.read_csv(ASSETS / "recorded_fees.csv.gz", dtype=KEY_DTYPES, low_memory=False)
    rows = rows.merge(fees, on=KEYS, how="left", validate="one_to_one")
    club_catalog = pd.read_csv(ROOT / "club_catalog_named.csv", dtype={"club_id": "string"})
    club_names = dict(
        club_catalog.dropna(subset=["club_id", "club_name"])[["club_id", "club_name"]]
        .itertuples(index=False, name=None)
    )
    for side in ("from", "to"):
        club_id, name = f"{side}_club_id", f"{side}_club_name"
        for code, label in rows[[club_id, name]].dropna().itertuples(index=False, name=None):
            club_names.setdefault(code, label)
        rows[name] = rows[club_id].map(club_names).fillna(rows[name]).fillna("Klub tidak diketahui")
    rows["player_name"] = rows["player_name"].fillna("Pemain tidak diketahui")
    rows["route"] = rows["from_club_name"] + " → " + rows["to_club_name"]
    model = load_model()
    if list(model.feature_names_) != metadata["features"]:
        raise ValueError("Urutan fitur model berbeda dari metadata V3.")
    rows["estimate"] = np.maximum(
        0, model.predict(feature_frame(rows, model.feature_names_), thread_count=2)
    )
    rows["gap"] = rows["estimate"] - rows["transfer_fee"]
    rows["gap_pct"] = 100 * rows["gap"] / rows["transfer_fee"].where(rows["transfer_fee"].gt(0))
    return rows, club_names, metadata


def normalize(value):
    return "".join(
        char for char in unicodedata.normalize("NFKD", str(value).casefold())
        if not unicodedata.combining(char)
    ).strip()


def format_money(value, currency, fx, signed=False):
    if pd.isna(value):
        return "—"
    prefix = "−" if value < 0 else "+" if signed and value > 0 else ""
    return f"{prefix}{SYMBOLS[currency]}{abs(value) * fx[currency]:,.0f}"


def format_pct(value):
    return "—" if pd.isna(value) else f"{value:+.1f}%"


def transfer_label(row):
    return f"{row.transfer_date} · {row.route}"


def gap_table(frame, money, include_player=False):
    display = pd.DataFrame({
        "Tanggal": frame["transfer_date"],
        "Transfer": frame["route"],
        "Split": frame["source_split"].map(SPLITS),
        "Prediksi": frame["estimate"].map(money),
        "Tercatat": frame["transfer_fee"].map(money),
        "Selisih": frame["gap"].map(lambda v: money(v, signed=True)),
        "Selisih (%)": frame["gap_pct"].map(format_pct),
    })
    if include_player:
        display.insert(0, "Pemain", frame["player_name"])
    return display


def render_photo(row):
    file = ASSETS / "portraits" / f"{row.player_id}.webp"
    info = read_json(ASSETS / "portraits" / "credits.json").get(row.player_id)
    if file.is_file() and info:
        st.image(str(file), width=135)
        st.caption(f"[{info['credit']}]({info['source']}) · [{info['license']}]({info['licenseUrl']}) · diperkecil")
    else:
        initials = "".join(word[0] for word in row.player_name.split()[:2])
        st.html(f'<span class="v3-player-initials" aria-hidden="true">{escape(initials)}</span>')
        st.caption("Foto belum tersedia.")


def render_comparison(row, money, currency, fx):
    st.subheader("Perbandingan biaya transfer")
    prediction, actual, difference = st.columns(3)
    prediction.metric("Prediksi CatBoost V3", money(row.estimate))
    actual.metric("Biaya transfer tercatat", money(row.transfer_fee))
    difference.metric("Selisih prediksi", money(row.gap, signed=True))
    if pd.isna(row.transfer_fee):
        st.caption("Biaya transfer tercatat belum tersedia untuk skenario ini.")
        return
    direction = "Estimasi terlalu tinggi" if row.gap > 0 else "Estimasi terlalu rendah" if row.gap < 0 else "Estimasi sama dengan biaya tercatat"
    percent = f" ({format_pct(row.gap_pct)} dari biaya tercatat)" if pd.notna(row.gap_pct) else ""
    st.caption(f"{direction}{percent}. Selisih dihitung dari prediksi dikurangi biaya tercatat.")
    figure = go.Figure()
    for label, value, color in [
        ("Prediksi", row.estimate, "#c7ef43"),
        ("Tercatat", row.transfer_fee, "#dce4d4"),
    ]:
        figure.add_bar(
            x=[value * fx[currency]], y=[label], orientation="h", marker_color=color,
            customdata=[money(value)], hovertemplate="%{customdata}<extra></extra>",
        )
    figure.update_layout(
        height=165,
        margin=dict(l=0, r=10, t=5, b=25),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5ebdf"),
    )
    figure.update_xaxes(rangemode="tozero", ticksuffix=f" {currency}", gridcolor="#30382c")
    figure.update_yaxes(autorange="reversed")
    st.plotly_chart(figure, theme=None, config={"displayModeBar": False})


def render_explorer(rows, club_names, currency, fx, money):
    search = st.sidebar.text_input("Cari nama atau ID pemain", key="v3_query")
    players = rows[["player_id", "player_name"]].drop_duplicates("player_id")
    term = normalize(search)
    matches = players.loc[
        players["player_name"].map(normalize).str.contains(term, regex=False)
        | players["player_id"].str.startswith(term, na=False)
    ].copy()
    if matches.empty:
        st.info("Pemain tidak ditemukan. Coba nama atau ID lain.")
        return
    preferred = st.session_state.get("v3_player_id", "418560")
    matches["priority"] = (
        matches["player_id"].eq(preferred) | matches["player_name"].map(normalize).eq(term)
    ).fillna(False)
    matches = matches.sort_values(["priority", "player_name"], ascending=[False, True])
    labels = dict(zip(matches.player_id, matches.player_name))
    player = st.sidebar.selectbox(
        "Pemain", options=list(labels), format_func=lambda code: f"{labels[code]} · {code}",
        key="v3_player_widget",
    )
    st.session_state["v3_player_id"] = player
    history = rows.loc[rows.player_id.eq(player)].sort_values("transfer_date", ascending=False)
    indices = history.index.tolist()
    preferred_case = st.session_state.get("v3_case_id")
    index = st.sidebar.selectbox(
        "Skenario historis", options=indices,
        index=indices.index(preferred_case) if preferred_case in indices else 0,
        format_func=lambda i: transfer_label(history.loc[i]), key=f"v3_case_{player}",
    )
    st.session_state["v3_case_id"] = index
    row = history.loc[index]
    with st.container(key="v3_player_card", border=True):
        summary, portrait = st.columns([4, 1], vertical_alignment="center")
        with summary:
            st.html('<div class="v3-section-label">PROFIL TRANSFER</div>')
            st.header(row.player_name)
            st.markdown(f"**{row.route}**")
            st.caption(f"{row.transfer_date} · {SPLITS.get(row.source_split, row.source_split)}")
        with portrait:
            render_photo(row)
    with st.container(key="v3_comparison_card", border=True):
        render_comparison(row, money, currency, fx)
    with st.expander("Data sebelum transfer"):
        age, market, minutes, production = st.columns(4)
        age.metric("Usia", "—" if pd.isna(row.age_at_transfer) else f"{row.age_at_transfer:.1f} tahun")
        market.metric("Market value", money(row.mv_pretransfer_strict_eur))
        minutes.metric("Menit bermain · 365 hari", "—" if pd.isna(row.minutes_365d) else f"{row.minutes_365d:,.0f}")
        goals_assists = row.goals_365d + row.assists_365d
        production.metric("Gol + assist · 365 hari", "—" if pd.isna(goals_assists) else f"{goals_assists:,.0f}")
    with st.expander("Simulasi klub tujuan"):
        st.caption("Simulasi memakai tanggal dan kondisi pemain dari skenario historis yang dipilih.")
        options = sorted(
            (code for code in club_names if code not in {row.from_club_id, row.to_club_id}),
            key=lambda code: club_names[code].casefold(),
        )
        club = st.selectbox(
            "Klub tujuan alternatif", options, index=None,
            format_func=lambda code: club_names[code], key=f"v3_club_{row.name}",
        )
        if club is not None:
            alternative = history.loc[[index]].copy()
            alternative["to_club_id"] = club
            model = load_model()
            estimate = max(0.0, model.predict(feature_frame(alternative, model.feature_names_), thread_count=2)[0])
            st.metric("Estimasi untuk tujuan alternatif", money(estimate))
            st.caption(f"Selisih dari prediksi historis {money(estimate - row.estimate, signed=True)}.")
    st.subheader("Riwayat transfer pemain")
    st.dataframe(gap_table(history, money), hide_index=True)


def open_in_explorer(row):
    st.session_state["v3_view"] = EXPLORER
    st.session_state["v3_query"] = ""
    st.session_state["v3_player_widget"] = row.player_id
    st.session_state["v3_player_id"] = row.player_id
    st.session_state["v3_case_id"] = row.name
    st.session_state[f"v3_case_{row.player_id}"] = row.name


def render_gaps(rows, metadata, money):
    metrics = metadata["final_test_metrics"]
    a, b, c, d = st.columns(4)
    a.metric("Transfer holdout", f"{metadata['test_records']:,}")
    b.metric("MAE holdout", money(metrics["MAE_EUR"]))
    c.metric("R² holdout", f"{metrics['R2']:.4f}")
    d.metric("WAPE holdout", f"{metrics['WAPE_pct']:.2f}%")
    st.caption("Metrik evaluasi V3 untuk periode 2024–2025.")
    a, b, c = st.columns(3)
    split = a.selectbox("Split data", ["Test", "Validation", "Train", "All"], format_func=lambda s: SPLITS.get(s, "Semua"))
    direction = b.selectbox("Arah selisih", ["Semua", "Terlalu tinggi", "Terlalu rendah"])
    sort_by = c.selectbox("Urutkan berdasarkan", ["Selisih nominal terbesar", "Selisih persentase terbesar", "Biaya tercatat terbesar"])
    query = normalize(st.text_input("Filter nama pemain atau klub"))
    filtered = rows.loc[rows.transfer_fee.notna()]
    if split != "All":
        filtered = filtered.loc[filtered.source_split.eq(split)]
    if direction == "Terlalu tinggi":
        filtered = filtered.loc[filtered.gap.gt(0)]
    elif direction == "Terlalu rendah":
        filtered = filtered.loc[filtered.gap.lt(0)]
    if query:
        haystack = (filtered.player_name + " " + filtered.route).map(normalize)
        filtered = filtered.loc[haystack.str.contains(query, regex=False)]
    order = {
        "Selisih nominal terbesar": filtered.gap.abs(),
        "Selisih persentase terbesar": filtered.gap_pct.abs(),
        "Biaya tercatat terbesar": filtered.transfer_fee,
    }[sort_by]
    filtered = filtered.loc[order.sort_values(ascending=False, kind="stable").index]
    st.caption(f"{len(filtered):,} transfer cocok dengan filter.")
    if filtered.empty:
        st.info("Tidak ada transfer yang cocok dengan filter.")
        return
    event = st.dataframe(
        gap_table(filtered, money, include_player=True), hide_index=True,
        on_select="rerun", selection_mode="single-row",
        key=f"v3_gap_table_{split}_{direction}_{sort_by}_{query}",
    )
    if event.selection.rows:
        row = filtered.iloc[event.selection.rows[0]]
        st.button("Buka di Transfer Explorer", on_click=open_in_explorer, args=(row,))


def main():
    st.set_page_config(page_title="Transfer Fee Lab V3", page_icon="⚽", layout="wide")
    try:
        rows, club_names, metadata = load_catalog()
        sources = read_json(ASSETS / "sources.json")
        css, stadium = football_theme()
    except (CatBoostError, OSError, ValueError, KeyError) as error:
        st.error(f"File V3 belum dapat dimuat. Periksa folder proyek. Detail: {error}")
        st.stop()
    view = st.sidebar.radio("Menu Transfer Fee Lab", [EXPLORER, GAPS], key="v3_view")
    currency = st.sidebar.selectbox("Mata uang tampilan", ["EUR", "USD", "IDR"], key="v3_currency")
    fx = sources["fx"]

    def money(value, signed=False):
        return format_money(value, currency, fx, signed)

    render_cover(css, stadium)
    section = "01 / EKSPLORASI" if view == EXPLORER else "02 / EVALUASI MODEL"
    st.html(f'<div class="v3-section-label">{section}</div>')
    st.title(view)
    if view == EXPLORER:
        st.caption(f"{rows.player_id.nunique():,} pemain · {len(rows):,} skenario historis")
        render_explorer(rows, club_names, currency, fx, money)
    else:
        render_gaps(rows, metadata, money)
    st.divider()
    fx_info = "Mata uang dasar model EUR."
    if currency != "EUR":
        fx_info = f"Kurs tampilan €1 = {fx[currency]:,.4f} {currency} · [ECB {fx['date']}]({FX_URL})."
    st.caption(f"{fx_info} [Sumber biaya transfer tercatat]({sources['fees']}).")
    st.caption(
        f"Foto stadion [Lloyd Morgan]({STADIUM_SOURCE}) · "
        "[CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/) · dipotong dan diperkecil."
    )


if __name__ == "__main__":
    main()
