# Transfer Fee Lab V3 · Streamlit

Aplikasi Streamlit ini memakai file model CatBoost V3 yang sudah dilatih
di Colab. Prediksi katalog dihitung dari model saat aplikasi dibuka dan
disimpan di cache. Simulasi klub tujuan menghitung ulang prediksi dengan
mengubah ID klub tujuan pada skenario yang dipilih.

## Jalankan dari ZIP

Ekstrak ZIP. Buka terminal dari folder yang berisi `requirements.txt`.

```sh
python -m pip install -r requirements.txt
python -m streamlit run pages/2_Transfer_Fee_Lab.py
```

Di Windows, pakai `py` sebagai pengganti `python` jika diperlukan.

## Tambahkan ke aplikasi Streamlit lama

Salin dua item berikut ke repo Football Transfer Intelligence yang sudah
memuat model, metadata, katalog pemain, katalog klub, dan dependensi V3.

- `pages/2_Transfer_Fee_Lab.py` ke folder `pages/`.
- Folder `transfer_fee_lab_assets/` ke root repo.

Gunakan `app.py` sebagai entrypoint aplikasi multipage yang lama. Halaman
**Transfer Fee Lab** akan muncul di sidebar. Untuk memasang versi ini sebagai
aplikasi terpisah, gunakan isi ZIP sebagai repo dan tetapkan
`pages/2_Transfer_Fee_Lab.py` sebagai main file.

File `.streamlit/config.toml` menyesuaikan tema dasar widget dan tabel. Jika
memakai repo lama dengan pengaturan tema sendiri, tambahkan bagian `[theme]`
hanya jika ingin memakai palet yang sama pada seluruh aplikasi Streamlit.

Untuk akses terbatas di Streamlit Community Cloud, gunakan repo GitHub private
dan cek **Settings → Sharing**. Jika menambahkan halaman ke aplikasi yang
sudah terhubung ke GitHub, cek akses aplikasi itu sebelum mendorong perubahan.
Lihat [panduan akses Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app).

## Data

- Model, metadata, dan katalog V3 berasal dari
  [repo Football Transfer Intelligence](https://github.com/elissaaaf2473-sudo/football-transfer-intelligence)
  pada commit `39c2304676442b00a54bad01f4cd86baff6423a4`.
- Biaya transfer tercatat berasal dari
  [Transfermarkt Datasets](https://github.com/dcaribou/transfermarkt-datasets)
  snapshot 6 Juli 2026. Pencocokan memakai ID pemain, tanggal, ID klub asal,
  dan ID klub tujuan. Berkas yang disertakan hanya memuat 15.824 biaya yang
  cocok dengan katalog V3.
- USD dan IDR memakai kurs referensi ECB 9 November 2026 untuk tampilan.
- Kredit dan lisensi lima foto Wikimedia Commons ada di
  `transfer_fee_lab_assets/portraits/credits.json`.
- `transfer_fee_lab_assets/stadium.webp` adalah versi yang dipotong dan
  diperkecil dari [Inside the Emirates Stadium – Panorama](https://commons.wikimedia.org/wiki/File:Inside_the_Emirates_Stadium_-_Panorama.jpg)
  oleh Lloyd Morgan, dengan lisensi [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/).
