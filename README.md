# Ekstraktor Teks Image 

Aplikasi Python berbasis OCR (EasyOCR & OpenCV) untuk mengekstraksi **Kolom NIK (16-Digit)** dari gambar tabel cetak/dokumen secara presisi dan mengonversinya langsung ke format **CSV**.

---

## 🌟 Fitur Utama

- **Offline & Pemrosesan Lokal**: Menggunakan EasyOCR dan OpenCV tanpa perlu koneksi internet, API Key, atau layanan cloud berbayar.
- **Auto-Cropping Kolom NIK Presisi**: Memotong khusus area Kolom NIK (Kolom ke-3) & menghilangkan noise/tanda tangan/coretan di kolom lain.
- **Pembersihan Noise Warna (HSV Masking)**: Menghilangkan bekas arsir stabilo merah/pink/oranye secara otomatis agar angka di bawahnya terbaca bersih.
- **Pre-processing Citra Optimal (Binarisasi Otsu & Upscaling)**: Memperbesar resolusi 2x lipat & mengatur kontras otomatis untuk menangkap cetakan angka kecil.
- **Kamus Koreksi Karakter Otomatis**: Memperbaiki otomatis kesalahan pembacaan OCR pada printer (misal `B` -> `3`, `O` -> `0`, `S` -> `5`, `I/l` -> `1`, serta memastikan prefiks wilayah).
- **Pengurutan Baris & Pemformatan CSV**: Menghasilkan data NIK yang diurutkan dari atas ke bawah serta menjaga integritas format NIK (awalan tanda petik `'` agar digit `0` dan 16 digit tidak terpotong oleh Microsoft Excel).
- **Batch Processing**: Mendukung pemrosesan 1 gambar spesifik maupun seluruh gambar dalam satu direktori/folder secara otomatis.

---

## 🏗 Struktur Proyek

```text
Image To Csv/
├── input/             # Folder tempat meletakkan gambar input (.jpg, .png, .jpeg, dll)
├── output/            # Folder tempat menyimpan hasil file CSV ekstraksi
├── main.py            # CLI entrypoint & logika utama ekstraksi NIK presisi
├── cleaning.ipynb     # Jupyter Notebook untuk eksplorasi & pembersihan data
├── requirements.txt   # Daftar pustaka & dependensi Python
└── README.md          # Dokumentasi proyek (Bahasa Indonesia)
```

---

## 💻 Panduan Instalasi & Penggunaan

### 1. Prasyarat System & Python
Pastikan Anda sudah menginstal Python (versi 3.8 - 3.11 direkomendasikan).

### 2. Membuat Virtual Environment (Opsional tetapi Disarankan)

**Windows**:
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalasi Dependensi
Jalankan perintah berikut untuk menginstal seluruh dependensi yang dibutuhkan:

```bash
pip install -r requirements.txt
```

---

## 🚀 Cara Penggunaan (CLI)

### 1. Ekstraksi NIK dari Beberapa Gambar Sekaligus (Batch Mode)
Letakkan semua gambar tabel Anda ke dalam folder `input/`, lalu jalankan:

```bash
python main.py -i input -o output
```
Setiap gambar akan diproses dan otomatis menghasilkan file CSV terpisah untuk masing-masing gambar di folder `output/`:
- `input/halaman1.jpg` ➔ `output/halaman1_nik.csv`
- `input/halaman2.jpg` ➔ `output/halaman2_nik.csv`

Jika gambar berada di sub-folder dalam `input/`, tambahkan opsi `-r` (recursive):
```bash
python main.py -i input -o output -r
```

### 2. Ekstraksi NIK dari 1 Gambar Spesifik
```bash
python main.py -i input/sampel.jpg -o output/hasil_nik.csv
```

---

## ⚙️ Opsi Parameter Command Line (CLI)

| Parameter | Singkatan | Tipe | Wajib | Deskripsi |
|-----------|-----------|------|-------|-----------|
| `--input` | `-i` | String | **Ya** | Jalur (path) ke file gambar tunggal atau folder berisi gambar. |
| `--output` | `-o` | String | **Ya** | Jalur (path) ke file CSV hasil atau folder tujuan output. |
| `--recursive` | `-r` | Flag | Tidak | Mencari file gambar di dalam sub-folder jika `--input` berupa folder. |

---

## 📊 Contoh Input & Hasil Output CSV

### Contoh Hasil Tabel di CSV (`output/halaman1_nik.csv`):
```csv
Baris,NIK,Confidence,Teks_OCR
1,'3302011205900001,98.5%,3302011205900001
2,'3302012508850002,96.2%,3302012508850002
3,'3302014101920003,99.1%,3302014101920003
```

> **Catatan Pemformatan NIK**: Format NIK diawali tanda petik tunggal (`'33...`) agar ketika file CSV dibuka di **Microsoft Excel**, angka NIK 16 digit tidak berubah menjadi notasi ilmiah (seperti `3.30201E+15`) dan digit angka nol di depan tidak hilang.

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan pengolahan dokumen dan tabel NIK secara efisien dan open-source di bawah lisensi [MIT License](LICENSE).
