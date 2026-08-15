import sys
import os
import re
import argparse
import logging
from pathlib import Path
from typing import List
import cv2
import numpy as np
import pandas as pd
import easyocr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("nik-extractor")

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def clean_and_correct_nik(raw_text: str) -> str:
    """Membersihkan teks dan memprioritaskan/memastikan digit NIK diawali dengan '33' (Jawa Tengah)."""
    char_map = {
        'B': '3', 'b': '6', 'O': '0', 'o': '0', 'Q': '0',
        'S': '5', 's': '5', 'Z': '2', 'z': '2', 'I': '1',
        'i': '1', 'l': '1', '|': '1', 'g': '9', 'q': '9',
        'A': '4', 'T': '7'
    }
    
    cleaned = ""
    for char in raw_text:
        if char in char_map:
            cleaned += char_map[char]
        elif char.isdigit():
            cleaned += char
            
    # 1. Prioritas Utama: Cari 16-digit angka yang berawalan '33'
    matches_33 = re.findall(r'33\d{14}', cleaned)
    if matches_33:
        return matches_33[0]

    # 2. Jika angka 3 pertama salah terbaca (misal misread '33' -> '83' atau '53'), koreksi menjadi '33' jika panjangnya 16 digit
    all_16 = re.findall(r'\d{16}', cleaned)
    if all_16:
        nik = all_16[0]
        # Jika diawali 3x misal 3775, 3875 -> pastikan 33
        if nik.startswith('3') and not nik.startswith('33'):
            return '33' + nik[2:]
        elif not nik.startswith('33'):
            return '33' + nik[2:]
        return nik

    # 3. Cari pola string yang diawali 33
    match_prov = re.search(r'33\d{14}', cleaned)
    if match_prov:
        return match_prov.group(0)
        
    return ""


def get_nik_column_crop(img: np.ndarray) -> np.ndarray:
    """Mencari letak Kolom NIK (Kolom 3) secara presisi dengan deteksi garis vertikal."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape[:2]

    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h / 30)))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_positions = []
    for c in contours:
        x, y, w_c, h_c = cv2.boundingRect(c)
        if h_c > h * 0.35:
            x_positions.append(x)

    x_positions = sorted(list(set(x_positions)))
    
    filtered_x = []
    for x in x_positions:
        if not filtered_x or (x - filtered_x[-1]) > w * 0.025:
            filtered_x.append(x)

    # Kolom NIK di antara Garis ke-2 dan Garis ke-3
    if len(filtered_x) >= 4:
        crop_x1 = max(0, filtered_x[2] + 2)
        crop_x2 = min(w, filtered_x[3] - 2)
        crop_y1 = int(h * 0.04)
        crop_y2 = int(h * 0.98)
        return img[crop_y1:crop_y2, crop_x1:crop_x2]

    # Fallback presisi
    return img[int(h * 0.04):int(h * 0.98), int(w * 0.165):int(w * 0.415)]


def extract_nik_from_image(image_path: str, output_csv: str):
    logger.info(f"Membaca gambar: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Gagal membaca gambar: {image_path}")
        return

    h, w = img.shape[:2]
    
    # 1. Pemotongan Statis Presisi MURNI Kolom NIK (Kolom 3): X 17.0% s/d 41.2%
    # Membuang total Kolom 1, Kolom 2 (Angka ratusan/lokal), Kolom 4, Kolom 5, dan Kolom 6
    crop_x1 = int(w * 0.170)
    crop_x2 = int(w * 0.412)
    crop_y1 = int(h * 0.04)
    crop_y2 = int(h * 0.98)
    
    nik_column_cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # 2. Hilangkan Arsir Stabilo Merah Menggunakan Masking Warna HSV
    hsv = cv2.cvtColor(nik_column_cropped, cv2.COLOR_BGR2HSV)
    
    # Rentang warna merah pada HSV (stabilo merah/pink/oranye)
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Ubah warna stabilo merah menjadi putih bersih
    cleaned_img = nik_column_cropped.copy()
    cleaned_img[red_mask > 0] = [255, 255, 255]
    
    # 3. Pre-processing: Konversi ke Grayscale & Perbesar 2x
    gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)
    h_c, w_c = gray.shape[:2]
    resized = cv2.resize(gray, (w_c * 2, h_c * 2), interpolation=cv2.INTER_CUBIC)
    
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    logger.info("Menjalankan EasyOCR presisi tinggi baris demi baris...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    
    # Single-pass scan terarah untuk menghindari duplikasi/keterbolakan baris
    results = reader.readtext(thresh, allowlist='0123456789BObsSZIz|gqAT', paragraph=False)
    results = sorted(results, key=lambda item: item[0][0][1])
    
    nik_list = []
    seen_niks = set()
    for (bbox, text, prob) in results:
        valid_nik = clean_and_correct_nik(text)
        if valid_nik and len(valid_nik) == 16 and valid_nik not in seen_niks:
            seen_niks.add(valid_nik)
            nik_list.append({
                "Baris": len(nik_list) + 1,
                "NIK": f"'{valid_nik}",
                "Confidence": f"{prob * 100:.1f}%",
                "Teks_OCR": text
            })

    if not nik_list:
        logger.warning(f"Tidak ditemukan NIK 16 digit pada {image_path}")
        return

    # Pastikan folder output ada
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(nik_list)
    df.to_csv(output_csv, index=False)
    print(f"\n[+] SUKSES! Ditemukan tepat {len(df)} NIK dari Kolom NIK.")
    print(f"[+] Hasil disimpan di: {output_csv}\n")
    print(df[["Baris", "NIK", "Confidence"]].to_string(index=False))


def get_image_files(input_path: Path, recursive: bool = False) -> List[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
            return [input_path]
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

    if input_path.is_dir():
        pattern = "**/*" if recursive else "*"
        image_files = [
            p for p in input_path.glob(pattern)
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
        return sorted(image_files)

    return []


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract NIK 16-digit column specifically from images and convert into CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to an image file or a directory containing images."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to the output CSV file."
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search images recursively in subdirectories if --input is a directory."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        image_files = get_image_files(input_path, recursive=args.recursive)
    except Exception as e:
        logger.error(f"Error reading input: {e}")
        sys.exit(1)

    if not image_files:
        logger.error(f"No valid image files found in {input_path}")
        sys.exit(1)

    logger.info(f"Ditemukan {len(image_files)} gambar untuk diproses.")

    for idx, img_file in enumerate(image_files, 1):
        print(f"\n=========================================")
        print(f"  Memproses [{idx}/{len(image_files)}]: {img_file.name}")
        print(f"=========================================")
        
        # Jika input berupa folder/banyak gambar, buat file CSV terpisah sesuai nama gambarnya
        if input_path.is_dir():
            if output_path.suffix == ".csv":
                # Output diisi folder misal: output/
                out_dir = output_path.parent
            else:
                out_dir = output_path
            
            target_csv = out_dir / f"{img_file.stem}_nik.csv"
        else:
            # Jika input berupa 1 file gambar spesifik
            target_csv = output_path if output_path.suffix == ".csv" else output_path / f"{img_file.stem}_nik.csv"

        extract_nik_from_image(str(img_file), str(target_csv))


if __name__ == "__main__":
    main()
