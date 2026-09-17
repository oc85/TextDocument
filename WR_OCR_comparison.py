# ============================================================
# CHEMICAL DOCUMENT YOLO + OCR INSPECTION & MULTI-TABLE PARSER
# ============================================================

import csv
import math
import re
import threading
import time
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import easyocr
import numpy as np
import pymupdf
import torch

from PIL import Image, ImageTk
from ultralytics import YOLO


# ============================================================
# 1. MAIN SETTINGS
# ============================================================

MODEL_PATH = Path(r"D:\Kaggle\WRtools_005\best.pt")

IMAGE_SIZE = 1280

CONFIDENCE = 0.03

IOU_THRESHOLD = 0.45

NON_TABLE_MAX_OVERLAP = 0.05

MAX_DETECTIONS = 2000

PDF_DPI = 300

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. EXPECTED YOLO CLASSES
# ============================================================

EXPECTED_CLASSES = {
    0: "element_symbol",
    1: "element_value",
    2: "unit",
    3: "limit_indicator",
    4: "table",
    5: "headers",
}

OCR_CLASSES = {
    "element_symbol",
    "element_value",
    "unit",
    "limit_indicator",
    "headers",
}

STRUCTURAL_CLASSES = {
    "table",
    "headers",
}

NO_OCR_CLASSES = {
    "table",
}


# ============================================================
# 3. CHEMICAL NAMES TO SYMBOLS MAPPING
# ============================================================

FULL_CHEMICAL_NAMES = {
    "hydrogen": "H", "helium": "He", "lithium": "Li", "beryllium": "Be",
    "boron": "B", "carbon": "C", "nitrogen": "N", "oxygen": "O",
    "fluorine": "F", "neon": "Ne", "sodium": "Na", "magnesium": "Mg",
    "aluminium": "Al", "aluminum": "Al", "silicon": "Si", "phosphorus": "P",
    "sulfur": "S", "sulphur": "S", "chlorine": "Cl", "argon": "Ar",
    "potassium": "K", "calcium": "Ca", "scandium": "Sc", "titanium": "Ti",
    "vanadium": "V", "chromium": "Cr", "manganese": "Mn", "iron": "Fe",
    "cobalt": "Co", "nickel": "Ni", "copper": "Cu", "zinc": "Zn",
    "gallium": "Ga", "germanium": "Ge", "arsenic": "As", "selenium": "Se",
    "bromine": "Br", "krypton": "Kr", "rubidium": "Rb", "strontium": "Sr",
    "yttrium": "Y", "zirconium": "Zr", "niobium": "Nb", "molybdenum": "Mo",
    "technetium": "Tc", "ruthenium": "Ru", "rhodium": "Rh", "palladium": "Pd",
    "silver": "Ag", "cadmium": "Cd", "indium": "In", "tin": "Sn",
    "antimony": "Sb", "tellurium": "Te", "iodine": "I", "xenon": "Xe",
    "cesium": "Cs", "barium": "Ba", "lanthanum": "La", "cerium": "Ce",
    "praseodymium": "Pr", "neodymium": "Nd", "promethium": "Pm", "samarium": "Sm",
    "europium": "Eu", "gadolinium": "Gd", "terbium": "Tb", "dysprosium": "Dy",
    "holmium": "Ho", "erbium": "Er", "thulium": "Tm", "ytterbium": "Yb",
    "lutetium": "Lu", "hafnium": "Hf", "tantalum": "Ta", "tungsten": "W",
    "rhenium": "Re", "osmium": "Os", "iridium": "Ir", "platinum": "Pt",
    "gold": "Au", "mercury": "Hg", "thallium": "Tl", "lead": "Pb",
    "bismuth": "Bi", "polonium": "Po", "astatine": "At", "radon": "Rn",
    "francium": "Fr", "radium": "Ra", "actinium": "Ac", "thorium": "Th",
    "protactinium": "Pa", "uranium": "U", "neptunium": "Np", "plutonium": "Pu",
}


# ------------------------------------------------------------
# ELEMENT SYMBOL MAPPING
# ------------------------------------------------------------

ELEMENT_MAPPING = {
    "al": "Al", "a1": "Al", "ai": "Al", "a|": "Al",
    "b": "B",
    "c": "C",
    "cd": "Cd",
    "cb": "Cb",
    "cr": "Cr", "gr": "Cr", "c r": "Cr",
    "co": "Co", "c0": "Co",
    "cu": "Cu",
    "fe": "Fe", "f e": "Fe", "iron": "Fe",
    "hf": "Hf",
    "mn": "Mn",
    "mo": "Mo", "m0": "Mo",
    "nb": "Nb",
    "ni": "Ni", "nl": "Ni", "n1": "Ni",
    "p": "P",
    "pb": "Pb", "pb.": "Pb",
    "re": "Re",
    "s": "S",
    "si": "Si", "sl": "Si", "s1": "Si", "s|": "Si",
    "sn": "Sn",
    "ta": "Ta",
    "ti": "Ti",
    "tl": "Tl",
    "v": "V",
    "w": "W", "vv": "W",
    "zr": "Zr",
    "ag": "Ag",
    "bi": "Bi",
}


# ------------------------------------------------------------
# UNIT MAPPING
# ------------------------------------------------------------

UNIT_MAPPING = {
    "%": "%", "percent": "%", "percentage": "%", "0/0": "%", "o/o": "%",
    "ppm": "ppm", "prm": "ppm", "ppn": "ppm", "pom": "ppm", "prn": "ppm",
    "ppb": "ppb", "wt%": "wt%", "wt.%": "wt%",
    "t": "t", "(t)": "t", "ton": "t", "tons": "t", "tonne": "t", "tonnes": "t",
    "ktj": "t", "kt": "t", "tj": "t", "(tj)": "t", "[t]": "t",
    "kg": "kg", "(kg)": "kg", "kgs": "kg", "k9": "kg", "kq": "kg", "kc": "kg",
    "mt": "mt", "(mt)": "mt", "m.t": "mt", "m/t": "mt", "metricton": "mt",
    "metrictons": "mt", "metrictonne": "mt", "metrictonnes": "mt",
}


# ------------------------------------------------------------
# LIMIT INDICATOR MAPPING
# ------------------------------------------------------------

LIMIT_MAPPING = {
    "max": "MAX", "maximum": "MAX", "maxima": "MAX", "rnax": "MAX", "rnaximum": "MAX", "m4x": "MAX",
    "min": "MIN", "minimum": "MIN", "minima": "MIN", "rnin": "MIN", "rninimum": "MIN",
}


# ------------------------------------------------------------
# SIGN MAPPING
# ------------------------------------------------------------

SIGN_MAPPING = {
    "<": "<", "‹": "<", "≤": "<=", "<=": "<=",
    ">": ">", "›": ">", "≥": ">=", ">=": ">=",
}


# ------------------------------------------------------------
# SPECIAL ELEMENT VALUE MAPPING
# ------------------------------------------------------------

VALUE_MAPPING = {
    "-": "-", "–": "-", "—": "-", "−": "-",
    "bal": "Balance", "bai": "Balance", "ba1": "Balance", "bal.": "Balance",
    "8": "Balance", "83": "Balance", "b8": "Balance",
    "balance": "Balance", "-balance": "Balance", "--balance": "Balance",
    "report": "Report", "rep": "Report",
    "banch": "Banch", "trace": "Trace",
}


# ------------------------------------------------------------
# RANGE MAPPING
# ------------------------------------------------------------

RANGE_MAPPING = {
    "–": "-", "—": "-", "−": "-",
}


# ============================================================
# 4. VALID ELEMENTS
# ============================================================

VALID_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Cb",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}


# ============================================================
# 5. OCR SETTINGS
# ============================================================

OCR_SCALE = 4.0

OCR_PAD_X = 0.08
OCR_PAD_Y = 0.12

VALUE_PAD_X = 0.01
VALUE_PAD_Y = 0.03

RANGE_PAD_X = 0.06
RANGE_PAD_Y = 0.05

OCR_MIN_CONFIDENCE = 0.15
SYMBOL_MIN_CONFIDENCE = 0.10

DUPLICATE_SYMBOL_FALLBACKS = {
    "Ti": ["Tl"],
    "Tl": ["Ti"],
}


# ============================================================
# 6. OVERLAY SETTINGS
# ============================================================

SHOW_OVERLAY = True
SHOW_OVERLAY_TEXT = True
SHOW_CLASS_PREFIX = True

OVERLAY_BOX_THICKNESS = 1
OVERLAY_FONT_SCALE = 0.34
OVERLAY_FONT_THICKNESS = 1
OVERLAY_TEXT_PADDING = 3
OVERLAY_LABEL_ALPHA = 0.82
OVERLAY_MAX_TEXT_LENGTH = 22


# ============================================================
# 7. DISPLAY COLOURS
# ============================================================

CLASS_COLOURS = {
    "element_symbol": (215, 50, 50),
    "element_value": (255, 140, 0),
    "unit": (30, 170, 60),
    "limit_indicator": (145, 70, 200),
    "table": (0, 190, 210),
    "headers": (90, 95, 230),
}

CLASS_SHORT = {
    "element_symbol": "E",
    "element_value": "V",
    "unit": "U",
    "limit_indicator": "L",
    "table": "T",
    "headers": "H",
}


def rgb_to_hex(colour):
    return "#{:02x}{:02x}{:02x}".format(int(colour[0]), int(colour[1]), int(colour[2]))


def format_duration(seconds):
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remaining_seconds = seconds - minutes * 60
    return f"{minutes:02d}m {remaining_seconds:05.2f}s"


# ============================================================
# ROTATION & DESKEWING UTILITIES
# ============================================================

def rotate_image_by_angle(image, angle):
    if abs(angle) < 0.1:
        return image, None
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    return rotated, M


def estimate_angle_from_headers(detections):
    headers = [d for d in detections if d["class_name"] == "headers"]
    if len(headers) < 2:
        return 0.0

    headers.sort(key=lambda h: h["cx"])
    xs = np.array([h["cx"] for h in headers], dtype=np.float64)
    ys = np.array([h["cy"] for h in headers], dtype=np.float64)

    A = np.vstack([xs, np.ones(len(xs))]).T
    m, c = np.linalg.lstsq(A, ys, rcond=None)[0]

    angle_rad = math.atan(m)
    angle_deg = math.degrees(angle_rad)

    if abs(angle_deg) > 20.0:
        return 0.0

    return angle_deg


# ============================================================
# HELPER & OCR CONFUSION CORRECTION FUNCTIONS
# ============================================================

def clean_text(value):
    value = str(value).replace("\n", " ")
    return re.sub(r"\s+", " ", value).strip()


def mapping_key(text):
    return clean_text(text).lower()


def apply_mapping(text, mapping):
    key = mapping_key(text)
    return mapping.get(key)


def fix_ocr_digit_confusions(text):
    if not text:
        return text

    clean = text.strip()

    confusions = {
        "OLOOuzJ": "0.0002",
        "0.00022": "0.0002",
        "0.000022": "0.0002",
        "0.85": "0.65",
        "8": "Bal",
        "83": "Bal",
        "B8": "Bal",
    }

    if clean in confusions:
        return confusions[clean]

    clean = re.sub(r"^0(0\d+)$", r"0.\1", clean)
    clean = re.sub(r"^0([1-9]\d*)$", r"0.\1", clean)
    clean = re.sub(r"^0[ \t]*1[ \t]*(\d)$", r"0.\1", clean)

    char_map = {
        'O': '0', 'o': '0', 'D': '0', 'Q': '0',
        'I': '1', 'l': '1', '|': '1', 'i': '1',
        'Z': '2', 'z': '2',
        'S': '5', 's': '5',
        'J': '', 'j': '',
    }

    if re.search(r'\d', clean) or any(c in clean for c in "OLOOuzJ"):
        res = []
        for ch in clean:
            if ch in char_map:
                res.append(char_map[ch])
            else:
                res.append(ch)
        clean = "".join(res)

    return clean


# ============================================================
# LOAD IMAGE & PDF
# ============================================================

def load_image_rgb(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not open image:\n{path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def pdf_page_to_rgb(pdf_path, page_number, dpi=PDF_DPI):
    with pymupdf.open(str(pdf_path)) as document:
        if document.page_count == 0:
            raise ValueError("The PDF contains no pages.")
        page_number = max(0, min(page_number, document.page_count - 1))
        page = document[page_number]
        scale = dpi / 72.0
        matrix = pymupdf.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image.copy()


def padded_crop(image, detection):
    x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    class_name = detection["class_name"]

    if class_name == "element_value":
        pad_x_fraction, pad_y_fraction = VALUE_PAD_X, VALUE_PAD_Y
    elif class_name == "value_range":
        pad_x_fraction, pad_y_fraction = RANGE_PAD_X, RANGE_PAD_Y
    else:
        pad_x_fraction, pad_y_fraction = OCR_PAD_X, OCR_PAD_Y

    pad_x = max(0, int(width * pad_x_fraction))
    pad_y = max(0, int(height * pad_y_fraction))

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(image.shape[1], x2 + pad_x)
    crop_y2 = min(image.shape[0], y2 + pad_y)

    return image[crop_y1:crop_y2, crop_x1:crop_x2]


def advanced_multiline_header_crop(image, detection):
    """Expand a header box vertically to include stacked header lines.

    Example: YOLO may cover only ``number`` while ``Batch`` is printed on the
    line immediately above it. This crop deliberately includes both lines but
    keeps horizontal expansion small to avoid the neighbouring column.
    """
    x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = max(6, int(width * 0.18))
    pad_above = max(10, int(height * 1.55))
    pad_below = max(8, int(height * 0.85))
    return image[
        max(0, y1 - pad_above):min(image.shape[0], y2 + pad_below),
        max(0, x1 - pad_x):min(image.shape[1], x2 + pad_x)
    ]


def strict_header_crop(image, detection):
    """Crop strictly to the YOLO header box so neighbouring/underlying text cannot leak into header OCR."""
    x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
    return image[
        max(0, y1):min(image.shape[0], y2),
        max(0, x1):min(image.shape[1], x2)
    ]


def fast_multiline_header_crop(image, detection):
    """Small fast expansion for stacked headings such as Batch / number."""
    x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = max(4, int(width * 0.12))
    pad_above = max(8, int(height * 1.20))
    pad_below = max(6, int(height * 0.65))
    return image[
        max(0, y1 - pad_above):min(image.shape[0], y2 + pad_below),
        max(0, x1 - pad_x):min(image.shape[1], x2 + pad_x)
    ]


def preprocess_variants(crop):
    if crop is None or crop.size == 0:
        return []

    enlarged = cv2.resize(crop, None, fx=OCR_SCALE, fy=OCR_SCALE, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_RGB2GRAY) if enlarged.ndim == 3 else enlarged.copy()

    denoised = cv2.bilateralFilter(gray, 5, 75, 75)
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(sharpened)
    otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)

    return [clahe, otsu, adaptive, gray]


def preprocess_value_variants(crop):
    if crop is None or crop.size == 0:
        return []

    enlarged = cv2.resize(crop, None, fx=6.0, fy=6.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_RGB2GRAY) if enlarged.ndim == 3 else enlarged.copy()

    padded = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4)).apply(padded)
    gamma_corrected = np.array(255 * (padded / 255.0) ** 1.8, dtype=np.uint8)

    return [padded, clahe, gamma_corrected]


def preprocess_advanced_variants(crop):
    """Slower, high-detail views for poor-resolution text."""
    if crop is None or crop.size == 0:
        return []

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop.copy()
    variants = []
    for scale in (6.0, 8.0, 10.0):
        enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        enlarged = cv2.copyMakeBorder(enlarged, 24, 24, 32, 32, cv2.BORDER_CONSTANT, value=255)
        denoised = cv2.bilateralFilter(enlarged, 5, 45, 45)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(denoised)
        blur = cv2.GaussianBlur(clahe, (0, 0), 1.2)
        sharp = cv2.addWeighted(clahe, 1.8, blur, -0.8, 0)
        otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        adaptive = cv2.adaptiveThreshold(
            sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 41, 7
        )
        variants.extend([enlarged, clahe, sharp, otsu, adaptive])
    return variants


def analyse_numeric_glyph_layout(crop):
    """Estimate digit count and visible decimal position from image geometry.

    This does not recognise the characters. It checks the printed shapes so an
    OCR string cannot silently remove a dot (1.4 -> 14) or add a zero
    (0.005 -> 0.0005).
    """
    if crop is None or crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop.copy()
    gray = cv2.resize(gray, None, fx=5.0, fy=5.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.copyMakeBorder(gray, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Ignore thin remnants exactly on a YOLO crop edge/table border.
    edge = max(2, int(min(binary.shape[:2]) * 0.025))
    binary[:edge, :] = 0
    binary[-edge:, :] = 0
    binary[:, :edge] = 0
    binary[:, -edge:] = 0

    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    components = []
    image_h, image_w = binary.shape[:2]
    for index in range(1, count):
        x, y, width, height, area = [int(v) for v in stats[index]]
        if area < 6 or width <= 0 or height <= 0:
            continue
        if width > image_w * 0.72 or height > image_h * 0.92:
            continue
        components.append({
            "x": x, "y": y, "w": width, "h": height, "area": area,
            "cx": float(centroids[index][0]), "cy": float(centroids[index][1]),
        })

    if len(components) < 2:
        return None

    max_height = max(item["h"] for item in components)
    max_area = max(item["area"] for item in components)
    digits = [
        item for item in components
        if item["h"] >= max_height * 0.55 and item["area"] >= max_area * 0.025
    ]
    digits.sort(key=lambda item: item["cx"])
    if len(digits) < 2:
        return None

    median_height = float(np.median([item["h"] for item in digits]))
    median_width = float(np.median([item["w"] for item in digits]))
    median_top = float(np.median([item["y"] for item in digits]))
    digit_ids = {id(item) for item in digits}
    dots = [
        item for item in components
        if id(item) not in digit_ids
        and item["h"] <= median_height * 0.38
        and item["w"] <= max(3.0, median_width * 0.45)
        and item["cy"] >= median_top + median_height * 0.58
        and digits[0]["cx"] < item["cx"] < digits[-1]["cx"]
    ]
    if not dots:
        return {"digit_count": len(digits), "has_decimal": False, "decimal_after": None}

    # Prefer the compact lower dot closest to the digit baseline.
    dot = max(dots, key=lambda item: (item["cy"], -item["area"]))
    decimal_after = sum(item["cx"] < dot["cx"] for item in digits)
    if decimal_after <= 0 or decimal_after >= len(digits):
        return {"digit_count": len(digits), "has_decimal": False, "decimal_after": None}

    return {
        "digit_count": len(digits),
        "has_decimal": True,
        "decimal_after": decimal_after,
    }


def analyse_numeric_glyph_layout_multiview(crop):
    """Repeat dot/layout analysis at several thresholds for faint tiny dots."""
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop.copy()
    views = [gray]
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(gray)
    views.append(clahe)
    for source in (gray, clahe):
        for threshold_value in (110, 140, 170, 200, 225):
            views.append(cv2.threshold(source, threshold_value, 255, cv2.THRESH_BINARY)[1])

    layouts = []
    for view in views:
        layout = analyse_numeric_glyph_layout(view)
        if layout:
            layouts.append(layout)
    decimal_layouts = [item for item in layouts if item.get("has_decimal")]
    if decimal_layouts:
        counts = {}
        for item in decimal_layouts:
            key = (item["digit_count"], item["decimal_after"])
            counts[key] = counts.get(key, 0) + 1
        best_key = max(counts, key=lambda key: counts[key])
        return {
            "digit_count": best_key[0],
            "has_decimal": True,
            "decimal_after": best_key[1],
            "layout_votes": counts[best_key],
        }
    return layouts[0] if layouts else None


def add_visual_decimal_candidates(candidates, crop):
    """Create candidates whose decimal position matches the visible glyphs."""
    layout = analyse_numeric_glyph_layout_multiview(crop)
    if not layout or not layout.get("has_decimal"):
        return candidates, layout

    target_digits = int(layout["digit_count"])
    decimal_after = int(layout["decimal_after"])
    corrected = []

    for candidate in candidates:
        final, valid = normalise_value(candidate.get("raw", ""))
        if not valid or not re.fullmatch(r"[<>]=?[+-]?[0-9.,]+", final):
            continue

        prefix_match = re.match(r"^[<>]=?[+-]?", final)
        prefix = prefix_match.group(0) if prefix_match else ""
        numeric_body = final[len(prefix):]
        digits = re.sub(r"\D", "", numeric_body)
        if not digits:
            continue

        # Match the number of printed tall glyphs. Excess OCR zeros are removed
        # from the decimal body; missing interior zeros are restored there.
        if len(digits) > target_digits:
            # If OCR inserted a leading artifact before its own decimal
            # (70.005), retain the digit(s) nearest that decimal (0), rather
            # than blindly retaining the first character (7).
            source_separator = max(numeric_body.rfind("."), numeric_body.rfind(","))
            if source_separator >= 0:
                source_left = re.sub(r"\D", "", numeric_body[:source_separator])
                left = source_left[-decimal_after:]
            else:
                left = digits[:decimal_after]
            right = digits[-max(0, target_digits - decimal_after):]
            digits = left + right
        elif len(digits) < target_digits:
            left = digits[:min(decimal_after, len(digits))]
            right = digits[len(left):]
            missing = target_digits - len(digits)
            digits = left + ("0" * missing) + right

        if len(digits) != target_digits:
            continue
        rebuilt = prefix + digits[:decimal_after] + "." + digits[decimal_after:]
        rebuilt_final, rebuilt_valid = normalise_value(rebuilt)
        if rebuilt_valid:
            corrected.append({
                **candidate,
                "raw": candidate.get("raw", ""),
                "visual_final": rebuilt_final,
                "layout_verified": True,
                "confidence": max(0.10, float(candidate.get("confidence", 0.0))),
            })

    # Convert visual_final into the standard candidate raw field only for the
    # consensus stage; retain source_raw for the GUI/debug display.
    for item in corrected:
        candidates.append({
            **item,
            "source_raw": item["raw"],
            "raw": item["visual_final"],
            "variant": f"layout_{item.get('variant', -1)}",
            "forced": True,
        })
    return candidates, layout


def strict_dash_fallback(crop):
    if crop is None or crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop.copy()
    gray = cv2.resize(gray, None, fx=5.0, fy=5.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.copyMakeBorder(gray, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_h, image_w = binary.shape[:2]

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width * height < 20:
            continue
        if width / max(height, 1) < 3.2:
            continue
        if width > image_w * 0.65 or height > image_h * 0.22:
            continue
        cx, cy = x + width / 2, y + height / 2
        if abs(cx - image_w / 2) < image_w * 0.32 and abs(cy - image_h / 2) < image_h * 0.32:
            return True
    return False


# ============================================================
# NORMALISATION ROUTINES
# ============================================================

def normalise_element(text):
    cleaned = clean_text(text)

    # Some document headings are detected by YOLO as element_symbol because
    # they occupy the same visual type of box as chemical column headings.
    # Resolve these structural labels before trying chemical-element matching.
    structural_compact = re.sub(r"[^a-z0-9]", "", cleaned.lower())

    batch_forms = (
        "batch", "batchno", "batchnum", "batchnumber", "batchnr",
        "bat", "batno", "batnum", "batnumber", "atch", "atchno",
        "atchnumber", "btch", "bch", "bafch", "balch", "baich",
    )
    lot_forms = (
        "lot", "lots", "lotno", "lotnum", "lotnumber", "lotsnumber",
        "lotnr", "lotn", "lotnumbers",
    )
    number_forms = {
        "number", "numbers", "num", "no", "nr",
        "numbe", "numb", "nurnber", "nurnbcr", "nurnbr", "nurnber",
    }
    net_weight_forms = (
        "netweight", "netweights", "netwt", "netwgt", "nettweight",
        "weightnet", "wtnet", "wgtnet", "netmass", "massnet",
        "massweight", "weightmass", "net",
    )
    weight_forms = (
        "weight", "weights", "wight", "weght", "weignt", "we1ght",
        "weigt", "wieght", "wgt", "wt", "mass",
    )

    has_number = (
        "number" in structural_compact
        or "nurnber" in structural_compact
        or "num" in structural_compact
        or structural_compact.endswith(("no", "nr"))
    )

    # Structural labels intentionally accepted inside element_symbol boxes.
    # Keep these checks BEFORE chemical-element matching.
    if structural_compact in number_forms:
        return "Batch number", True
    if any(form in structural_compact for form in batch_forms):
        return ("Batch number" if has_number else "Batch"), True
    if any(form in structural_compact for form in lot_forms):
        return ("Lot number" if has_number else "Lot"), True
    if structural_compact in net_weight_forms or any(
        form in structural_compact
        for form in ("netweight", "nettweight", "netmass", "massnet", "weightnet")
    ):
        return "Net weight", True
    if structural_compact in weight_forms:
        return ("Net weight" if "mass" in structural_compact else "Weight"), True

    paren_match = re.search(r"\((.*?)\)", cleaned)
    if paren_match:
        inside = paren_match.group(1).strip()
        if inside in VALID_ELEMENTS:
            return inside, True

    words = re.findall(r"[A-Za-z]+", cleaned)
    for word in words:
        w_lower = word.lower()
        if w_lower in FULL_CHEMICAL_NAMES:
            return FULL_CHEMICAL_NAMES[w_lower], True
        mapped = apply_mapping(word, ELEMENT_MAPPING)
        if mapped is not None:
            return mapped, True
        if len(word) in (1, 2):
            candidate = word[0].upper() + (word[1].lower() if len(word) == 2 else "")
            if candidate in VALID_ELEMENTS:
                return candidate, True

    return cleaned, False


def normalise_unit(text):
    mapped = apply_mapping(text, UNIT_MAPPING)
    if mapped is not None:
        return mapped, True
    token = clean_text(text).lower().replace(" ", "")
    if "%" in token:
        return "%", True
    if "ppm" in token:
        return "ppm", True
    if "ppb" in token:
        return "ppb", True
    tonne_token = re.sub(r"[^a-z]", "", token)
    if tonne_token in {"t", "kt", "tj", "ktj", "ton", "tons", "tonne", "tonnes"}:
        return "t", True
    compact_unit = re.sub(r"[^a-z0-9]", "", token)
    if compact_unit in {"kg", "kgs", "k9", "kq", "kc", "kilogram", "kilograms"}:
        return "kg", True
    if compact_unit in {
        "mt", "mtt", "metricton", "metrictons", "metrictonne", "metrictonnes"
    }:
        return "mt", True
    return clean_text(text), False


def normalise_limit(text):
    mapped = apply_mapping(text, LIMIT_MAPPING)
    if mapped is not None:
        return mapped, True
    token = clean_text(text).lower()
    if "max" in token:
        return "MAX", True
    if "min" in token:
        return "MIN", True
    return clean_text(text), False


def normalise_sign(text):
    mapped = apply_mapping(text, SIGN_MAPPING)
    if mapped is not None:
        return mapped, True
    value = clean_text(text).replace("≤", "<=").replace("≥", ">=")
    if value in {"<", ">", "<=", ">="}:
        return value, True
    return value, False


def normalise_value(text):
    original = clean_text(text)

    # Text values are valid in element_value boxes as well as numbers.
    # Strip punctuation only for semantic matching, so forms such as
    # --Balance, -Balance, Bal., --bal-- and similar OCR remain recoverable.
    semantic_letters = re.sub(r"[^a-z]", "", original.lower())

    if "balance" in semantic_letters or semantic_letters.startswith("bal"):
        return "Balance", True

    if semantic_letters in {"report", "rep"} or semantic_letters.startswith("report"):
        return "Report", True

    original = fix_ocr_digit_confusions(original)

    mapped = apply_mapping(original, VALUE_MAPPING)
    if mapped is not None:
        return mapped, True

    clean_val = re.sub(r"\s*(?:%|wt%|ppm|ppb)$", "", original, flags=re.IGNORECASE)
    clean_val = re.sub(r"(?<=\d)\s*[-–—−]\s*[^0-9\s.+-]+\s*(?=\d)", " - ", clean_val)
    clean_val = re.sub(r"(?<=\d)\s*[-–—−]\s*(?=\d)", " - ", clean_val)

    value = clean_val.translate(str.maketrans({
        "O": "0", "o": "0",
        "I": "1", "l": "1", "|": "1",
    }))
    value = re.sub(r"\s+", " ", value).strip()

    sign_prefix = ""
    separator_test = value
    if separator_test[:1] in "+-":
        sign_prefix, separator_test = separator_test[0], separator_test[1:]
    if "," in separator_test and "." in separator_test:
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+\.\d+", separator_test):
            value = sign_prefix + separator_test
        elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d+", separator_test):
            integer_part, decimal_part = separator_test.rsplit(",", 1)
            value = sign_prefix + integer_part.replace(".", ",") + "." + decimal_part
        else:
            last_separator = max(separator_test.rfind("."), separator_test.rfind(","))
            integer_part = re.sub(r"[.,]", "", separator_test[:last_separator])
            decimal_part = re.sub(r"[.,]", "", separator_test[last_separator + 1:])
            value = sign_prefix + integer_part + "." + decimal_part
    elif "," in separator_test:
        value = sign_prefix + separator_test.replace(",", ".")

    if value == "-":
        return "-", True

    number_pattern = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    range_match = re.fullmatch(rf"({number_pattern})\s*-\s*({number_pattern})", value)
    if range_match:
        return f"{range_match.group(1)} - {range_match.group(2)}", True

    val_no_space = value.replace(" ", "")
    if re.fullmatch(number_pattern, val_no_space):
        return val_no_space, True

    if re.fullmatch(rf"[<>]=?{number_pattern}", val_no_space):
        return val_no_space, True

    if re.fullmatch(r"[A-Za-z]+", original):
        return original, True

    return value, False


def normalise_header(text):
    original = clean_text(text)
    if not original:
        return "", False

    lower = original.lower()
    compact = re.sub(r"[^a-z]", "", lower)

    # Header OCR is deliberately semantic: YOLO already says this box is a
    # header, so surrounding OCR garbage must not defeat a clearly visible
    # header keyword. Examples:
    #   "I_ Element Bismus" -> "Element"
    #   "Symble EI"         -> "Symbol"
    #   "Ld1R Percentage%"  -> "Percentage %"
    header_keyword_rules = (
        (("percentage", "percentag", "percent", "perc"), "Percentage %"),
        (("element", "elernent", "elemnt", "elment", "e1ement"), "Element"),
        (("symbol", "symble", "symbo", "symbl", "syrnbol", "syml"), "Symbol"),
        (("analysis", "ana1ysis", "analysls", "anaiysis"), "Analysis"),
    )
    for variants, final in header_keyword_rules:
        if any(token in compact for token in variants):
            return final, True

    # A standalone % header is also a percentage heading.
    if "%" in original and len(re.sub(r"[^A-Za-z]", "", original)) <= 6:
        return "Percentage %", True

    batch_forms = {
        "batch", "bat", "atch", "btch", "bafch", "balch", "baich",
        "batchno", "batchnum", "batchnumber", "batchnr",
        "batnumber", "atchnumber", "btchnumber",
    }
    lot_forms = {
        "lot", "lots", "lotno", "lotnumber", "lotsnumber", "lotnum", "lotnr",
    }
    weight_forms = {
        "weight", "weights", "wight", "weght", "weignt", "we1ght",
        "weigt", "wieght", "netweight", "nettweight", "weightnet",
        "netwt", "wt", "wgt", "mass", "netmass", "massnet",
    }

    has_number = (
        "number" in compact or "nurnber" in compact
        or compact.endswith(("no", "num", "nr"))
    )
    has_batch = any(form in compact for form in batch_forms)
    has_lot = any(form in compact for form in lot_forms)
    has_weight = any(form in compact for form in weight_forms)
    has_net = "net" in compact or "nett" in compact
    has_mass = "mass" in compact

    if has_batch:
        return "Batch number" if has_number else "Batch", True
    if has_lot:
        return "Lot number" if has_number else "Lot", True
    if has_weight:
        if has_net or has_mass:
            return "Net weight", True
        return "Weight", True
    if compact in {"number", "numbe", "numb", "nurnber", "nurnbcr", "nurnbr"}:
        return "Batch number", True

    return original, True


BATCH_HEADER_WORDS = {
    "batch", "bat", "lot", "lots", "sample", "number", "no", "batchnumber", "lotnumber"
}


def is_batch_header_text(text):
    compact = re.sub(r"[^a-z]", "", clean_text(text).lower())
    words = re.findall(r"[a-z]+", clean_text(text).lower())
    return compact in BATCH_HEADER_WORDS or any(word in BATCH_HEADER_WORDS for word in words)


def normalise_batch_identifier(text):
    """Keep the structure of identifiers such as 9/3647A."""
    value = clean_text(text).upper().replace(" ", "")
    value = value.replace("\\", "/").replace("|", "/")
    value = re.sub(r"[^A-Z0-9/._-]", "", value)
    value = re.sub(r"/{2,}", "/", value)
    valid = bool(value and re.fullmatch(r"[A-Z0-9]+(?:[/._-][A-Z0-9]+)*", value))
    return value, valid


def normalise_range(text):
    value = clean_text(text)
    for old, new in (("–", "-"), ("—", "-"), ("−", "-")):
        value = value.replace(old, new)
    value = value.replace(",", ".").replace(" ", "")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?-[+-]?\d+(?:\.\d+)?", value):
        return value, True
    return value, False


def get_allowlist(class_name):
    if class_name == "element_symbol":
        return "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz()"
    if class_name == "unit":
        return "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789%./()[]"
    if class_name == "limit_indicator":
        return "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    if class_name == "value_range":
        return "0123456789.,+-"
    if class_name == "sign":
        return "<>=-"
    return None


# ============================================================
# RUN EASYOCR
# ============================================================

def run_easyocr_candidates(reader, variants, allowlist, text_threshold=0.35, low_text=0.15, link_threshold=0.20):
    candidates = []
    for variant_index, variant in enumerate(variants):
        try:
            results = reader.readtext(
                variant, detail=1, paragraph=False, allowlist=allowlist,
                decoder="greedy", contrast_ths=0.05, adjust_contrast=0.7,
                text_threshold=text_threshold, low_text=low_text, link_threshold=link_threshold
            )
            if not results:
                continue
            raw = clean_text(" ".join(item[1] for item in results))
            confidence = float(np.mean([item[2] for item in results]))
            if raw:
                candidates.append({"raw": raw, "confidence": confidence, "variant": variant_index})
        except Exception:
            continue
    return candidates


def read_fast_header(reader, crop):
    """Fast semantic header OCR; complete category beats a partial word."""
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789%()./_-"
    candidates = run_easyocr_candidates(
        reader, preprocess_variants(crop), allowlist,
        text_threshold=0.28, low_text=0.10, link_threshold=0.16
    )
    if not candidates:
        return "", 0.0, False, ""

    prepared = []
    semantic_headers = {"Batch", "Batch number", "Lot", "Lot number", "Weight", "Net weight", "Analysis", "Element", "Symbol", "Percentage %"}
    for item in candidates:
        final, valid = normalise_header(item["raw"])
        if not final:
            continue
        prepared.append({
            **item,
            "final": final,
            "valid": valid,
            "semantic": final in semantic_headers,
            "word_count": len(re.findall(r"[A-Za-z0-9]+", clean_text(item["raw"]))),
        })
    if not prepared:
        return "", 0.0, False, ""

    best = max(
        prepared,
        key=lambda item: (
            item["semantic"],
            item["word_count"],
            len(item["final"]),
            float(item["confidence"]),
        )
    )
    return best["final"], float(best["confidence"]), bool(best["valid"]), best["raw"]


def run_forced_whole_crop_candidates(reader, variants, allowlist):
    """Bypass EasyOCR text detection and recognise the complete YOLO crop."""
    candidates = []
    for variant_index, variant in enumerate(variants):
        if variant is None or variant.size == 0:
            continue
        height, width = variant.shape[:2]
        for decoder in ("greedy", "beamsearch"):
            try:
                results = reader.recognize(
                    variant,
                    horizontal_list=[[0, width, 0, height]],
                    free_list=[],
                    decoder=decoder,
                    beamWidth=12,
                    detail=1,
                    paragraph=False,
                    allowlist=allowlist,
                    contrast_ths=0.02,
                    adjust_contrast=0.85,
                )
                if not results:
                    continue
                raw = clean_text(" ".join(item[1] for item in results))
                confidence = float(np.mean([item[2] for item in results]))
                if raw:
                    candidates.append({
                        "raw": raw, "confidence": confidence,
                        "variant": variant_index, "forced": True,
                        "decoder": decoder,
                    })
            except Exception:
                continue
    return candidates


def choose_consensus_candidate(candidates, normaliser, prefer_slash=False, prefer_decimal=False):
    processed = []
    for candidate in candidates:
        final, valid = normaliser(candidate["raw"])
        if valid and final:
            processed.append({**candidate, "final": final})
    if not processed:
        return "", 0.0, False, ""

    # Image geometry is stronger evidence than OCR confidence for decimal
    # placement and printed character count.
    if prefer_decimal and any(item.get("layout_verified", False) for item in processed):
        processed = [item for item in processed if item.get("layout_verified", False)]

    # In a confirmed Batch/Lot column, a detected separator is meaningful and
    # must not be discarded merely because a digits-only hallucination has a
    # higher confidence (9/3647A must not become 9326474).
    if prefer_slash and any("/" in item["final"] for item in processed):
        processed = [item for item in processed if "/" in item["final"]]
        if any(re.search(r"[A-Z]$", item["final"]) for item in processed):
            processed = [item for item in processed if re.search(r"[A-Z]$", item["final"])]

    # If any advanced view sees a decimal point, do not allow a digits-only
    # reading such as 15 to replace 1.5, or 00001 to replace 0.0001.
    if prefer_decimal and any(re.search(r"\d\.\d", item["final"]) for item in processed):
        processed = [item for item in processed if re.search(r"\d\.\d", item["final"])]

    variants_by_text = {}
    forced_by_text = {}
    best_confidence = {}
    for item in processed:
        key = item["final"]
        variants_by_text.setdefault(key, set()).add(item.get("variant", -1))
        forced_by_text[key] = forced_by_text.get(key, 0) + int(item.get("forced", False))
        best_confidence[key] = max(best_confidence.get(key, 0.0), float(item["confidence"]))

    def score(item):
        final = item["final"]
        result = 2.0 * len(variants_by_text[final]) + best_confidence[final]
        result += 0.30 * forced_by_text[final]
        if prefer_slash and "/" in final:
            result += 2.25
        if prefer_slash and re.search(r"[A-Z]$", final):
            result += 0.60
        if prefer_decimal and re.search(r"\d\.\d", final):
            result += 1.50
        if prefer_decimal and re.fullmatch(r"[<>]=?0\.\d+", final):
            result += 1.00
        return result

    best = max(processed, key=lambda item: (score(item), item["confidence"], len(item["final"])))
    return best["final"], float(best["confidence"]), True, best.get("source_raw", best["raw"])


def read_batch_identifier_advanced(reader, crop):
    variants = preprocess_advanced_variants(crop)
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/._-"
    candidates = run_easyocr_candidates(
        reader, variants, allowlist, text_threshold=0.18, low_text=0.05, link_threshold=0.10
    )
    candidates.extend(run_forced_whole_crop_candidates(reader, variants, allowlist))
    return choose_consensus_candidate(candidates, normalise_batch_identifier, prefer_slash=True)


def read_element_value_advanced(reader, crop):
    # Keep the established text-value priority for Bal/Balance/Trace.
    fast_result = read_element_value(reader, crop)
    variants = preprocess_advanced_variants(crop)
    allowlist = "0123456789.,+-%<>"
    candidates = run_easyocr_candidates(
        reader, variants, allowlist, text_threshold=0.16, low_text=0.04, link_threshold=0.08
    )
    candidates.extend(run_forced_whole_crop_candidates(reader, variants, allowlist))
    candidates, visual_layout = add_visual_decimal_candidates(candidates, crop)
    advanced = choose_consensus_candidate(candidates, normalise_value, prefer_decimal=True)

    if fast_result[2] and fast_result[0] in {"Bal", "Balance", "Trace", "-"}:
        return fast_result
    if advanced[2]:
        return advanced
    return fast_result


def read_multiline_header_advanced(reader, crop):
    """Read and combine one- or two-line column headings."""
    variants = preprocess_advanced_variants(crop)
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789%./_-"
    candidates = run_easyocr_candidates(
        reader, variants, allowlist,
        text_threshold=0.15, low_text=0.035, link_threshold=0.08
    )
    candidates.extend(run_forced_whole_crop_candidates(reader, variants, allowlist))
    if not candidates:
        return "", 0.0, False, ""

    prepared = []
    for item in candidates:
        raw = clean_text(item.get("raw", ""))
        if not raw:
            continue
        compact = re.sub(r"[^a-z]", "", raw.lower())
        words = re.findall(r"[A-Za-z0-9%]+", raw)
        has_batch = "batch" in compact or re.search(r"\bbat\b", raw, re.IGNORECASE)
        has_lot = "lot" in compact
        has_number = (
            "number" in compact or "nurnber" in compact
            or bool(re.search(r"\b(?:no|nr)\b", raw, re.IGNORECASE))
        )

        final, _ = normalise_header(raw)
        semantic = final in {
            "Batch", "Batch number", "Lot", "Lot number", "Weight", "Net weight",
            "Analysis", "Element", "Symbol", "Percentage %"
        }

        prepared.append({
            **item, "raw": raw, "final": final,
            "category_score": int(semantic) * 6 + int(has_batch or has_lot) * 4 + int(has_number) * 2,
            "word_score": min(len(words), 4),
        })

    if not prepared:
        return "", 0.0, False, ""
    best = max(
        prepared,
        key=lambda item: (
            item["category_score"], item["word_score"],
            len(item["final"]), float(item["confidence"])
        )
    )
    return best["final"], float(best["confidence"]), True, best["raw"]


def read_element_value(reader, crop):
    variants = preprocess_value_variants(crop)
    if not variants:
        return "", 0.0, False, ""

    text_results = run_easyocr_candidates(
        reader, variants, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-", text_threshold=0.25, low_text=0.08, link_threshold=0.15
    )

    mapped_text_candidates = []
    for candidate in text_results:
        raw_candidate = clean_text(candidate["raw"])

        # First use the full value normaliser. This makes semantic text values
        # reachable from the fast OCR path instead of falling through to the
        # numeric-only OCR pass and becoming "?".
        final_text, valid_text = normalise_value(raw_candidate)

        if valid_text and final_text in {"Balance", "Report", "Trace"}:
            mapped_text_candidates.append({
                "raw": candidate["raw"],
                "final": final_text,
                "confidence": candidate["confidence"]
            })
            continue

        raw_fixed = fix_ocr_digit_confusions(raw_candidate)
        mapped = apply_mapping(raw_fixed, VALUE_MAPPING)
        if mapped is not None and mapped != "-":
            mapped_text_candidates.append({
                "raw": candidate["raw"],
                "final": mapped,
                "confidence": candidate["confidence"]
            })

    if mapped_text_candidates:
        # Prefer the semantic result supported by OCR confidence.
        best = max(mapped_text_candidates, key=lambda item: item["confidence"])
        return best["final"], best["confidence"], True, best["raw"]

    numeric_results = run_easyocr_candidates(
        reader, variants, "0123456789.,+-%<>", text_threshold=0.25, low_text=0.08, link_threshold=0.15
    )

    numeric_candidates = []
    for candidate in numeric_results:
        raw = fix_ocr_digit_confusions(candidate["raw"])
        final, valid = normalise_value(raw)
        if valid and final != "-":
            numeric_candidates.append({"raw": candidate["raw"], "final": final, "confidence": candidate["confidence"]})

    if numeric_candidates:
        support = {}
        best_confidence = {}
        for item in numeric_candidates:
            key = item["final"]
            support[key] = support.get(key, 0) + 1
            best_confidence[key] = max(best_confidence.get(key, 0.0), float(item["confidence"]))

        has_small_decimal = any(re.fullmatch(r"[<>]=?0\.\d+", key) for key in support)

        def value_candidate_score(item):
            final = item["final"]
            score = support[final] * 2.0 + best_confidence[final]
            if re.fullmatch(r"[<>]=?0\.\d+", final):
                score += 0.35
            if has_small_decimal and re.fullmatch(r"0{2,}\d+", final):
                score -= 1.50
            if re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", final):
                score += 0.10
            return score

        best = max(numeric_candidates, key=lambda item: (
            value_candidate_score(item),
            best_confidence[item["final"]],
            -len(item["final"])
        ))
        return best["final"], best["confidence"], True, best["raw"]

    dash_candidates = [c for c in numeric_results if clean_text(c["raw"]).replace("–", "-").replace("—", "-").replace("−", "-") == "-"]
    if dash_candidates:
        best = max(dash_candidates, key=lambda item: item["confidence"])
        return "-", best["confidence"], True, best["raw"]

    if strict_dash_fallback(crop):
        return "-", 0.70, True, "-"

    return "", 0.0, False, ""


def read_value_range(reader, crop):
    variants = preprocess_variants(crop)
    candidates = run_easyocr_candidates(reader, variants, "0123456789.,+-", text_threshold=0.30, low_text=0.10, link_threshold=0.16)
    if not candidates:
        return "", 0.0, False, ""

    processed = []
    for candidate in candidates:
        final, valid = normalise_range(candidate["raw"])
        processed.append({
            "raw": candidate["raw"], "final": final, "valid": valid, "confidence": candidate["confidence"]
        })

    processed.sort(key=lambda item: (item["valid"], item["confidence"], len(item["final"])), reverse=True)
    best = processed[0]
    return best["final"], best["confidence"], best["valid"], best["raw"]


def read_crop_text(reader, image, detection):
    class_name = detection["class_name"]
    if class_name in NO_OCR_CLASSES:
        return "", 0.0, True, ""

    crop = (
        strict_header_crop(image, detection)
        if class_name == "headers"
        else padded_crop(image, detection)
    )
    if crop is None or crop.size == 0:
        return "", 0.0, False, ""

    if class_name == "element_value":
        return read_element_value(reader, crop)
    if class_name == "value_range":
        return read_value_range(reader, crop)
    if class_name == "headers":
        return read_fast_header(reader, crop)

    allowlist = get_allowlist(class_name)
    candidates = run_easyocr_candidates(reader, preprocess_variants(crop), allowlist, text_threshold=0.38, low_text=0.18, link_threshold=0.22)
    if not candidates:
        return "", 0.0, False, ""

    if class_name == "element_symbol":
        best_by_symbol = {}
        for candidate in candidates:
            if float(candidate["confidence"]) <= SYMBOL_MIN_CONFIDENCE:
                continue
            candidate_text, candidate_valid = normalise_element(candidate["raw"])
            if not candidate_valid:
                continue
            previous = best_by_symbol.get(candidate_text)
            if previous is None or candidate["confidence"] > previous["confidence"]:
                best_by_symbol[candidate_text] = {
                    "text": candidate_text, "raw": candidate["raw"], "confidence": float(candidate["confidence"])
                }

        symbol_alternatives = sorted(best_by_symbol.values(), key=lambda item: item["confidence"], reverse=True)
        detection["ocr_alternatives"] = symbol_alternatives
        if symbol_alternatives:
            best_symbol = symbol_alternatives[0]
            return best_symbol["text"], best_symbol["confidence"], True, best_symbol["raw"]

        rejected_best = max(candidates, key=lambda item: item["confidence"])
        return "", float(rejected_best["confidence"]), False, rejected_best["raw"]

    raw_text, confidence = max(((item["raw"], item["confidence"]) for item in candidates), key=lambda item: (item[1], len(item[0])))

    if class_name == "element_symbol":
        final_text, mapped = normalise_element(raw_text)
    elif class_name == "unit":
        final_text, mapped = normalise_unit(raw_text)
    elif class_name == "limit_indicator":
        final_text, mapped = normalise_limit(raw_text)
    elif class_name == "sign":
        final_text, mapped = normalise_sign(raw_text)
    elif class_name == "headers":
        final_text, mapped = normalise_header(raw_text)
    else:
        final_text, mapped = raw_text, False

    return final_text, confidence, mapped, raw_text


def read_crop_text_advanced(reader, image, detection, batch_identifier=False):
    crop = (
        strict_header_crop(image, detection)
        if detection["class_name"] == "headers"
        else padded_crop(image, detection)
    )
    if crop is None or crop.size == 0:
        return "", 0.0, False, ""
    if batch_identifier:
        return read_batch_identifier_advanced(reader, crop)
    if detection["class_name"] == "element_value":
        return read_element_value_advanced(reader, crop)
    if detection["class_name"] == "headers":
        result = read_multiline_header_advanced(reader, crop)
        if result[2]:
            return result
    return read_crop_text(reader, image, detection)


def detection_is_under_batch_header(detection, recognised_headers):
    if detection["class_name"] != "element_value":
        return False
    candidates = []
    for header in recognised_headers:
        header_text = header.get("text") or header.get("raw_text", "")
        if not is_batch_header_text(header_text) or detection["cy"] <= header["cy"]:
            continue
        header_width = max(1, header["x2"] - header["x1"])
        value_width = max(1, detection["x2"] - detection["x1"])
        x_distance = abs(detection["cx"] - header["cx"])
        tolerance = max(45.0, 0.75 * max(header_width, value_width))
        if x_distance <= tolerance:
            candidates.append((x_distance, header))
    return bool(candidates)


def apply_advanced_column_decimal_consistency(detections):
    """Recover a missing dot from consistent neighbouring values in a column."""
    values = [
        item for item in detections
        if item.get("ocr_applied")
        and item.get("class_name") == "element_value"
        and not item.get("batch_identifier", False)
        and item.get("text")
    ]
    if len(values) < 3:
        return []

    widths = [max(1, item["x2"] - item["x1"]) for item in values]
    x_tolerance = max(35.0, float(np.median(widths)) * 0.55)
    clusters = []
    for item in sorted(values, key=lambda value: value["cx"]):
        target = None
        for cluster in clusters:
            if abs(item["cx"] - cluster["cx"]) <= x_tolerance:
                target = cluster
                break
        if target is None:
            clusters.append({"cx": item["cx"], "items": [item]})
        else:
            target["items"].append(item)
            target["cx"] = float(np.mean([value["cx"] for value in target["items"]]))

    corrections = []
    for cluster in clusters:
        if len(cluster["items"]) < 3:
            continue
        patterns = {}
        for item in cluster["items"]:
            text = clean_text(item.get("text", "")).replace(",", ".")
            match = re.fullmatch(r"([+-]?)(\d+)\.(\d+)", text)
            if not match:
                continue
            digit_count = len(match.group(2)) + len(match.group(3))
            decimal_after = len(match.group(2))
            key = (digit_count, decimal_after)
            patterns[key] = patterns.get(key, 0) + 1

        if not patterns:
            continue
        dominant, votes = max(patterns.items(), key=lambda pair: pair[1])
        if votes < 2:
            continue
        target_digit_count, decimal_after = dominant

        for item in cluster["items"]:
            current = clean_text(item.get("text", ""))
            match = re.fullmatch(r"([+-]?)(\d+)", current)
            if not match or len(match.group(2)) != target_digit_count:
                continue
            digits = match.group(2)
            corrected = match.group(1) + digits[:decimal_after] + "." + digits[decimal_after:]
            old_text = item["text"]
            item["text"] = corrected
            item["mapped"] = True
            item["column_decimal_corrected"] = True
            item["warning"] = float(item.get("ocr_conf") or 0.0) < OCR_MIN_CONFIDENCE
            corrections.append({
                "x": item["cx"], "y": item["cy"],
                "before": old_text, "after": corrected,
                "neighbour_votes": votes,
            })
    return corrections


def merge_stacked_batch_lot_headers(detections):
    """Merge stacked Batch/Lot and Net/Mass/Weight header boxes."""
    headers = [
        item for item in detections
        if item.get("class_name") == "headers"
        and item.get("ocr_applied")
        and (item.get("text") or item.get("raw_text"))
    ]
    changes = []
    used = set()
    for index, first in enumerate(headers):
        if index in used:
            continue
        first_text = clean_text(first.get("text") or first.get("raw_text", ""))
        first_compact = re.sub(r"[^a-z]", "", first_text.lower())
        first_height = max(1, first["y2"] - first["y1"])
        group = [(index, first)]

        for other_index, other in enumerate(headers):
            if other_index == index or other_index in used:
                continue
            other_text = clean_text(other.get("text") or other.get("raw_text", ""))
            other_compact = re.sub(r"[^a-z]", "", other_text.lower())
            relevant = any(
                word in (first_compact + " " + other_compact)
                for word in ("batch", "bat", "atch", "lot", "number", "no", "weight", "mass", "net", "wt")
            )
            if not relevant:
                continue
            other_height = max(1, other["y2"] - other["y1"])
            x_tolerance = max(30.0, 0.45 * max(first["x2"] - first["x1"], other["x2"] - other["x1"]))
            y_tolerance = 2.8 * max(first_height, other_height)
            if abs(first["cx"] - other["cx"]) <= x_tolerance and abs(first["cy"] - other["cy"]) <= y_tolerance:
                group.append((other_index, other))

        combined_parts = []
        for group_index, header in sorted(group, key=lambda pair: (pair[1]["cy"], pair[1]["cx"])):
            part = clean_text(header.get("text") or header.get("raw_text", ""))
            if part and part.lower() not in {value.lower() for value in combined_parts}:
                combined_parts.append(part)
        combined = " ".join(combined_parts)
        compact = re.sub(r"[^a-z]", "", combined.lower())
        if ("batch" in compact or "bat" in compact) and ("number" in compact or "no" in compact):
            merged = "Batch number"
        elif "lot" in compact and ("number" in compact or "no" in compact):
            merged = "Lot number"
        elif ("weight" in compact or "mass" in compact or "wt" in compact) and "net" in compact:
            merged = "Net weight"
        else:
            continue

        for group_index, header in group:
            used.add(group_index)
            before = header.get("text", "")
            header["text"] = merged
            header["mapped"] = True
            header["multiline_header_merged"] = True
            changes.append({"before": before, "after": merged, "x": header["cx"], "y": header["cy"]})
    return changes


def merge_split_net_weight_element_symbols(detections):
    """Merge adjacent element_symbol boxes ``Net`` + ``Weight`` into one box."""
    symbol_boxes = [
        item for item in detections
        if item.get("class_name") == "element_symbol"
        and item.get("ocr_applied")
        and (item.get("raw_text") or item.get("text"))
    ]

    net_boxes = []
    weight_boxes = []
    for item in symbol_boxes:
        raw = clean_text(item.get("raw_text") or item.get("text", ""))
        compact = re.sub(r"[^a-z]", "", raw.lower())
        if compact in {"net", "nett", "nct", "nel"}:
            net_boxes.append(item)
        elif compact in {
            "weight", "weights", "wight", "weght", "weignt", "we1ght",
            "wgt", "wt", "mass"
        }:
            weight_boxes.append(item)

    used_ids = set()
    replacements = {}
    changes = []

    for net_box in net_boxes:
        if id(net_box) in used_ids:
            continue
        net_height = max(1, net_box["y2"] - net_box["y1"])
        possible = []
        for weight_box in weight_boxes:
            if id(weight_box) in used_ids:
                continue
            weight_height = max(1, weight_box["y2"] - weight_box["y1"])
            same_line = abs(net_box["cy"] - weight_box["cy"]) <= 0.65 * max(net_height, weight_height)
            horizontal_gap = max(
                0,
                max(net_box["x1"], weight_box["x1"]) - min(net_box["x2"], weight_box["x2"])
            )
            max_gap = 1.25 * max(net_height, weight_height)
            if same_line and horizontal_gap <= max_gap:
                possible.append((abs(net_box["cx"] - weight_box["cx"]), weight_box))
        if not possible:
            continue

        _, weight_box = min(possible, key=lambda pair: pair[0])
        used_ids.update({id(net_box), id(weight_box)})
        left, right = sorted((net_box, weight_box), key=lambda item: item["x1"])
        merged = dict(left)
        merged.update({
            "x1": min(net_box["x1"], weight_box["x1"]),
            "y1": min(net_box["y1"], weight_box["y1"]),
            "x2": max(net_box["x2"], weight_box["x2"]),
            "y2": max(net_box["y2"], weight_box["y2"]),
            "text": "Net weight",
            "raw_text": clean_text(
                f"{left.get('raw_text') or left.get('text', '')} "
                f"{right.get('raw_text') or right.get('text', '')}"
            ),
            "det_conf": max(float(net_box.get("det_conf", 0.0)), float(weight_box.get("det_conf", 0.0))),
            "ocr_conf": float(np.mean([
                float(net_box.get("ocr_conf") or 0.0),
                float(weight_box.get("ocr_conf") or 0.0),
            ])),
            "mapped": True,
            "warning": False,
            "net_weight_merged": True,
        })
        merged["cx"] = (merged["x1"] + merged["x2"]) / 2.0
        merged["cy"] = (merged["y1"] + merged["y2"]) / 2.0
        replacements[id(left)] = merged
        changes.append({
            "before": [net_box.get("raw_text", ""), weight_box.get("raw_text", "")],
            "after": "Net weight",
            "x": merged["cx"], "y": merged["cy"],
        })

    if not changes:
        return detections, changes

    merged_detections = []
    for item in detections:
        item_id = id(item)
        if item_id in replacements:
            merged_detections.append(replacements[item_id])
        elif item_id not in used_ids:
            merged_detections.append(item)
    return sort_reading_order(merged_detections), changes


def sort_reading_order(detections):
    if not detections:
        return detections
    median_height = max(1, int(np.median([d["y2"] - d["y1"] for d in detections])))
    return sorted(detections, key=lambda d: (round(d["cy"] / median_height), d["x1"]))


def containing_table_key(detection, tables):
    matching_tables = []
    for table_index, table in enumerate(tables):
        if table["x1"] <= detection["cx"] <= table["x2"] and table["y1"] <= detection["cy"] <= table["y2"]:
            table_area = max(1, (table["x2"] - table["x1"]) * (table["y2"] - table["y1"]))
            matching_tables.append({
                "confidence": float(table.get("det_conf", 0.0)),
                "area": table_area, "index": table_index,
            })
    if not matching_tables:
        return "page"
    best_table = max(matching_tables, key=lambda item: (item["confidence"], item["area"]))
    return best_table["index"]


def resolve_duplicate_element_symbols(detections):
    raw_tables = [d for d in detections if d["class_name"] == "table"]
    raw_tables.sort(key=lambda d: d.get("det_conf", 0.0), reverse=True)
    tables = []
    for table in raw_tables:
        if not any(smaller_box_overlap_ratio(table, ext) >= 0.60 for ext in tables):
            tables.append(table)

    symbols_by_table = {}
    for d in detections:
        if d["class_name"] == "element_symbol" and d.get("text"):
            table_key = containing_table_key(d, tables)
            symbols_by_table.setdefault(table_key, []).append(d)

    replacements = 0
    for table_symbols in symbols_by_table.values():
        used_symbols = {d["text"] for d in table_symbols if d.get("text")}
        symbols_by_text = {}
        for d in table_symbols:
            symbols_by_text.setdefault(d["text"], []).append(d)

        for duplicated_symbol, duplicates in symbols_by_text.items():
            if len(duplicates) <= 1:
                continue
            duplicates.sort(key=lambda d: d.get("ocr_conf", 0.0), reverse=True)

            for duplicate in duplicates[1:]:
                replacement = None
                preferred_fallbacks = DUPLICATE_SYMBOL_FALLBACKS.get(duplicated_symbol, [])
                for fallback_text in preferred_fallbacks:
                    for alternative in duplicate.get("ocr_alternatives", []):
                        if (alternative["text"] == fallback_text and fallback_text not in used_symbols 
                                and alternative["confidence"] > SYMBOL_MIN_CONFIDENCE):
                            replacement = alternative
                            break
                    if replacement is not None:
                        break

                if replacement is None:
                    for alternative in duplicate.get("ocr_alternatives", []):
                        alternative_text = alternative["text"]
                        if (alternative_text != duplicated_symbol and alternative_text not in used_symbols 
                                and alternative["confidence"] > SYMBOL_MIN_CONFIDENCE):
                            replacement = alternative
                            break

                if replacement is None:
                    for fallback_text in preferred_fallbacks:
                        if fallback_text in used_symbols:
                            continue
                        inferred_conf = max(SYMBOL_MIN_CONFIDENCE + 0.001, float(duplicate.get("ocr_conf", SYMBOL_MIN_CONFIDENCE + 0.001)) * 0.85)
                        replacement = {
                            "text": fallback_text,
                            "raw": duplicate.get("raw_text", duplicated_symbol),
                            "confidence": inferred_conf,
                            "inferred": True,
                        }
                        break

                if replacement is None:
                    duplicate["warning"] = True
                    duplicate["duplicate_symbol"] = True
                    continue

                duplicate["duplicate_resolved_from"] = duplicate["text"]
                duplicate["text"] = replacement["text"]
                duplicate["raw_text"] = replacement["raw"]
                duplicate["ocr_conf"] = replacement["confidence"]
                duplicate["mapped"] = True
                duplicate["inferred_symbol"] = replacement.get("inferred", False)
                duplicate["warning"] = (replacement["confidence"] < OCR_MIN_CONFIDENCE or duplicate["inferred_symbol"])
                used_symbols.add(replacement["text"])
                replacements += 1

    return replacements


def smaller_box_overlap_ratio(first, second):
    intersection_x1 = max(first["x1"], second["x1"])
    intersection_y1 = max(first["y1"], second["y1"])
    intersection_x2 = min(first["x2"], second["x2"])
    intersection_y2 = min(first["y2"], second["y2"])
    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    if intersection_area <= 0:
        return 0.0

    first_area = max(1, (first["x2"] - first["x1"]) * (first["y2"] - first["y1"]))
    second_area = max(1, (second["x2"] - second["x1"]) * (second["y2"] - second["y1"]))

    return intersection_area / min(first_area, second_area)


def filter_non_table_overlaps(detections, maximum_overlap=NON_TABLE_MAX_OVERLAP):
    table_detections = [d for d in detections if d["class_name"] == "table"]
    non_table_detections = [d for d in detections if d["class_name"] != "table"]
    non_table_detections.sort(key=lambda d: d["det_conf"], reverse=True)

    kept = []
    for candidate in non_table_detections:
        conflicts = any(smaller_box_overlap_ratio(candidate, existing) > maximum_overlap for existing in kept)
        if not conflicts:
            kept.append(candidate)

    return table_detections + kept


def make_overlay_label(detection):
    class_name = detection["class_name"]
    if not detection.get("ocr_applied", False) or class_name in NO_OCR_CLASSES:
        short = CLASS_SHORT.get(class_name, "?")
        return f"{short}: {class_name}" if SHOW_CLASS_PREFIX else class_name

    text = detection.get("text", "") or "?"
    if len(text) > OVERLAY_MAX_TEXT_LENGTH:
        text = text[:OVERLAY_MAX_TEXT_LENGTH - 3] + "..."
    return text


def draw_transparent_label(image, label, x, y, colour):
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, OVERLAY_FONT_THICKNESS
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad = OVERLAY_TEXT_PADDING

    x1, y1 = max(0, x), max(0, y)
    x2 = min(image.shape[1] - 1, x1 + text_width + pad * 2)
    y2 = min(image.shape[0] - 1, y1 + text_height + baseline + pad * 2)

    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(overlay, OVERLAY_LABEL_ALPHA, image, 1.0 - OVERLAY_LABEL_ALPHA, 0, image)
    cv2.putText(image, label, (x1 + pad, y1 + text_height + pad), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return x1, y1, x2, y2


def rectangles_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def draw_overlay(image, detections, show_box_text=True):
    output = image.copy()
    occupied_labels = []

    for detection in detections:
        x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
        class_name = detection["class_name"]
        colour = CLASS_COLOURS.get(class_name, (120, 120, 120))
        cv2.rectangle(output, (x1, y1), (x2, y2), colour, OVERLAY_BOX_THICKNESS)

        if not SHOW_OVERLAY_TEXT or not show_box_text:
            continue

        label = make_overlay_label(detection)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label, font, OVERLAY_FONT_SCALE, OVERLAY_FONT_THICKNESS)
        label_width = text_width + OVERLAY_TEXT_PADDING * 2
        label_height = text_height + baseline + OVERLAY_TEXT_PADDING * 2

        candidates = [
            (x2 + 4, y1),
            (max(0, x1 - label_width - 4), y1),
            (x1, y1 - label_height - 2),
            (x1, y2 + 2),
        ]

        chosen = None
        for candidate_x, candidate_y in candidates:
            candidate_x, candidate_y = max(0, candidate_x), max(0, candidate_y)
            if candidate_x + label_width >= output.shape[1]:
                candidate_x = max(0, output.shape[1] - label_width - 1)
            if candidate_y + label_height >= output.shape[0]:
                candidate_y = max(0, output.shape[0] - label_height - 1)

            rect = (candidate_x, candidate_y, candidate_x + label_width, candidate_y + label_height)
            if not any(rectangles_overlap(rect, existing) for existing in occupied_labels):
                chosen = (candidate_x, candidate_y, rect)
                break

        if chosen is None:
            candidate_x, candidate_y = max(0, x2 + 3), max(0, y1)
            if candidate_x + label_width >= output.shape[1]:
                candidate_x = max(0, x1 - label_width - 3)
            rect = (candidate_x, candidate_y, candidate_x + label_width, candidate_y + label_height)
            chosen = (candidate_x, candidate_y, rect)

        label_x, label_y, rect = chosen
        draw_transparent_label(output, label, label_x, label_y, colour)
        occupied_labels.append(rect)

    return output


# ============================================================
# MULTI-TABLE STRUCTURED PARSER WITH BATCH / LOT SUPPORT
# ============================================================

# ============================================================
# MULTI-TABLE STRUCTURED PARSER WITH ENHANCED BATCH / LOT DETECTOR
# ============================================================

def is_batch_lot_table(tokens):
    """Checks all tokens in the header area for batch/lot key terms."""
    if not tokens:
        return False
    
    # Look at top-most tokens (header zone)
    min_y = min(t["cy"] for t in tokens)
    header_tokens = [t for t in tokens if abs(t["cy"] - min_y) < 60]
    
    combined_header_text = " ".join((t.get("text") or t.get("raw_text", "")).lower() for t in header_tokens)
    
    return any(k in combined_header_text for k in ["batch", "bat", "lot", "sample", "number", "no."])


def parse_batch_lot_table_tokens(tokens):
    if not tokens:
        return {}

    # 1. Group tokens into horizontal lines
    tokens_sorted = sorted(tokens, key=lambda t: t["cy"])
    lines = []
    current_line = []
    last_cy = -100

    for t in tokens_sorted:
        if not current_line or abs(t["cy"] - last_cy) < 25:
            current_line.append(t)
            last_cy = np.mean([item["cy"] for item in current_line])
        else:
            lines.append(sorted(current_line, key=lambda item: item["x1"]))
            current_line = [t]
            last_cy = t["cy"]
    if current_line:
        lines.append(sorted(current_line, key=lambda item: item["x1"]))

    if len(lines) < 2:
        return {}

    # 2. Merge multi-line header rows (e.g., "Batch" on top of "number")
    header_tokens = lines[0]
    data_start_idx = 1
    
    # If second row is also part of header (e.g., Net / Weight (t))
    if len(lines) > 2 and not any(re.search(r'\d{3,}', t.get("text", "")) for t in lines[1]):
        header_tokens.extend(lines[1])
        data_start_idx = 2

    # 3. Build Column Definitions from Headers
    columns = []
    # Cluster header tokens by X position
    header_tokens_sorted = sorted(header_tokens, key=lambda t: t["cx"])
    col_clusters = []
    
    for ht in header_tokens_sorted:
        ht_text = ht.get("text") or ht.get("raw_text", "")
        placed = False
        for cluster in col_clusters:
            if abs(cluster["cx"] - ht["cx"]) < 70:
                cluster["texts"].append(ht_text)
                cluster["cx"] = np.mean([cluster["cx"], ht["cx"]])
                placed = True
                break
        if not placed:
            col_clusters.append({"cx": ht["cx"], "texts": [ht_text]})

    for cluster in col_clusters:
        combined_text = " ".join(cluster["texts"])
        norm_el, is_el = normalise_element(combined_text)
        
        # Check if column is Batch/Lot
        is_batch_col = any(k in combined_text.lower() for k in ["batch", "bat", "lot", "number", "no.", "sample"])
        
        # Fallback element lookup in text (e.g. "Ni %" -> "Ni")
        if not is_el:
            for word in re.findall(r"[A-Za-z]+", combined_text):
                cand, valid = normalise_element(word)
                if valid:
                    norm_el, is_el = cand, True
                    break

        columns.append({
            "x_center": cluster["cx"],
            "text": combined_text,
            "is_batch": is_batch_col,
            "element": norm_el if is_el else None
        })

    # Ensure at least one Batch ID column exists (default to first column if undetected)
    if not any(c["is_batch"] for c in columns) and columns:
        columns[0]["is_batch"] = True

    # 4. Process Data Rows
    batch_tables = {}
    data_lines = lines[data_start_idx:]

    for line in data_lines:
        row_data = {}
        batch_id = None

        for t in line:
            t_text = t.get("text") or t.get("raw_text", "")
            if not t_text:
                continue

            # Assign token to nearest column by X position
            best_col = min(columns, key=lambda c: abs(c["x_center"] - t["cx"]))
            if abs(best_col["x_center"] - t["cx"]) < 120:
                if best_col["is_batch"]:
                    batch_id = t_text
                elif best_col["element"]:
                    row_data[best_col["element"]] = t_text

        if batch_id and row_data:
            batch_key = f"Batch {batch_id}"
            batch_rows = batch_tables.setdefault(batch_key, [])
            
            for el, val in row_data.items():
                batch_rows.append({
                    "element": el,
                    "orig_val": val, "val": val,
                    "orig_min": "-", "min": "-",
                    "orig_max": "-", "max": "-",
                    "orig_unit": "%", "unit": "%",
                    "edited": False
                })

    return batch_tables


def extract_chemical_tables_by_region(detections):
    if not detections:
        return {}

    table_boxes = [d for d in detections if d["class_name"] == "table"]
    table_boxes.sort(key=lambda t: (t["y1"], t["x1"]))

    tables_map = {}
    tokens = [d for d in detections if d.get("ocr_applied") and d.get("text")]

    for idx, tbl in enumerate(table_boxes, start=1):
        tbl_tokens = [
            t for t in tokens
            if tbl["x1"] <= t["cx"] <= tbl["x2"] and tbl["y1"] <= t["cy"] <= tbl["y2"]
        ]
        
        if is_batch_lot_table(tbl_tokens):
            batch_maps = parse_batch_lot_table_tokens(tbl_tokens)
            if batch_maps:
                for batch_name, rows in batch_maps.items():
                    tables_map[f"T{idx} - Batch: {batch_name}"] = rows
                continue

        parsed_rows = parse_single_table_tokens(tbl_tokens)
        if parsed_rows:
            tables_map[f"Table {idx}"] = parsed_rows

    bound_tokens = {
        id(t) for tbl in table_boxes
        for t in tokens if tbl["x1"] <= t["cx"] <= tbl["x2"] and tbl["y1"] <= t["cy"] <= tbl["y2"]
    }
    unbound_tokens = [t for t in tokens if id(t) not in bound_tokens]

    if is_batch_lot_table(unbound_tokens):
        batch_maps = parse_batch_lot_table_tokens(unbound_tokens)
        if batch_maps:
            for batch_name, rows in batch_maps.items():
                tables_map[f"Batch: {batch_name}"] = rows
    else:
        unbound_rows = parse_single_table_tokens(unbound_tokens)
        if unbound_rows:
            if not tables_map:
                tables_map["Table 1"] = unbound_rows
            else:
                tables_map["Unbound Elements"] = unbound_rows

    return tables_map


def parse_single_table_tokens(tokens):
    """
    Chemical-table parser using intelligent horizontal projection.

    Main rule:
        The Y-centre of every element_symbol defines its row.

    Row boundaries:
        Halfway between the Y-centres of neighbouring element symbols.

    Therefore a value can belong to only ONE chemical element row.

    Within each row:
        X position / Min-Max headers determine the column.

    If several detections compete:
        1. strongest horizontal alignment with element centre
        2. YOLO/OCR confidence
    """

    # ==========================================================
    # 1. GET CHEMICAL ELEMENT SYMBOLS
    # ==========================================================

    symbols = [
        t for t in tokens
        if t["class_name"] == "element_symbol"
        and t["text"] in VALID_ELEMENTS
    ]

    if not symbols:
        return []

    symbols.sort(key=lambda s: (s["cy"], s["cx"]))

    # ==========================================================
    # 2. REMOVE CLOSE DUPLICATE SYMBOLS
    # ==========================================================

    filtered_symbols = []

    for s in symbols:

        if filtered_symbols:

            prev = filtered_symbols[-1]

            dist = np.sqrt(
                (prev["cx"] - s["cx"]) ** 2 +
                (prev["cy"] - s["cy"]) ** 2
            )

            if prev["text"] == s["text"] and dist < 120:
                continue

        filtered_symbols.append(s)

    symbols = filtered_symbols

    # ==========================================================
    # 3. DETECTION GROUPS
    # ==========================================================

    values = [
        t for t in tokens
        if t["class_name"] == "element_value"
    ]

    units = [
        t for t in tokens
        if t["class_name"] == "unit"
    ]

    headers = [
        t for t in tokens
        if t["class_name"] == "headers"
    ]

    limits = [
        t for t in tokens
        if t["class_name"] == "limit_indicator"
    ]

    # ==========================================================
    # 4. FIND MIN / MAX HEADER CENTRES
    # ==========================================================

    min_hdr_center = None
    max_hdr_center = None

    for h in headers:

        htext = clean_text(
            h.get("raw_text") or h.get("text", "")
        ).lower()

        h_center = h["cx"]

        if "min" in htext:
            min_hdr_center = h_center

        elif "max" in htext:
            max_hdr_center = h_center

    # ==========================================================
    # 5. DETECTION CONFIDENCE
    # ==========================================================

    def detection_confidence(det):

        candidates = [
            det.get("ocr_confidence"),
            det.get("ocr_conf"),
            det.get("confidence"),
            det.get("conf"),
        ]

        for value in candidates:

            if value is None:
                continue

            try:
                value = float(value)

                if value > 1.0:
                    value /= 100.0

                return max(0.0, min(value, 1.0))

            except Exception:
                pass

        return 0.0

    # ==========================================================
    # 6. BUILD INTELLIGENT ROW BOUNDARIES
    # ==========================================================

    row_regions = []

    for i, sym in enumerate(symbols):

        current_y = float(sym["cy"])

        # ------------------------------------------------------
        # UPPER boundary
        # ------------------------------------------------------

        if i == 0:

            if len(symbols) > 1:

                next_y = float(symbols[i + 1]["cy"])
                gap = next_y - current_y

                upper = current_y - gap / 2.0

            else:

                upper = float(sym["y1"])

        else:

            previous_y = float(symbols[i - 1]["cy"])

            upper = (
                previous_y +
                current_y
            ) / 2.0

        # ------------------------------------------------------
        # LOWER boundary
        # ------------------------------------------------------

        if i == len(symbols) - 1:

            if len(symbols) > 1:

                previous_y = float(symbols[i - 1]["cy"])
                gap = current_y - previous_y

                lower = current_y + gap / 2.0

            else:

                lower = float(sym["y2"])

        else:

            next_y = float(symbols[i + 1]["cy"])

            lower = (
                current_y +
                next_y
            ) / 2.0

        row_regions.append({
            "symbol": sym,
            "upper": upper,
            "lower": lower,
        })

    # ==========================================================
    # 7. SCORE A DETECTION AGAINST A ROW
    # ==========================================================

    def row_score(det, sym, upper, lower):
        """
        Strong preference for detection centre lying on the same
        horizontal projection as the element-symbol centre.
        """

        sym_y = float(sym["cy"])
        det_y = float(det["cy"])

        dy = abs(det_y - sym_y)

        half_height = max(
            (lower - upper) / 2.0,
            1.0
        )

        alignment = max(
            0.0,
            1.0 - dy / half_height
        )

        conf = detection_confidence(det)

        # Geometry deliberately stronger than OCR confidence.
        return (
            0.80 * alignment +
            0.20 * conf
        )

    # ==========================================================
    # 8. ASSIGN EVERY VALUE TO EXACTLY ONE ELEMENT ROW
    # ==========================================================

    values_by_row = {
        i: []
        for i in range(len(row_regions))
    }

    for val in values:

        val_y = float(val["cy"])

        possible_rows = []

        for i, region in enumerate(row_regions):

            if region["upper"] <= val_y < region["lower"]:

                score = row_score(
                    val,
                    region["symbol"],
                    region["upper"],
                    region["lower"]
                )

                possible_rows.append(
                    (score, i)
                )

        # Normally only one row is possible because the regions
        # do not overlap. MAX score protects boundary cases.
        if possible_rows:

            possible_rows.sort(
                key=lambda item: item[0],
                reverse=True
            )

            best_score, best_row = possible_rows[0]

            value_copy = dict(val)
            value_copy["_row_score"] = best_score

            values_by_row[best_row].append(value_copy)

    # ==========================================================
    # 9. ASSIGN LIMITS TO EXACTLY ONE ROW
    # ==========================================================

    limits_by_row = {
        i: []
        for i in range(len(row_regions))
    }

    for lim in limits:

        lim_y = float(lim["cy"])

        possible_rows = []

        for i, region in enumerate(row_regions):

            if region["upper"] <= lim_y < region["lower"]:

                score = row_score(
                    lim,
                    region["symbol"],
                    region["upper"],
                    region["lower"]
                )

                possible_rows.append(
                    (score, i)
                )

        if possible_rows:

            possible_rows.sort(
                key=lambda item: item[0],
                reverse=True
            )

            _, best_row = possible_rows[0]

            limits_by_row[best_row].append(lim)

    # ==========================================================
    # 10. ASSIGN UNITS TO EXACTLY ONE ROW
    # ==========================================================

    units_by_row = {
        i: []
        for i in range(len(row_regions))
    }

    for unit in units:

        unit_y = float(unit["cy"])

        possible_rows = []

        for i, region in enumerate(row_regions):

            if region["upper"] <= unit_y < region["lower"]:

                score = row_score(
                    unit,
                    region["symbol"],
                    region["upper"],
                    region["lower"]
                )

                possible_rows.append(
                    (score, i)
                )

        if possible_rows:

            possible_rows.sort(
                key=lambda item: item[0],
                reverse=True
            )

            _, best_row = possible_rows[0]

            units_by_row[best_row].append(unit)

    # ==========================================================
    # 11. BUILD STRUCTURED CHEMICAL ROWS
    # ==========================================================

    rows = []

    for row_index, region in enumerate(row_regions):

        sym = region["symbol"]
        sym_text = sym["text"]

        line_values = values_by_row[row_index]

        # Values left -> right
        line_values.sort(
            key=lambda v: v["cx"]
        )

        row_limits = limits_by_row[row_index]

        row_units = units_by_row[row_index]

        val_text = "-"
        min_val = "-"
        max_val = "-"

        # ======================================================
        # 12. CLASSIFY VALUES INTO VALUE / MIN / MAX
        # ======================================================

        for val in line_values:

            curr_text = val["text"]
            val_cx = float(val["cx"])

            near_limit = None

            # --------------------------------------------------
            # Explicit MIN/MAX detection on SAME projected row
            # --------------------------------------------------

            if row_limits:

                matching_limits = sorted(
                    row_limits,
                    key=lambda lim:
                        abs(float(lim["cy"]) - float(val["cy"])) +
                        0.20 * abs(float(lim["cx"]) - val_cx)
                )

                for lim in matching_limits:

                    # Do not use a limit extremely far away
                    if abs(float(lim["cx"]) - val_cx) > 180:
                        continue

                    lim_txt = clean_text(
                        lim.get("text", "")
                    ).upper()

                    lim_raw = clean_text(
                        lim.get("raw_text", "")
                    ).upper()

                    combined = (
                        lim_txt + " " + lim_raw
                    )

                    if "MIN" in combined:
                        near_limit = "MIN"
                        break

                    if "MAX" in combined:
                        near_limit = "MAX"
                        break

            # --------------------------------------------------
            # Inline < / >
            # --------------------------------------------------

            has_inline_max = (
                "<=" in curr_text or
                "<" in curr_text
            )

            has_inline_min = (
                ">=" in curr_text or
                ">" in curr_text
            )

            # --------------------------------------------------
            # Header-column distance
            # --------------------------------------------------

            dist_to_min = (
                abs(val_cx - min_hdr_center)
                if min_hdr_center is not None
                else float("inf")
            )

            dist_to_max = (
                abs(val_cx - max_hdr_center)
                if max_hdr_center is not None
                else float("inf")
            )

            # --------------------------------------------------
            # HEADER POSITION HAS STRONG PRIORITY
            # --------------------------------------------------

            if (
                min_hdr_center is not None
                and dist_to_min < dist_to_max
                and dist_to_min < 250
            ):

                # If more than one detection competes for Min,
                # keep strongest horizontal projection.
                if min_val == "-":
                    min_val = curr_text

                else:

                    existing = next(
                        (
                            x for x in line_values
                            if x["text"] == min_val
                        ),
                        None
                    )

                    if (
                        existing is None or
                        val.get("_row_score", 0) >
                        existing.get("_row_score", 0)
                    ):
                        min_val = curr_text

            elif (
                max_hdr_center is not None
                and dist_to_max < dist_to_min
                and dist_to_max < 250
            ):

                if max_val == "-":
                    max_val = curr_text

                else:

                    existing = next(
                        (
                            x for x in line_values
                            if x["text"] == max_val
                        ),
                        None
                    )

                    if (
                        existing is None or
                        val.get("_row_score", 0) >
                        existing.get("_row_score", 0)
                    ):
                        max_val = curr_text

            elif near_limit == "MIN" or has_inline_min:

                min_val = curr_text

            elif near_limit == "MAX" or has_inline_max:

                max_val = curr_text

            else:

                # ------------------------------------------------
                # No explicit headers/limits
                # ------------------------------------------------

                if len(line_values) == 1:

                    val_text = curr_text

                elif len(line_values) >= 2:

                    # left = minimum
                    # right = maximum
                    position = line_values.index(val)

                    if position == 0:
                        min_val = curr_text

                    elif position == len(line_values) - 1:
                        max_val = curr_text

                    else:
                        val_text = curr_text

        # ======================================================
        # 13. UNIT FOR THIS EXACT ROW
        # ======================================================

        unit_text = "%"

        if row_units:

            reference_y = float(sym["cy"])

            best_unit = max(
                row_units,
                key=lambda u: (
                    -abs(float(u["cy"]) - reference_y),
                    detection_confidence(u)
                )
            )

            unit_text = best_unit["text"]

        else:

            # Header-based unit fallback
            for h in headers:

                if (
                    h["x1"] - 20
                    <= sym["cx"]
                    <= h["x2"] + 20
                    and h["cy"] < sym["cy"]
                ):

                    h_lower = clean_text(
                        h.get("raw_text")
                        or h.get("text", "")
                    ).lower()

                    if "ppm" in h_lower:
                        unit_text = "ppm"
                        break

                    elif "ppb" in h_lower:
                        unit_text = "ppb"
                        break

                    elif "%" in h_lower or "percent" in h_lower:
                        unit_text = "%"
                        break

        # ======================================================
        # 14. SAVE ROW
        # ======================================================

        rows.append({
            "element": sym_text,

            "orig_val": val_text,
            "val": val_text,

            "orig_min": min_val,
            "min": min_val,

            "orig_max": max_val,
            "max": max_val,

            "orig_unit": unit_text,
            "unit": unit_text,

            "edited": False
        })

    return rows

# ============================================================
# GUI CLASS WITH FULL INSPECTOR & MULTI-TABLE SUPPORT
# ============================================================

class OCRInspectionTool:

    def __init__(self, root, model, reader):
        self.root = root
        self.model = model
        self.reader = reader
        self.file_path = None
        self.current_image = None
        self.annotated_image = None
        self.detections = []
        self.tables_data = {}
        self.current_page = 0
        self.pdf_pages = 0
        self.tk_image = None
        self.zoom = 1.0
        self.min_zoom = 0.05
        self.max_zoom = 8.0
        self.busy = False

        root.title("Chemical Document Viewer - YOLO / YOLO + OCR")
        try:
            root.state("zoomed")
        except Exception:
            screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
            root.geometry(f"{int(screen_w * 0.95)}x{int(screen_h * 0.90)}")
        root.minsize(950, 600)

        self._build_gui()

    def _build_gui(self):
        toolbar = ttk.Frame(self.root, padding=5)
        toolbar.pack(fill=tk.X)

        self.open_button = ttk.Button(toolbar, text="Open PDF / image", command=self.upload_file)
        self.open_button.pack(side=tk.LEFT, padx=2)

        self.yolo_button = ttk.Button(
            toolbar, text="Run YOLO Only", command=lambda: self.start_inspection(use_ocr=False, show_box_text=True)
        )
        self.yolo_button.pack(side=tk.LEFT, padx=2)

        self.colour_button = ttk.Button(
            toolbar, text="YOLO Colours Only", command=lambda: self.start_inspection(use_ocr=False, show_box_text=False)
        )
        self.colour_button.pack(side=tk.LEFT, padx=2)

        self.run_button = ttk.Button(
            toolbar, text="Run YOLO + OCR", command=lambda: self.start_inspection(use_ocr=True, show_box_text=True)
        )
        self.run_button.pack(side=tk.LEFT, padx=2)

        self.advanced_button = ttk.Button(
            toolbar,
            text="Run Advanced OCR",
            command=lambda: self.start_inspection(use_ocr=True, show_box_text=True, advanced_ocr=True)
        )
        self.advanced_button.pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.previous_button = ttk.Button(toolbar, text="◀ Previous", command=lambda: self.change_page(-1))
        self.previous_button.pack(side=tk.LEFT, padx=2)

        self.next_button = ttk.Button(toolbar, text="Next ▶", command=lambda: self.change_page(1))
        self.next_button.pack(side=tk.LEFT, padx=2)

        self.page_label = ttk.Label(toolbar, text="Page 0 / 0")
        self.page_label.pack(side=tk.LEFT, padx=8)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(toolbar, text="Fit Page", command=self.fit_page).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Fit Width", command=self.fit_width).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="100%", command=self.reset_zoom).pack(side=tk.LEFT, padx=2)

        ttk.Button(toolbar, text="−", width=3, command=lambda: self.zoom_button(1 / 1.20)).pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar, text="+", width=3, command=lambda: self.zoom_button(1.20)).pack(side=tk.LEFT, padx=1)

        self.zoom_label = ttk.Label(toolbar, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=8)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.overlay_button = ttk.Button(toolbar, text="Hide Detection Overlay", command=self.toggle_overlay)
        self.overlay_button.pack(side=tk.LEFT, padx=2)

        self.status_label = ttk.Label(toolbar, text="Ready", foreground="#174ea6")
        self.status_label.pack(side=tk.RIGHT, padx=5)

        self.file_label = ttk.Label(self.root, text="No document loaded", padding=(8, 2, 8, 4))
        self.file_label.pack(fill=tk.X)

        self.main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        image_frame = ttk.Frame(self.main_pane)
        result_frame = ttk.Frame(self.main_pane, width=480)

        self.main_pane.add(image_frame, weight=5)
        self.main_pane.add(result_frame, weight=2)

        self.canvas = tk.Canvas(image_frame, bg="#383838", highlightthickness=0, cursor="fleur")
        x_scroll = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        y_scroll = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.canvas.yview)

        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)

        self.canvas.bind("<MouseWheel>", self.mouse_zoom)
        self.canvas.bind("<Shift-MouseWheel>", self.horizontal_scroll)
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.drag_pan)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.drag_pan)
        self.canvas.bind("<Double-Button-1>", lambda event: self.fit_page())

        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_raw = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.tab_raw, text="Raw Detections")

        self.tab_debug = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.tab_debug, text="Advanced OCR Debug")

        debug_toolbar = ttk.Frame(self.tab_debug)
        debug_toolbar.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(
            debug_toolbar,
            text="Advanced OCR processing log and full error traceback",
            font=("Segoe UI", 10, "bold")
        ).pack(side=tk.LEFT)
        ttk.Button(debug_toolbar, text="Copy Debug", command=self.copy_debug_output).pack(side=tk.RIGHT, padx=2)
        ttk.Button(debug_toolbar, text="Clear", command=self.clear_debug_output).pack(side=tk.RIGHT, padx=2)

        debug_container = ttk.Frame(self.tab_debug)
        debug_container.pack(fill=tk.BOTH, expand=True)
        self.debug_text = tk.Text(
            debug_container, wrap=tk.NONE, font=("Consolas", 9),
            background="#111827", foreground="#e5e7eb", insertbackground="white"
        )
        debug_y = ttk.Scrollbar(debug_container, orient=tk.VERTICAL, command=self.debug_text.yview)
        debug_x = ttk.Scrollbar(debug_container, orient=tk.HORIZONTAL, command=self.debug_text.xview)
        self.debug_text.configure(yscrollcommand=debug_y.set, xscrollcommand=debug_x.set)
        self.debug_text.grid(row=0, column=0, sticky="nsew")
        debug_y.grid(row=0, column=1, sticky="ns")
        debug_x.grid(row=1, column=0, sticky="ew")
        debug_container.rowconfigure(0, weight=1)
        debug_container.columnconfigure(0, weight=1)
        self.debug_text.insert(tk.END, "Press Run Advanced OCR to create a detailed processing log.\n")

        ttk.Label(self.tab_raw, text="YOLO Detections & Raw OCR Results", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(2, 5))

        legend_frame = ttk.LabelFrame(self.tab_raw, text="YOLO box colours", padding=5)
        legend_frame.pack(fill=tk.X, pady=(0, 7))

        self.colour_legend = tk.Text(legend_frame, height=5, wrap=tk.NONE, relief=tk.FLAT, borderwidth=0, background="#f7f7f7", cursor="arrow", font=("Segoe UI", 9))
        self.colour_legend.pack(fill=tk.X)

        legend_descriptions = {
            "element_symbol": "Element symbol", "element_value": "Element value", "unit": "Unit",
            "limit_indicator": "Limit indicator", "table": "Complete table region", "headers": "Header region",
        }

        for class_id, class_name in EXPECTED_CLASSES.items():
            colour = CLASS_COLOURS.get(class_name, (80, 80, 80))
            tag_name = f"legend_{class_name}"
            self.colour_legend.tag_configure(tag_name, foreground=rgb_to_hex(colour), font=("Segoe UI", 9, "bold"))
            description = legend_descriptions.get(class_name, class_name)
            self.colour_legend.insert(tk.END, f"■   {class_id}   {description}\n", tag_name)
        self.colour_legend.config(state=tk.DISABLED)

        timing_frame = ttk.LabelFrame(self.tab_raw, text="Processing time", padding=5)
        timing_frame.pack(fill=tk.X, pady=(0, 7))

        self.timing_text = tk.Text(timing_frame, height=5, wrap=tk.NONE, relief=tk.FLAT, borderwidth=0, background="#f7f7f7", cursor="arrow", font=("Consolas", 8))
        self.timing_text.pack(fill=tk.X)
        self.timing_text.insert(tk.END, "Run YOLO or YOLO + OCR to display timing.")
        self.timing_text.config(state=tk.DISABLED)

        columns = ("no", "class", "raw", "final", "det_conf", "ocr_conf", "warning")
        raw_container = ttk.Frame(self.tab_raw)
        raw_container.pack(fill=tk.BOTH, expand=True)

        self.table = ttk.Treeview(raw_container, columns=columns, show="headings", selectmode="extended")
        headings = {"no": "#", "class": "Class", "raw": "Raw OCR", "final": "Final", "det_conf": "YOLO", "ocr_conf": "OCR", "warning": "!"}
        widths = {"no": 35, "class": 100, "raw": 85, "final": 85, "det_conf": 45, "ocr_conf": 45, "warning": 25}

        for column in columns:
            self.table.heading(column, text=headings[column])
            self.table.column(column, width=widths[column], minwidth=25, anchor="w")

        raw_y = ttk.Scrollbar(raw_container, orient=tk.VERTICAL, command=self.table.yview)
        raw_x = ttk.Scrollbar(raw_container, orient=tk.HORIZONTAL, command=self.table.xview)

        self.table.configure(yscrollcommand=raw_y.set, xscrollcommand=raw_x.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        raw_y.grid(row=0, column=1, sticky="ns")
        raw_x.grid(row=1, column=0, sticky="ew")

        raw_container.rowconfigure(0, weight=1)
        raw_container.columnconfigure(0, weight=1)

        self.table.bind("<<TreeviewSelect>>", self.select_detection)
        self._setup_raw_table_copy()

        ttk.Label(self.tab_raw, text="Wheel: zoom   |   Drag: move page   |   Right-click table: copy data", foreground="#666666", padding=(0, 5)).pack(anchor="w")

        self._set_navigation_state()

    def _setup_raw_table_copy(self):
        self.raw_context_menu = tk.Menu(self.root, tearoff=0)
        self.raw_context_menu.add_command(label="Copy Selected Row(s)", command=lambda: self.copy_tree_rows(self.table))
        self.raw_context_menu.add_command(label="Copy Entire Table", command=lambda: self.copy_tree_all(self.table))
        self.table.bind("<Button-3>", lambda e: self._show_context_menu(e, self.table, self.raw_context_menu))
        self.table.bind("<Control-c>", lambda e: self.copy_tree_rows(self.table))

    def _append_debug_text(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.debug_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.debug_text.see(tk.END)

    def debug_log(self, message):
        """Thread-safe log writer used by Advanced OCR."""
        self.root.after(0, self._append_debug_text, str(message))

    def clear_debug_output(self):
        self.debug_text.delete("1.0", tk.END)

    def copy_debug_output(self):
        text = self.debug_text.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.set_status("Advanced OCR debug output copied.", "#137333")

    def update_chemical_tables_notebook(self):
        permanent_tabs = {str(self.tab_raw), str(self.tab_debug)}
        for tab_id in list(self.notebook.tabs()):
            if tab_id not in permanent_tabs:
                self.notebook.forget(tab_id)

        for tab_name, rows_data in self.tables_data.items():
            tab_frame = ttk.Frame(self.notebook, padding=5)
            self.notebook.add(tab_frame, text=tab_name)

            top_bar = ttk.Frame(tab_frame)
            top_bar.pack(fill=tk.X, pady=(2, 5))
            ttk.Label(top_bar, text=f"{tab_name} Structured Results", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
            ttk.Button(top_bar, text="Export CSV", command=lambda d=rows_data, name=tab_name: self.export_chem_csv(d, name)).pack(side=tk.RIGHT)

            chem_columns = ("element", "val", "min", "max", "unit")
            container = ttk.Frame(tab_frame)
            container.pack(fill=tk.BOTH, expand=True)

            tree = ttk.Treeview(container, columns=chem_columns, show="headings", selectmode="extended")
            tree.heading("element", text="Chemical Element")
            tree.heading("val", text="Value")
            tree.heading("min", text="Min")
            tree.heading("max", text="Max")
            tree.heading("unit", text="Unit")

            tree.column("element", width=110, anchor="w")
            tree.column("val", width=90, anchor="w")
            tree.column("min", width=90, anchor="w")
            tree.column("max", width=90, anchor="w")
            tree.column("unit", width=70, anchor="w")

            y_scroll = ttk.Scrollbar(container, orient=tk.VERTICAL, command=tree.yview)
            x_scroll = ttk.Scrollbar(container, orient=tk.HORIZONTAL, command=tree.xview)

            tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
            tree.grid(row=0, column=0, sticky="nsew")
            y_scroll.grid(row=0, column=1, sticky="ns")
            x_scroll.grid(row=1, column=0, sticky="ew")

            container.rowconfigure(0, weight=1)
            container.columnconfigure(0, weight=1)

            tree.tag_configure("edited_val", foreground="#d32f2f", font=("Segoe UI", 9, "bold"))

            for idx, r in enumerate(rows_data):
                item_id = str(idx)
                val_disp = f"{r['val']} [was: {r['orig_val']}]" if (r["edited"] and r["val"] != r["orig_val"]) else r["val"]
                min_disp = f"{r['min']} [was: {r['orig_min']}]" if (r["edited"] and r["min"] != r["orig_min"]) else r["min"]
                max_disp = f"{r['max']} [was: {r['orig_max']}]" if (r["edited"] and r["max"] != r["orig_max"]) else r["max"]

                tag = "edited_val" if r["edited"] else ""
                tree.insert("", tk.END, iid=item_id, values=(r["element"], val_disp, min_disp, max_disp, r["unit"]), tags=(tag,))

            tree.bind("<Double-1>", lambda event, t=tree, data=rows_data: self.on_cell_double_click(event, t, data))
            self._setup_chem_tree_copy(tree)

    def _setup_chem_tree_copy(self, tree):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Copy Selected Row(s)", command=lambda: self.copy_tree_rows(tree))
        menu.add_command(label="Copy Entire Table", command=lambda: self.copy_tree_all(tree))
        tree.bind("<Button-3>", lambda e: self._show_context_menu(e, tree, menu))
        tree.bind("<Control-c>", lambda e: self.copy_tree_rows(tree))

    def _show_context_menu(self, event, treeview, menu):
        row_id = treeview.identify_row(event.y)
        if row_id and row_id not in treeview.selection():
            treeview.selection_set(row_id)
        menu.post(event.x_root, event.y_root)

    def copy_tree_rows(self, treeview):
        selected = treeview.selection()
        if not selected:
            return
        lines = []
        for item in selected:
            vals = [str(v) for v in treeview.item(item, "values")]
            lines.append("\t".join(vals))
        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(lines))
        self.set_status("Selected rows copied to clipboard.", "#137333")

    def copy_tree_all(self, treeview):
        items = treeview.get_children()
        if not items:
            return
        lines = []
        for item in items:
            vals = [str(v) for v in treeview.item(item, "values")]
            lines.append("\t".join(vals))
        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(lines))
        self.set_status("Entire table copied to clipboard.", "#137333")

    def export_chem_csv(self, rows_data, tab_name):
        if not rows_data:
            messagebox.showwarning("No Data", f"No data available in {tab_name} to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title=f"Save {tab_name} As"
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Chemical Element", "Value", "Min", "Max", "Unit"])
                for row in rows_data:
                    writer.writerow([row["element"], row["val"], row["min"], row["max"], row["unit"]])
            messagebox.showinfo("Export Successful", f"Table exported to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def on_cell_double_click(self, event, tree, rows_data):
        region = tree.identify_region(event.x, event.y)
        if region != "cell":
            return

        column = tree.identify_column(event.x)
        row_id = tree.identify_row(event.y)
        if not row_id or column not in ("#2", "#3", "#4", "#5"):
            return

        col_index = int(column.replace("#", "")) - 1
        col_keys = ["element", "val", "min", "max", "unit"]
        target_key = col_keys[col_index]

        row_idx = int(row_id)
        current_val = rows_data[row_idx][target_key]

        x, y, w, h = tree.bbox(row_id, column)
        entry = ttk.Entry(tree)
        entry.place(x=x, y=y, width=w, height=h)
        entry.insert(0, current_val)
        entry.focus_set()

        def save_edit(_event=None):
            new_text = entry.get().strip()
            entry.destroy()
            if new_text and new_text != current_val:
                rows_data[row_idx][target_key] = new_text
                rows_data[row_idx]["edited"] = True

                r = rows_data[row_idx]
                val_disp = f"{r['val']} [was: {r['orig_val']}]" if (r["edited"] and r["val"] != r["orig_val"]) else r["val"]
                min_disp = f"{r['min']} [was: {r['orig_min']}]" if (r["edited"] and r["min"] != r["orig_min"]) else r["min"]
                max_disp = f"{r['max']} [was: {r['orig_max']}]" if (r["edited"] and r["max"] != r["orig_max"]) else r["max"]

                tree.item(row_id, values=(
                    r["element"],
                    val_disp,
                    min_disp,
                    max_disp,
                    r["unit"]
                ), tags=("edited_val",))

        entry.bind("<Return>", save_edit)
        entry.bind("<FocusOut>", lambda e: entry.destroy())
        entry.bind("<Escape>", lambda e: entry.destroy())

    def update_timing_display(self, timing_info):
        lines = [
            f"YOLO:         {format_duration(timing_info['yolo_seconds'])}",
            f"Whole OCR:   {format_duration(timing_info['ocr_total_seconds'])}",
            f"OCR work:    {format_duration(timing_info['ocr_recognition_seconds'])}",
            f"OCR load:    {format_duration(timing_info['ocr_load_seconds'])}",
            f"Total:       {format_duration(timing_info['total_seconds'])}",
        ]
        self.timing_text.config(state=tk.NORMAL)
        self.timing_text.delete("1.0", tk.END)
        self.timing_text.insert(tk.END, "\n".join(lines))
        self.timing_text.config(state=tk.DISABLED)

    def set_status(self, text, colour="#174ea6"):
        self.status_label.config(text=text, foreground=colour)

    def upload_file(self):
        path = filedialog.askopenfilename(
            title="Select chemical document",
            filetypes=[("PDF and images", "*.pdf *.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("PDF", "*.pdf"), ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if not path:
            return
        self.file_path = Path(path)
        self.current_page = 0
        try:
            if self.file_path.suffix.lower() == ".pdf":
                with pymupdf.open(str(self.file_path)) as doc:
                    self.pdf_pages = doc.page_count
            else:
                self.pdf_pages = 1
            self.load_current_page()
            self.file_label.config(text=str(self.file_path))
            self.set_status("Document loaded.")
        except Exception as error:
            self.file_path = None
            messagebox.showerror("Open failed", str(error))

    def load_current_page(self):
        if not self.file_path:
            return
        if self.file_path.suffix.lower() == ".pdf":
            self.current_image = pdf_page_to_rgb(self.file_path, self.current_page)
        else:
            self.current_image = load_image_rgb(self.file_path)

        self.annotated_image = None
        self.detections = []
        self.tables_data = {}
        self.clear_table()
        self.page_label.config(text=f"Page {self.current_page + 1} / {self.pdf_pages}")
        self._set_navigation_state()
        self.root.after(80, self.fit_page)

    def change_page(self, direction):
        if self.busy or not self.file_path:
            return
        new_page = self.current_page + direction
        if 0 <= new_page < self.pdf_pages:
            self.current_page = new_page
            try:
                self.load_current_page()
                self.set_status("Page loaded.")
            except Exception as error:
                messagebox.showerror("Page error", str(error))

    def _set_navigation_state(self):
        prev_st = tk.NORMAL if (self.file_path and self.current_page > 0 and not self.busy) else tk.DISABLED
        next_st = tk.NORMAL if (self.file_path and self.current_page < self.pdf_pages - 1 and not self.busy) else tk.DISABLED
        self.previous_button.config(state=prev_st)
        self.next_button.config(state=next_st)

    def toggle_overlay(self):
        global SHOW_OVERLAY
        SHOW_OVERLAY = not SHOW_OVERLAY
        self.overlay_button.config(text="Hide Detection Overlay" if SHOW_OVERLAY else "Show Detection Overlay")
        image = self.get_display_image()
        if image is not None:
            self.show_image(image, keep_view=True)

    def start_inspection(self, use_ocr, show_box_text=True, advanced_ocr=False):
        if self.current_image is None:
            messagebox.showwarning("No document", "Open a PDF or image first.")
            return
        if self.busy:
            return

        self.busy = True
        self.open_button.config(state=tk.DISABLED)
        self.run_button.config(state=tk.DISABLED)
        self.advanced_button.config(state=tk.DISABLED)
        self.yolo_button.config(state=tk.DISABLED)
        self.colour_button.config(state=tk.DISABLED)
        self._set_navigation_state()

        if advanced_ocr:
            self.clear_debug_output()
            self._append_debug_text("Advanced OCR started")
            self._append_debug_text(f"Image shape: {self.current_image.shape}")
            self._append_debug_text(f"Device: {DEVICE}; OCR reader loaded: {self.reader is not None}")
            self.notebook.select(self.tab_debug)

        mode_text = "YOLO + Advanced OCR" if advanced_ocr else ("YOLO + OCR" if use_ocr else ("YOLO only" if show_box_text else "YOLO colours only"))
        self.set_status(f"Running {mode_text}...")

        image = self.current_image.copy()
        threading.Thread(target=self._process_image, args=(image, use_ocr, show_box_text, advanced_ocr), daemon=True).start()

    def _process_image(self, image, use_ocr, show_box_text, advanced_ocr=False):
        stage = "initialisation"
        try:
            processing_started = time.perf_counter()
            yolo_started = time.perf_counter()

            if advanced_ocr:
                self.debug_log("Stage 1: running first YOLO pass")

            stage = "first YOLO prediction"
            first_pass_result = self.model.predict(
                source=image, imgsz=IMAGE_SIZE, conf=CONFIDENCE, iou=IOU_THRESHOLD,
                max_det=MAX_DETECTIONS, device=DEVICE, agnostic_nms=False, verbose=False
            )[0]

            stage = "reading first YOLO boxes"
            temp_detections = []
            if first_pass_result.boxes is not None:
                for box in first_pass_result.boxes:
                    class_id = int(box.cls[0].cpu().item())
                    coordinates = box.xyxy[0].cpu().numpy().round().astype(int)
                    x1, y1, x2, y2 = coordinates
                    temp_detections.append({
                        "class_id": class_id,
                        "class_name": str(self.model.names[class_id]),
                        "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
                    })

            header_angle = estimate_angle_from_headers(temp_detections)
            if advanced_ocr:
                self.debug_log(f"First YOLO pass: {len(temp_detections)} boxes; estimated angle: {header_angle:.3f}")

            if abs(header_angle) > 0.3:
                stage = "rotating image"
                image, _ = rotate_image_by_angle(image, header_angle)
                self.current_image = image.copy()

                if advanced_ocr:
                    self.debug_log("Stage 2: rerunning YOLO after rotation")
                stage = "second YOLO prediction"
                result = self.model.predict(
                    source=image, imgsz=IMAGE_SIZE, conf=CONFIDENCE, iou=IOU_THRESHOLD,
                    max_det=MAX_DETECTIONS, device=DEVICE, agnostic_nms=False, verbose=False
                )[0]
            else:
                result = first_pass_result

            yolo_seconds = time.perf_counter() - yolo_started
            ocr_load_seconds = 0.0
            ocr_category_times = {cn: {"count": 0, "total_seconds": 0.0} for cn in OCR_CLASSES}
            detections = []

            stage = "building final detections"
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].cpu().item())
                    coordinates = box.xyxy[0].cpu().numpy().round().astype(int)
                    x1, y1, x2, y2 = coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

                    detections.append({
                        "class_id": class_id,
                        "class_name": str(self.model.names[class_id]),
                        "det_conf": float(box.conf[0].cpu().item()),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
                    })

            detections_before_filter = len(detections)
            detections = filter_non_table_overlaps(detections)
            removed_overlaps = detections_before_filter - len(detections)
            detections = sort_reading_order(detections)
            if advanced_ocr:
                class_counts = {}
                for item in detections:
                    class_counts[item["class_name"]] = class_counts.get(item["class_name"], 0) + 1
                self.debug_log(f"Filtered detections: {len(detections)}; classes: {class_counts}")

            ocr_detections = [d for d in detections if use_ocr and d["class_name"] in OCR_CLASSES]
            if advanced_ocr:
                # Headers must be read first so Batch/Lot columns can be identified
                # even when the YOLO table rectangle is smaller than the visible table.
                ocr_detections.sort(key=lambda d: (d["class_name"] != "headers", d["cy"], d["x1"]))
            ocr_total = len(ocr_detections)

            if use_ocr and ocr_total > 0 and self.reader is None:
                stage = "loading EasyOCR"
                self.root.after(0, self.set_status, "Loading EasyOCR...")
                ocr_load_started = time.perf_counter()
                self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
                ocr_load_seconds = time.perf_counter() - ocr_load_started
                if advanced_ocr:
                    self.debug_log(f"EasyOCR loaded in {ocr_load_seconds:.3f} seconds")

            for index, detection in enumerate(ocr_detections, start=1):
                stage = f"OCR detection {index}/{ocr_total}"
                if index == 1 or index % 10 == 0 or index == ocr_total:
                    label = "Advanced OCR" if advanced_ocr else "OCR"
                    self.root.after(0, self.set_status, f"{label} {index} / {ocr_total}")

                ocr_item_started = time.perf_counter()
                if advanced_ocr:
                    recognised_headers = [
                        d for d in detections
                        if d["class_name"] == "headers" and d.get("ocr_applied", False)
                    ]
                    batch_identifier = detection_is_under_batch_header(detection, recognised_headers)
                    self.debug_log(
                        f"Box {index}/{ocr_total}: class={detection['class_name']}, "
                        f"xyxy=({detection['x1']},{detection['y1']},{detection['x2']},{detection['y2']}), "
                        f"YOLO={detection['det_conf']:.3f}, batch_identifier={batch_identifier}"
                    )
                    try:
                        final_text, ocr_conf, mapped, raw_text = read_crop_text_advanced(
                            self.reader, image, detection, batch_identifier=batch_identifier
                        )
                    except Exception as item_error:
                        item_traceback = traceback.format_exc()
                        self.debug_log(
                            f"ERROR inside box {index}: {type(item_error).__name__}: {item_error}\n"
                            f"{item_traceback}"
                        )
                        # Continue with the established fast OCR for this box so
                        # one advanced failure does not cancel the whole page.
                        final_text, ocr_conf, mapped, raw_text = read_crop_text(
                            self.reader, image, detection
                        )
                        detection["advanced_fallback"] = True
                    detection["advanced_ocr"] = True
                    detection["batch_identifier"] = batch_identifier
                else:
                    final_text, ocr_conf, mapped, raw_text = read_crop_text(self.reader, image, detection)
                ocr_item_seconds = time.perf_counter() - ocr_item_started

                detection["ocr_duration_seconds"] = ocr_item_seconds
                category_timing = ocr_category_times[detection["class_name"]]
                category_timing["count"] += 1
                category_timing["total_seconds"] += ocr_item_seconds

                detection["text"] = final_text
                detection["raw_text"] = raw_text
                detection["ocr_conf"] = ocr_conf
                detection["mapped"] = mapped
                detection["ocr_applied"] = True
                detection["warning"] = (not final_text or ocr_conf < OCR_MIN_CONFIDENCE or not mapped)
                if advanced_ocr:
                    self.debug_log(
                        f"Result {index}: raw={raw_text!r}, final={final_text!r}, "
                        f"OCR={ocr_conf:.3f}, mapped={mapped}, "
                        f"fallback={detection.get('advanced_fallback', False)}"
                    )

            net_weight_merge_corrections = []
            if use_ocr:
                stage = "merging split Net and Weight element symbols"
                detections, net_weight_merge_corrections = merge_split_net_weight_element_symbols(detections)
                if advanced_ocr:
                    for correction in net_weight_merge_corrections:
                        self.debug_log(
                            f"Net/Weight box merge at ({correction['x']:.1f}, {correction['y']:.1f}): "
                            f"{correction['before']!r} -> {correction['after']!r}"
                        )

            header_merge_corrections = []
            if use_ocr:
                stage = "merging multiline semantic headers"
                header_merge_corrections = merge_stacked_batch_lot_headers(detections)
                if advanced_ocr:
                    for correction in header_merge_corrections:
                        self.debug_log(
                            f"Multiline header merge at ({correction['x']:.1f}, {correction['y']:.1f}): "
                            f"{correction['before']!r} -> {correction['after']!r}"
                        )

            column_decimal_corrections = []
            if advanced_ocr:
                stage = "checking decimal consistency by column"
                column_decimal_corrections = apply_advanced_column_decimal_consistency(detections)
                for correction in column_decimal_corrections:
                    self.debug_log(
                        f"Column decimal correction at ({correction['x']:.1f}, {correction['y']:.1f}): "
                        f"{correction['before']!r} -> {correction['after']!r} "
                        f"using {correction['neighbour_votes']} neighbouring decimal values"
                    )

            duplicate_symbol_replacements = 0
            if use_ocr:
                stage = "resolving duplicate element symbols"
                duplicate_symbol_replacements = resolve_duplicate_element_symbols(detections)

            for detection in detections:
                if not use_ocr or detection["class_name"] in NO_OCR_CLASSES:
                    detection["text"] = ""
                    detection["raw_text"] = ""
                    detection["ocr_conf"] = None
                    detection["mapped"] = True
                    detection["warning"] = False
                    detection["ocr_applied"] = False
                detection["removed_overlaps"] = removed_overlaps
                detection["duplicate_symbol_replacements"] = duplicate_symbol_replacements
                detection["column_decimal_corrections"] = len(column_decimal_corrections)
                detection["multiline_header_corrections"] = len(header_merge_corrections)
                detection["net_weight_box_merges"] = len(net_weight_merge_corrections)

            stage = "drawing overlay"
            overlay = draw_overlay(image, detections, show_box_text=show_box_text)
            ocr_recognition_seconds = sum(c["total_seconds"] for c in ocr_category_times.values())
            ocr_total_seconds = ocr_load_seconds + ocr_recognition_seconds
            total_seconds = time.perf_counter() - processing_started

            timing_info = {
                "yolo_seconds": yolo_seconds,
                "ocr_recognition_seconds": ocr_recognition_seconds,
                "ocr_load_seconds": ocr_load_seconds,
                "ocr_total_seconds": ocr_total_seconds,
                "total_seconds": total_seconds,
                "ocr_category_times": ocr_category_times,
            }

            timing_info["advanced_ocr"] = advanced_ocr
            if advanced_ocr:
                self.debug_log(f"Advanced OCR recognition finished in {total_seconds:.3f} seconds")
            self.root.after(0, self._inspection_complete_safe, detections, overlay, use_ocr, show_box_text, timing_info)
        except Exception as error:
            full_traceback = traceback.format_exc()
            if advanced_ocr:
                self.debug_log(
                    f"FATAL ERROR at stage: {stage}\n"
                    f"{type(error).__name__}: {error}\n{full_traceback}"
                )
            self.root.after(0, self._inspection_failed, str(error), full_traceback, advanced_ocr, stage)

    def _inspection_complete_safe(self, detections, overlay, use_ocr, show_box_text, timing_info):
        try:
            self._inspection_complete(detections, overlay, use_ocr, show_box_text, timing_info)
        except Exception as completion_error:
            completion_traceback = traceback.format_exc()
            advanced_ocr = bool(timing_info.get("advanced_ocr", False))
            if advanced_ocr:
                self._append_debug_text(
                    f"GUI COMPLETION ERROR: {type(completion_error).__name__}: "
                    f"{completion_error}\n{completion_traceback}"
                )
                self.notebook.select(self.tab_debug)
            self._inspection_failed(
                str(completion_error), completion_traceback,
                advanced_ocr, "displaying OCR results"
            )

    def _inspection_complete(self, detections, overlay, use_ocr, show_box_text, timing_info):
        self.detections = detections
        self.annotated_image = overlay
        self.update_timing_display(timing_info)
        self.clear_table()

        for index, detection in enumerate(detections, start=1):
            is_structural = detection["class_name"] in STRUCTURAL_CLASSES
            ocr_applied = detection.get("ocr_applied", False)
            warning = "⚠" if detection["warning"] else ""
            raw_display = "—" if not ocr_applied else detection.get("raw_text", "")
            final_display = detection.get("text", "") if ocr_applied else ("YOLO region" if is_structural else "YOLO box")
            ocr_display = "—" if not ocr_applied else f'{detection.get("ocr_conf", 0):.0%}'

            self.table.insert(
                "", tk.END, iid=str(index - 1),
                values=(index, detection["class_name"], raw_display, final_display, f'{detection["det_conf"]:.0%}', ocr_display, warning)
            )

        if use_ocr:
            try:
                if timing_info.get("advanced_ocr", False):
                    self._append_debug_text("Stage 3: building structured chemical tables")
                self.tables_data = extract_chemical_tables_by_region(detections)
                self.update_chemical_tables_notebook()
                if timing_info.get("advanced_ocr", False):
                    self._append_debug_text(f"Structured tables created: {len(self.tables_data)}")
            except Exception as table_error:
                table_traceback = traceback.format_exc()
                self.tables_data = {}
                self._append_debug_text(
                    f"STRUCTURED TABLE ERROR: {type(table_error).__name__}: {table_error}\n{table_traceback}"
                )
                self.notebook.select(self.tab_debug)

        self.show_image(self.get_display_image())
        self.busy = False
        self.open_button.config(state=tk.NORMAL)
        self.run_button.config(state=tk.NORMAL)
        self.advanced_button.config(state=tk.NORMAL)
        self.yolo_button.config(state=tk.NORMAL)
        self.colour_button.config(state=tk.NORMAL)
        self._set_navigation_state()

        warnings = sum(bool(d["warning"]) for d in detections if d["class_name"] in OCR_CLASSES)
        recognised = sum(bool(d.get("text")) for d in detections if d["class_name"] in OCR_CLASSES)
        ocr_total = sum(d["class_name"] in OCR_CLASSES for d in detections)
        structural_total = sum(d["class_name"] in STRUCTURAL_CLASSES for d in detections)
        removed_overlaps = detections[0].get("removed_overlaps", 0) if detections else 0
        duplicate_symbol_replacements = detections[0].get("duplicate_symbol_replacements", 0) if detections else 0

        if use_ocr:
            mode_name = "Advanced OCR" if timing_info.get("advanced_ocr", False) else "Fast OCR"
            status_text = f"{mode_name} | {len(detections)} boxes | {recognised}/{ocr_total} OCR | {structural_total} regions | {warnings} warnings | {duplicate_symbol_replacements} symbols corrected | {removed_overlaps} overlaps removed"
        elif show_box_text:
            status_text = f"YOLO only | {len(detections)} boxes | {structural_total} regions | {removed_overlaps} overlaps removed"
        else:
            status_text = f"YOLO colours only | {len(detections)} boxes | {structural_total} regions | {removed_overlaps} overlaps removed"

        self.set_status(status_text, "#137333")

    def _inspection_failed(self, error, full_traceback="", advanced_ocr=False, stage="unknown"):
        self.busy = False
        self.open_button.config(state=tk.NORMAL)
        self.run_button.config(state=tk.NORMAL)
        self.advanced_button.config(state=tk.NORMAL)
        self.yolo_button.config(state=tk.NORMAL)
        self.colour_button.config(state=tk.NORMAL)
        self._set_navigation_state()
        self.set_status("Processing failed.", "#b3261e")
        if advanced_ocr:
            self._append_debug_text(f"FAILED STAGE: {stage}")
            if full_traceback:
                self._append_debug_text(full_traceback)
            self.notebook.select(self.tab_debug)
            messagebox.showerror(
                "Advanced OCR error",
                f"{error}\n\nFull details are visible in the Advanced OCR Debug tab."
            )
        else:
            messagebox.showerror("YOLO / OCR error", error)

    def clear_table(self):
        for item in self.table.get_children():
            self.table.delete(item)

    def get_display_image(self):
        if SHOW_OVERLAY and self.annotated_image is not None:
            return self.annotated_image
        return self.current_image

    def show_image(self, image, keep_view=False):
        if image is None:
            return
        old_x, old_y = self.canvas.xview(), self.canvas.yview()
        height, width = image.shape[:2]
        shown_width = max(1, int(width * self.zoom))
        shown_height = max(1, int(height * self.zoom))

        interpolation = cv2.INTER_AREA if self.zoom < 1.0 else cv2.INTER_CUBIC
        shown = cv2.resize(image, (shown_width, shown_height), interpolation=interpolation)

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(shown))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.configure(scrollregion=(0, 0, shown_width, shown_height))
        self.zoom_label.config(text=f"{self.zoom * 100:.0f}%")

        if keep_view:
            if old_x:
                self.canvas.xview_moveto(old_x[0])
            if old_y:
                self.canvas.yview_moveto(old_y[0])

    def fit_page(self):
        image = self.get_display_image()
        if image is None:
            return
        self.root.update_idletasks()
        available_width = max(100, self.canvas.winfo_width() - 20)
        available_height = max(100, self.canvas.winfo_height() - 20)
        image_height, image_width = image.shape[:2]

        self.zoom = min(available_width / image_width, available_height / image_height)
        self.zoom = max(self.min_zoom, min(self.zoom, self.max_zoom))
        self.show_image(image)
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def fit_width(self):
        image = self.get_display_image()
        if image is None:
            return
        self.root.update_idletasks()
        available_width = max(100, self.canvas.winfo_width() - 20)
        image_width = image.shape[1]

        self.zoom = available_width / image_width
        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * 1.2))
        self.show_image(image)
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def reset_zoom(self):
        image = self.get_display_image()
        if image is None:
            return
        self.zoom = 1.0
        self.show_image(image)

    def zoom_button(self, factor):
        image = self.get_display_image()
        if image is None:
            return
        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * factor))
        self.show_image(image, keep_view=True)

    def mouse_zoom(self, event):
        image = self.get_display_image()
        if image is None:
            return "break"
        factor = 1.20 if event.delta > 0 else (1 / 1.20)
        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * factor))
        self.show_image(image)
        return "break"

    def horizontal_scroll(self, event):
        direction = -3 if event.delta > 0 else 3
        self.canvas.xview_scroll(direction, "units")
        return "break"

    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def drag_pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def select_detection(self, _event=None):
        selection = self.table.selection()
        if not selection or not self.detections:
            return

        index = int(selection[0])
        detection = self.detections[index]

        if self.zoom < 0.65:
            self.zoom = 0.90
            self.show_image(self.get_display_image())

        target_x, target_y = detection["cx"] * self.zoom, detection["cy"] * self.zoom
        bbox = self.canvas.bbox("all")
        if not bbox:
            return

        total_width, total_height = max(1, bbox[2]), max(1, bbox[3])
        view_width, view_height = self.canvas.winfo_width(), self.canvas.winfo_height()

        x_fraction = (target_x - view_width / 2) / total_width
        y_fraction = (target_y - view_height / 2) / total_height

        self.canvas.xview_moveto(max(0, min(1, x_fraction)))
        self.canvas.yview_moveto(max(0, min(1, y_fraction)))


# ============================================================
# VERIFY MODEL & MAIN ENTRY
# ============================================================

def verify_model_classes(model):
    actual = {int(key): str(value) for key, value in model.names.items()}
    if actual != EXPECTED_CLASSES:
        raise ValueError(
            "Loaded YOLO model does not match expected classes.\n\n"
            f"EXPECTED:\n{EXPECTED_CLASSES}\n\nACTUAL:\n{actual}"
        )


def main():
    print("\n" + "=" * 70 + "\nLOADING YOLO MODEL\n" + "=" * 70)
    print("Model:", MODEL_PATH)
    print("Device:", DEVICE)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO model not found:\n{MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    verify_model_classes(model)

    reader = None
    root = tk.Tk()
    app = OCRInspectionTool(root, model, reader)
    root.mainloop()


if __name__ == "__main__":
    main()
