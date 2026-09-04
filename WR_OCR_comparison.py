# ============================================================
# CHEMISTRY DOCUMENT DETECTION + OCR INSPECTION TOOL
# UPDATED FOR 6-CLASS MODEL
#
# MODEL:
# D:\Kaggle\WRtools_003\best.pt
#
# CLASSES:
# 0: element_symbol
# 1: element_value
# 2: unit
# 3: limit_indicator
# 4: value_range
# 5: sign
#
# OCR:
# EasyOCR
# padded crops
# multi-pass preprocessing
# class-specific allowlists
# chemistry-aware cleanup
# ============================================================

import os
import re
import fitz
import cv2
import numpy as np
import tkinter as tk

from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

import easyocr
import torch


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"D:\Kaggle\WRtools_003\best.pt"


# ------------------------------------------------------------
# YOLO SETTINGS
# ------------------------------------------------------------

IMAGE_SIZE = 1280

# Low threshold because this is an inspection tool.
# We want to see weaker detections too.
CONFIDENCE = 0.03

IOU_THRESHOLD = 0.45

MAX_DETECTIONS = 2000

PDF_DPI = 300

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# OCR SETTINGS
# ------------------------------------------------------------

OCR_SCALE = 5.0

# Relative padding around YOLO boxes
OCR_PAD_X = 0.15
OCR_PAD_Y = 0.25

# Minimum source-image padding
OCR_MIN_PAD_X = 4
OCR_MIN_PAD_Y = 3

# Ignore extremely weak OCR candidates
OCR_MIN_CONFIDENCE = 0.05


# ============================================================
# EXPECTED MODEL CLASSES
# ============================================================

EXPECTED_CLASSES = [
    "element_symbol",
    "element_value",
    "unit",
    "limit_indicator",
    "value_range",
    "sign",
]


# ============================================================
# OCR ALLOWLISTS
# ============================================================

OCR_ALLOWLISTS = {

    # Chemical symbols: Cr, Ni, Fe, Mo etc.
    "element_symbol":
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",

    # Numeric value only
    "element_value":
        "0123456789.,+-",

    # %, ppm, wt%, etc.
    "unit":
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz%/.",

    # MAX, MIN, Maximum, Minimum etc.
    "limit_indicator":
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",

    # 18-20, 0.05-0.10, 18 to 20 etc.
    "value_range":
        "0123456789.,+-–—",

    # < > <= >= ≤ ≥
    "sign":
        "<>=≤≥",

}


# ============================================================
# VALID CHEMICAL ELEMENT SYMBOLS
# ============================================================

CHEMICAL_ELEMENTS = {
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
    "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
    "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce",
    "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
    "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu"
}


# ============================================================
# LOAD YOLO
# ============================================================

print("=" * 70)
print("LOADING YOLO MODEL")
print("=" * 70)

print("Model:")
print(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found:\n{MODEL_PATH}"
    )

model = YOLO(MODEL_PATH)

print("\nYOLO loaded successfully.")

print("\nModel classes:")
print("-" * 70)

for class_id, class_name in model.names.items():
    print(f"{class_id}: {class_name}")

print("-" * 70)


# ============================================================
# VERIFY CLASS STRUCTURE
# ============================================================

actual_classes = [
    model.names[i]
    for i in sorted(model.names.keys())
]

if actual_classes != EXPECTED_CLASSES:

    print("\nWARNING:")
    print("Model classes do not exactly match expected classes.")

    print("\nExpected:")
    for i, name in enumerate(EXPECTED_CLASSES):
        print(f"  {i}: {name}")

    print("\nModel:")
    for i, name in model.names.items():
        print(f"  {i}: {name}")

else:
    print("\n✓ Model class structure is correct.")


# ============================================================
# LOAD EASYOCR
# ============================================================

print("\n" + "=" * 70)
print("LOADING EASYOCR")
print("=" * 70)

OCR_GPU = torch.cuda.is_available()

print("CUDA available:", torch.cuda.is_available())
print("EasyOCR GPU    :", OCR_GPU)

try:

    ocr_reader = easyocr.Reader(
        ["en"],
        gpu=OCR_GPU
    )

except Exception as error:

    print("\nEasyOCR GPU initialization failed.")
    print("Reason:", error)
    print("Falling back to CPU...")

    ocr_reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

print("EasyOCR loaded successfully.")


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image_rgb(path):

    image = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            f"Could not open image:\n{path}"
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# PDF PAGE -> RGB
# ============================================================

def pdf_page_to_rgb(
    pdf_path,
    page_number=0,
    dpi=PDF_DPI
):

    document = fitz.open(pdf_path)

    try:

        if len(document) == 0:
            raise ValueError(
                "PDF contains no pages."
            )

        page_number = max(
            0,
            min(
                page_number,
                len(document) - 1
            )
        )

        page = document[page_number]

        zoom = dpi / 72.0

        matrix = fitz.Matrix(
            zoom,
            zoom
        )

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        image = np.frombuffer(
            pix.samples,
            dtype=np.uint8
        )

        image = image.reshape(
            pix.height,
            pix.width,
            pix.n
        )

        if pix.n == 4:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2RGB
            )

        return image

    finally:
        document.close()


# ============================================================
# BASIC OCR TEXT CLEANUP
# ============================================================

def clean_ocr_text(text):

    text = str(text).strip()

    text = text.replace(
        "\n",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LEVENSHTEIN
# ============================================================

def levenshtein(a, b):

    a = str(a)
    b = str(b)

    if len(a) < len(b):
        return levenshtein(
            b,
            a
        )

    if len(b) == 0:
        return len(a)

    previous_row = list(
        range(
            len(b) + 1
        )
    )

    for i, char_a in enumerate(
        a,
        start=1
    ):

        current_row = [i]

        for j, char_b in enumerate(
            b,
            start=1
        ):

            insertions = previous_row[j] + 1

            deletions = current_row[j - 1] + 1

            substitutions = (
                previous_row[j - 1]
                + (char_a != char_b)
            )

            current_row.append(
                min(
                    insertions,
                    deletions,
                    substitutions
                )
            )

        previous_row = current_row

    return previous_row[-1]


# ============================================================
# ELEMENT SYMBOL NORMALIZATION
# ============================================================

def normalize_element_symbol(text):

    text = clean_ocr_text(text)

    candidate = re.sub(
        r"[^A-Za-z]",
        "",
        text
    )

    if not candidate:
        return text

    candidate = (
        candidate[0].upper()
        + candidate[1:].lower()
    )

    if candidate in CHEMICAL_ELEMENTS:
        return candidate

    # Do not aggressively correct long text
    if len(candidate) > 3:
        return candidate

    closest = None
    closest_distance = 999

    for element in CHEMICAL_ELEMENTS:

        distance = levenshtein(
            candidate.lower(),
            element.lower()
        )

        if distance < closest_distance:

            closest = element
            closest_distance = distance

    if closest_distance <= 1:
        return closest

    return candidate


# ============================================================
# ELEMENT VALUE NORMALIZATION
# ============================================================

def normalize_numeric_text(text):

    text = clean_ocr_text(text)

    text = text.replace(
        " ",
        ""
    )

    replacements = {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",

        "−": "-",
        "–": "-",
        "—": "-",

        ",": ".",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new
        )

    # Element value should remain numeric
    text = re.sub(
        r"[^0-9.+\-]",
        "",
        text
    )

    # Remove duplicate decimal points
    if text.count(".") > 1:

        first = text.find(".")

        text = (
            text[:first + 1]
            + text[first + 1:].replace(
                ".",
                ""
            )
        )

    return text


# ============================================================
# RANGE NORMALIZATION
# ============================================================

def normalize_range_text(text):

    text = clean_ocr_text(text)

    text = text.replace(
        " ",
        ""
    )

    replacements = {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",

        "−": "-",
        "–": "-",
        "—": "-",

        ",": ".",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new
        )

    # Keep range-related characters
    text = re.sub(
        r"[^0-9.+\-]",
        "",
        text
    )

    return text


# ============================================================
# UNIT NORMALIZATION
# ============================================================

def normalize_unit_text(text):

    original = clean_ocr_text(text)

    normalized = (
        original
        .replace(" ", "")
        .lower()
    )

    replacements = {
        "pprn": "ppm",
        "ppnn": "ppm",
        "ppni": "ppm",
        "pprm": "ppm",
        "ppin": "ppm",
        "wt.%": "wt%",
        "wt %": "wt%",
    }

    if normalized in replacements:
        return replacements[normalized]

    if normalized == "ppm":
        return "ppm"

    if normalized == "ppb":
        return "ppb"

    if normalized in {
        "%",
        "wt%",
        "wt.%",
    }:
        return normalized

    if "%" in normalized:
        return normalized

    return original


# ============================================================
# LIMIT INDICATOR NORMALIZATION
# ============================================================

def normalize_limit_indicator(text):

    original = clean_ocr_text(text)

    cleaned = re.sub(
        r"[^A-Za-z]",
        "",
        original
    ).lower()

    if not cleaned:
        return original

    max_variants = {
        "max",
        "maximum",
        "maxima",
        "mx",
        "ma",
    }

    min_variants = {
        "min",
        "minimum",
        "mn",
        "mi",
    }

    if cleaned in max_variants:
        return "MAX"

    if cleaned in min_variants:
        return "MIN"

    # Fuzzy correction
    if levenshtein(
        cleaned,
        "max"
    ) <= 1:
        return "MAX"

    if levenshtein(
        cleaned,
        "min"
    ) <= 1:
        return "MIN"

    if levenshtein(
        cleaned,
        "maximum"
    ) <= 2:
        return "MAX"

    if levenshtein(
        cleaned,
        "minimum"
    ) <= 2:
        return "MIN"

    return original.upper()


# ============================================================
# SIGN NORMALIZATION
# ============================================================

def normalize_sign_text(text):

    text = clean_ocr_text(text)

    text = (
        text
        .replace(" ", "")
        .replace("≤", "<=")
        .replace("≥", ">=")
    )

    # Common OCR confusions
    replacements = {
        "«": "<",
        "‹": "<",
        "＜": "<",
        "›": ">",
        "»": ">",
        "＞": ">",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new
        )

    if "<=" in text:
        return "<="

    if ">=" in text:
        return ">="

    if "<" in text:
        return "<"

    if ">" in text:
        return ">"

    if "=" in text:
        return "="

    return text


# ============================================================
# CLASS-AWARE OCR NORMALIZATION
# ============================================================

def normalize_by_class(
    text,
    class_name
):

    if not text:
        return ""

    if class_name == "element_symbol":

        return normalize_element_symbol(
            text
        )

    if class_name == "element_value":

        return normalize_numeric_text(
            text
        )

    if class_name == "value_range":

        return normalize_range_text(
            text
        )

    if class_name == "unit":

        return normalize_unit_text(
            text
        )

    if class_name == "limit_indicator":

        return normalize_limit_indicator(
            text
        )

    if class_name == "sign":

        return normalize_sign_text(
            text
        )

    return clean_ocr_text(
        text
    )


# ============================================================
# PADDED CROP
# ============================================================

def get_padded_crop(
    image,
    box
):

    image_h, image_w = image.shape[:2]

    x1 = int(box["x1"])
    y1 = int(box["y1"])
    x2 = int(box["x2"])
    y2 = int(box["y2"])

    box_w = max(
        x2 - x1,
        1
    )

    box_h = max(
        y2 - y1,
        1
    )

    # Signs are very small, so give them more relative padding
    if box.get("class_name") == "sign":

        pad_x_factor = 0.35
        pad_y_factor = 0.40

    else:

        pad_x_factor = OCR_PAD_X
        pad_y_factor = OCR_PAD_Y

    pad_x = max(
        OCR_MIN_PAD_X,
        int(
            box_w
            * pad_x_factor
        )
    )

    pad_y = max(
        OCR_MIN_PAD_Y,
        int(
            box_h
            * pad_y_factor
        )
    )

    x1_pad = max(
        0,
        x1 - pad_x
    )

    y1_pad = max(
        0,
        y1 - pad_y
    )

    x2_pad = min(
        image_w,
        x2 + pad_x
    )

    y2_pad = min(
        image_h,
        y2 + pad_y
    )

    crop = image[
        y1_pad:y2_pad,
        x1_pad:x2_pad
    ]

    return crop


# ============================================================
# CREATE OCR VARIANTS
# ============================================================

def create_ocr_variants(crop):

    variants = []

    if crop is None:
        return variants

    if crop.size == 0:
        return variants

    # --------------------------------------------------------
    # UPSCALE
    # --------------------------------------------------------

    enlarged = cv2.resize(
        crop,
        None,
        fx=OCR_SCALE,
        fy=OCR_SCALE,
        interpolation=cv2.INTER_CUBIC
    )

    variants.append(
        (
            "original",
            enlarged
        )
    )

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    if len(enlarged.shape) == 3:

        gray = cv2.cvtColor(
            enlarged,
            cv2.COLOR_RGB2GRAY
        )

    else:

        gray = enlarged.copy()

    variants.append(
        (
            "gray",
            gray
        )
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    variants.append(
        (
            "clahe",
            enhanced
        )
    )

    # --------------------------------------------------------
    # SHARPEN
    # --------------------------------------------------------

    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(
        enhanced,
        -1,
        sharpen_kernel
    )

    variants.append(
        (
            "sharpen",
            sharpened
        )
    )

    # --------------------------------------------------------
    # LIGHT DENOISE
    # --------------------------------------------------------

    denoised = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    variants.append(
        (
            "denoised",
            denoised
        )
    )

    # --------------------------------------------------------
    # OTSU
    # --------------------------------------------------------

    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    variants.append(
        (
            "otsu",
            otsu
        )
    )

    # --------------------------------------------------------
    # INVERTED OTSU
    # --------------------------------------------------------

    otsu_inverse = cv2.bitwise_not(
        otsu
    )

    variants.append(
        (
            "otsu_inv",
            otsu_inverse
        )
    )

    # --------------------------------------------------------
    # ADAPTIVE
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9
    )

    variants.append(
        (
            "adaptive",
            adaptive
        )
    )

    return variants


# ============================================================
# RUN EASYOCR ON ONE VARIANT
# ============================================================

def run_easyocr_variant(
    variant,
    class_name
):

    allowlist = OCR_ALLOWLISTS.get(
        class_name
    )

    kwargs = {
        "detail": 1,
        "paragraph": False,
        "decoder": "greedy",

        "text_threshold": 0.35,
        "low_text": 0.20,
        "link_threshold": 0.20,

        "add_margin": 0.05,

        "rotation_info": None,
    }

    if allowlist:
        kwargs["allowlist"] = allowlist

    try:

        results = ocr_reader.readtext(
            variant,
            **kwargs
        )

    except Exception:
        return []

    candidates = []

    for result in results:

        if len(result) < 3:
            continue

        text = clean_ocr_text(
            result[1]
        )

        confidence = float(
            result[2]
        )

        if (
            text
            and confidence
            >= OCR_MIN_CONFIDENCE
        ):

            candidates.append(
                {
                    "text": text,
                    "confidence": confidence
                }
            )

    return candidates


# ============================================================
# OCR CANDIDATE SCORING
# ============================================================

def score_candidate(
    text,
    confidence,
    class_name
):

    score = float(confidence)

    clean_text = normalize_by_class(
        text,
        class_name
    )

    if not clean_text:
        return -999


    # --------------------------------------------------------
    # ELEMENT SYMBOL
    # --------------------------------------------------------

    if class_name == "element_symbol":

        if clean_text in CHEMICAL_ELEMENTS:
            score += 0.70

        elif len(clean_text) <= 2:
            score += 0.10

        else:
            score -= 0.40


    # --------------------------------------------------------
    # ELEMENT VALUE
    # --------------------------------------------------------

    elif class_name == "element_value":

        if re.search(
            r"\d",
            clean_text
        ):
            score += 0.45
        else:
            score -= 0.60

        if "." in clean_text:
            score += 0.08


    # --------------------------------------------------------
    # VALUE RANGE
    # --------------------------------------------------------

    elif class_name == "value_range":

        numbers = re.findall(
            r"\d+(?:\.\d+)?",
            clean_text
        )

        if len(numbers) >= 2:
            score += 0.60

        elif len(numbers) == 1:
            score += 0.10

        else:
            score -= 0.60

        if "-" in clean_text:
            score += 0.20


    # --------------------------------------------------------
    # UNIT
    # --------------------------------------------------------

    elif class_name == "unit":

        lower = (
            clean_text
            .lower()
            .replace(" ", "")
        )

        if lower in {
            "%",
            "ppm",
            "ppb",
            "wt%",
            "wt.%",
        }:
            score += 0.65

        elif (
            "%" in lower
            or "ppm" in lower
        ):
            score += 0.35

        else:
            score -= 0.15


    # --------------------------------------------------------
    # LIMIT INDICATOR
    # --------------------------------------------------------

    elif class_name == "limit_indicator":

        upper = clean_text.upper()

        if upper in {
            "MAX",
            "MIN"
        }:
            score += 0.80

        elif upper in {
            "MAXIMUM",
            "MINIMUM"
        }:
            score += 0.65

        else:
            score -= 0.20


    # --------------------------------------------------------
    # SIGN
    # --------------------------------------------------------

    elif class_name == "sign":

        if clean_text in {
            "<",
            ">",
            "<=",
            ">=",
            "=",
        }:
            score += 0.90

        else:
            score -= 0.60

    return score


# ============================================================
# OCR ONE YOLO BOX
# ============================================================

def read_crop_text(
    image,
    box
):

    class_name = box.get(
        "class_name",
        ""
    )

    crop = get_padded_crop(
        image,
        box
    )

    if (
        crop is None
        or crop.size == 0
    ):

        return {
            "text": "",
            "ocr_confidence": 0.0,
            "ocr_variant": "",
            "raw_text": "",
        }

    variants = create_ocr_variants(
        crop
    )

    all_candidates = []

    for (
        variant_name,
        variant
    ) in variants:

        candidates = run_easyocr_variant(
            variant,
            class_name
        )

        for candidate in candidates:

            raw_text = candidate["text"]

            confidence = candidate[
                "confidence"
            ]

            normalized = normalize_by_class(
                raw_text,
                class_name
            )

            score = score_candidate(
                raw_text,
                confidence,
                class_name
            )

            all_candidates.append(
                {
                    "raw_text": raw_text,
                    "text": normalized,
                    "confidence": confidence,
                    "variant": variant_name,
                    "score": score,
                }
            )

    if not all_candidates:

        return {
            "text": "",
            "ocr_confidence": 0.0,
            "ocr_variant": "",
            "raw_text": "",
        }

    best = max(
        all_candidates,
        key=lambda x: x["score"]
    )

    return {
        "text": best["text"],
        "ocr_confidence": best["confidence"],
        "ocr_variant": best["variant"],
        "raw_text": best["raw_text"],
    }


# ============================================================
# YOLO DETECTION
# ============================================================

def run_yolo(image):

    results = model.predict(
        source=image,

        imgsz=IMAGE_SIZE,

        conf=CONFIDENCE,

        iou=IOU_THRESHOLD,

        max_det=MAX_DETECTIONS,

        agnostic_nms=False,

        device=DEVICE,

        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:

        class_id = int(
            box.cls[0]
            .cpu()
            .item()
        )

        class_name = model.names[
            class_id
        ]

        confidence = float(
            box.conf[0]
            .cpu()
            .item()
        )

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,

                "yolo_confidence":
                    confidence,

                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            }
        )

    return detections


# ============================================================
# CLASS COLORS
# RGB because image is RGB
# ============================================================

CLASS_COLORS = {

    "element_symbol":
        (255, 0, 0),

    "element_value":
        (0, 180, 255),

    "unit":
        (0, 180, 0),

    "limit_indicator":
        (180, 0, 255),

    "value_range":
        (255, 0, 180),

    "sign":
        (255, 140, 0),
}


# ============================================================
# SHORT LABELS
# ============================================================

SHORT_NAMES = {

    "element_symbol":
        "EL",

    "element_value":
        "VAL",

    "unit":
        "UNIT",

    "limit_indicator":
        "LIM",

    "value_range":
        "RANGE",

    "sign":
        "SIGN",
}


# ============================================================
# DRAW OCR OVERLAY
# ============================================================

def draw_ocr_overlay(
    image,
    detections
):

    output = image.copy()

    for d in detections:

        x1 = d["x1"]
        y1 = d["y1"]
        x2 = d["x2"]
        y2 = d["y2"]

        text = d.get(
            "text",
            ""
        )

        class_name = d.get(
            "class_name",
            ""
        )

        color = CLASS_COLORS.get(
            class_name,
            (255, 255, 255)
        )

        # Thin detection box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            1
        )

        short_name = SHORT_NAMES.get(
            class_name,
            class_name
        )

        if text:

            display_text = (
                f"{short_name}:{text}"
            )

        else:

            display_text = (
                f"{short_name}:?"
            )

        cv2.putText(
            output,
            display_text,

            (
                x1,
                max(
                    y1 - 2,
                    8
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.28,

            color,

            1,

            cv2.LINE_AA
        )

    return output


# ============================================================
# GUI
# ============================================================

class OCRInspectionTool:

    def __init__(
        self,
        root
    ):

        self.root = root

        self.root.title(
            "Chemical Table Detection + OCR Inspection"
        )

        self.root.geometry(
            "1650x950"
        )

        self.file_path = None

        self.current_image = None

        self.annotated_image = None

        self.tk_image = None

        self.current_page = 0

        self.pdf_pages = 0

        self.detections = []


        # ====================================================
        # MAIN FRAME
        # ====================================================

        main_frame = ttk.Frame(
            root,
            padding=8
        )

        main_frame.pack(
            fill=tk.BOTH,
            expand=True
        )


        # ====================================================
        # TOP CONTROLS
        # ====================================================

        control_panel = ttk.Frame(
            main_frame
        )

        control_panel.pack(
            fill=tk.X,
            pady=(0, 8)
        )


        ttk.Button(
            control_panel,

            text="Upload PDF / Image",

            command=
                self.upload_file

        ).pack(
            side=tk.LEFT,
            padx=(0, 5)
        )


        ttk.Button(
            control_panel,

            text="Run Detection + OCR",

            command=
                self.run_inspection

        ).pack(
            side=tk.LEFT,
            padx=5
        )


        ttk.Button(
            control_panel,

            text="Previous PDF Page",

            command=
                self.previous_page

        ).pack(
            side=tk.LEFT,
            padx=5
        )


        ttk.Button(
            control_panel,

            text="Next PDF Page",

            command=
                self.next_page

        ).pack(
            side=tk.LEFT,
            padx=5
        )


        self.file_label = ttk.Label(
            control_panel,

            text="No document loaded",

            font=(
                "Segoe UI",
                10
            )
        )

        self.file_label.pack(
            side=tk.LEFT,
            padx=15
        )


        self.page_label = ttk.Label(
            control_panel,
            text=""
        )

        self.page_label.pack(
            side=tk.LEFT,
            padx=5
        )


        self.status_label = ttk.Label(
            control_panel,

            text="Ready",

            font=(
                "Segoe UI",
                10,
                "bold"
            ),

            foreground="blue"
        )

        self.status_label.pack(
            side=tk.RIGHT,
            padx=5
        )


        # ====================================================
        # BODY
        # ====================================================

        body = ttk.Panedwindow(
            main_frame,
            orient=tk.HORIZONTAL
        )

        body.pack(
            fill=tk.BOTH,
            expand=True
        )


        # ====================================================
        # LEFT - DOCUMENT IMAGE
        # ====================================================

        image_frame = ttk.Frame(
            body
        )

        body.add(
            image_frame,
            weight=4
        )


        self.canvas = tk.Canvas(
            image_frame,
            bg="#333333"
        )


        x_scroll = ttk.Scrollbar(
            image_frame,

            orient=tk.HORIZONTAL,

            command=
                self.canvas.xview
        )


        y_scroll = ttk.Scrollbar(
            image_frame,

            orient=tk.VERTICAL,

            command=
                self.canvas.yview
        )


        self.canvas.configure(
            xscrollcommand=
                x_scroll.set,

            yscrollcommand=
                y_scroll.set
        )


        self.canvas.grid(
            row=0,
            column=0,
            sticky="nsew"
        )


        y_scroll.grid(
            row=0,
            column=1,
            sticky="ns"
        )


        x_scroll.grid(
            row=1,
            column=0,
            sticky="ew"
        )


        image_frame.rowconfigure(
            0,
            weight=1
        )


        image_frame.columnconfigure(
            0,
            weight=1
        )


        # ====================================================
        # RIGHT - OCR RESULTS
        # ====================================================

        result_frame = ttk.Frame(
            body,

            padding=(
                8,
                0,
                0,
                0
            )
        )

        body.add(
            result_frame,
            weight=2
        )


        title = ttk.Label(
            result_frame,

            text="Detected regions / OCR",

            font=(
                "Segoe UI",
                11,
                "bold"
            )
        )

        title.pack(
            anchor="w",
            pady=(0, 5)
        )


        columns = (
            "class",
            "text",
            "yolo",
            "ocr",
            "variant"
        )


        self.result_table = ttk.Treeview(
            result_frame,

            columns=columns,

            show="headings",

            selectmode="browse"
        )


        self.result_table.heading(
            "class",
            text="Class"
        )

        self.result_table.heading(
            "text",
            text="OCR Text"
        )

        self.result_table.heading(
            "yolo",
            text="YOLO"
        )

        self.result_table.heading(
            "ocr",
            text="OCR"
        )

        self.result_table.heading(
            "variant",
            text="OCR Pass"
        )


        self.result_table.column(
            "class",
            width=120
        )

        self.result_table.column(
            "text",
            width=180
        )

        self.result_table.column(
            "yolo",
            width=55,
            anchor="center"
        )

        self.result_table.column(
            "ocr",
            width=55,
            anchor="center"
        )

        self.result_table.column(
            "variant",
            width=80
        )


        result_scroll = ttk.Scrollbar(
            result_frame,

            orient=tk.VERTICAL,

            command=
                self.result_table.yview
        )


        self.result_table.configure(
            yscrollcommand=
                result_scroll.set
        )


        self.result_table.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True
        )


        result_scroll.pack(
            side=tk.RIGHT,
            fill=tk.Y
        )


    # ========================================================
    # UPDATE PAGE LABEL
    # ========================================================

    def update_page_label(self):

        if (
            self.file_path
            and self.file_path.lower().endswith(
                ".pdf"
            )
        ):

            self.page_label.config(
                text=(
                    f"Page "
                    f"{self.current_page + 1}"
                    f"/"
                    f"{self.pdf_pages}"
                )
            )

        else:

            self.page_label.config(
                text=""
            )


    # ========================================================
    # UPLOAD
    # ========================================================

    def upload_file(self):

        path = filedialog.askopenfilename(

            title="Select document",

            filetypes=[
                (
                    "PDF and images",

                    "*.pdf *.png *.jpg *.jpeg "
                    "*.bmp *.tif *.tiff *.webp"
                )
            ]
        )

        if not path:
            return

        self.file_path = path

        self.current_page = 0

        self.file_label.config(
            text=os.path.basename(
                path
            )
        )

        extension = os.path.splitext(
            path
        )[1].lower()

        try:

            if extension == ".pdf":

                doc = fitz.open(
                    path
                )

                self.pdf_pages = len(
                    doc
                )

                doc.close()

                self.current_image = (
                    pdf_page_to_rgb(
                        path,
                        0
                    )
                )

            else:

                self.pdf_pages = 1

                self.current_image = (
                    load_image_rgb(
                        path
                    )
                )

            self.annotated_image = None

            self.detections = []

            self.clear_results()

            self.show_image(
                self.current_image
            )

            self.update_page_label()

            self.status_label.config(
                text=(
                    "Document loaded. "
                    "Click 'Run Detection + OCR'."
                )
            )

        except Exception as e:

            messagebox.showerror(
                "Error",
                str(e)
            )


    # ========================================================
    # LOAD CURRENT PDF PAGE
    # ========================================================

    def load_current_pdf_page(self):

        if not self.file_path:
            return

        if not self.file_path.lower().endswith(
            ".pdf"
        ):
            return

        self.current_image = (
            pdf_page_to_rgb(
                self.file_path,
                self.current_page
            )
        )

        self.annotated_image = None

        self.detections = []

        self.clear_results()

        self.show_image(
            self.current_image
        )

        self.update_page_label()

        self.status_label.config(
            text="PDF page loaded."
        )


    # ========================================================
    # PREVIOUS PAGE
    # ========================================================

    def previous_page(self):

        if (
            not self.file_path
            or not self.file_path.lower().endswith(
                ".pdf"
            )
        ):
            return

        if self.current_page > 0:

            self.current_page -= 1

            self.load_current_pdf_page()


    # ========================================================
    # NEXT PAGE
    # ========================================================

    def next_page(self):

        if (
            not self.file_path
            or not self.file_path.lower().endswith(
                ".pdf"
            )
        ):
            return

        if (
            self.current_page
            < self.pdf_pages - 1
        ):

            self.current_page += 1

            self.load_current_pdf_page()


    # ========================================================
    # CLEAR RESULT TABLE
    # ========================================================

    def clear_results(self):

        for item in (
            self.result_table.get_children()
        ):

            self.result_table.delete(
                item
            )


    # ========================================================
    # POPULATE RESULTS
    # ========================================================

    def populate_results(
        self,
        detections
    ):

        self.clear_results()

        sorted_detections = sorted(

            detections,

            key=lambda d: (
                d["y1"],
                d["x1"]
            )
        )

        for detection in sorted_detections:

            yolo_conf = detection.get(
                "yolo_confidence",
                0
            )

            ocr_conf = detection.get(
                "ocr_confidence",
                0
            )

            self.result_table.insert(

                "",

                tk.END,

                values=(

                    detection.get(
                        "class_name",
                        ""
                    ),

                    detection.get(
                        "text",
                        ""
                    ),

                    f"{yolo_conf:.2f}",

                    f"{ocr_conf:.2f}",

                    detection.get(
                        "ocr_variant",
                        ""
                    )
                )
            )


    # ========================================================
    # RUN DETECTION + OCR
    # ========================================================

    def run_inspection(self):

        if self.current_image is None:

            messagebox.showwarning(
                "Warning",
                "Please upload a document first."
            )

            return

        try:

            # ------------------------------------------------
            # YOLO
            # ------------------------------------------------

            self.status_label.config(
                text=
                    "Detecting regions with YOLO..."
            )

            self.root.update_idletasks()

            detections = run_yolo(
                self.current_image
            )


            self.status_label.config(
                text=(
                    f"Detected "
                    f"{len(detections)} regions. "
                    f"Running OCR..."
                )
            )

            self.root.update_idletasks()


            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            for index, detection in enumerate(
                detections,
                start=1
            ):

                self.status_label.config(
                    text=(
                        f"OCR "
                        f"{index}/"
                        f"{len(detections)}"
                    )
                )

                self.root.update_idletasks()

                ocr_result = read_crop_text(
                    self.current_image,
                    detection
                )

                detection.update(
                    ocr_result
                )


            # ------------------------------------------------
            # STORE
            # ------------------------------------------------

            self.detections = detections


            # ------------------------------------------------
            # DRAW
            # ------------------------------------------------

            self.status_label.config(
                text=
                    "Rendering OCR overlay..."
            )

            self.root.update_idletasks()

            self.annotated_image = (
                draw_ocr_overlay(
                    self.current_image,
                    detections
                )
            )

            self.show_image(
                self.annotated_image
            )


            # ------------------------------------------------
            # TABLE
            # ------------------------------------------------

            self.populate_results(
                detections
            )


            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            recognized = sum(
                1
                for d in detections
                if d.get(
                    "text",
                    ""
                )
            )

            class_counts = {}

            for d in detections:

                name = d.get(
                    "class_name",
                    "unknown"
                )

                class_counts[name] = (
                    class_counts.get(
                        name,
                        0
                    ) + 1
                )

            print("\n" + "=" * 70)
            print("DETECTION SUMMARY")
            print("=" * 70)

            for name in EXPECTED_CLASSES:

                print(
                    f"{name:18s}: "
                    f"{class_counts.get(name, 0)}"
                )

            print("-" * 70)

            print(
                "Total detections:",
                len(detections)
            )

            print(
                "OCR recognized :",
                recognized
            )

            print("=" * 70)


            self.status_label.config(
                text=(
                    f"Complete: "
                    f"{len(detections)} boxes, "
                    f"{recognized} OCR results."
                )
            )

        except Exception as e:

            messagebox.showerror(
                "Error",
                str(e)
            )

            self.status_label.config(
                text="Processing failed."
            )


    # ========================================================
    # SHOW IMAGE
    # ========================================================

    def show_image(
        self,
        image
    ):

        if image is None:
            return

        h, w = image.shape[:2]

        pil_image = Image.fromarray(
            image
        )

        self.tk_image = (
            ImageTk.PhotoImage(
                pil_image
            )
        )

        self.canvas.delete(
            "all"
        )

        self.canvas.create_image(
            0,
            0,

            anchor=tk.NW,

            image=self.tk_image
        )

        self.canvas.configure(
            scrollregion=(
                0,
                0,
                w,
                h
            )
        )


# ============================================================
# START GUI
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = OCRInspectionTool(
        root
    )

    root.mainloop()
