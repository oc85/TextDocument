# ============================================================
# CHEMICAL DOCUMENT YOLO + OCR INSPECTION VIEWER
#
# FEATURES
# ------------------------------------------------------------
# - PDF / image viewer
# - Modern PyMuPDF import: import pymupdf
# - Automatic Fit Page
# - Fit Width
# - Mouse-wheel zoom AT cursor
# - Left mouse drag = pan document
# - Shift + mouse wheel = horizontal movement
# - Previous / next PDF page
# - YOLO detection
# - EasyOCR
# - Manual OCR mapping dictionaries near TOP of code
# - Raw OCR + corrected OCR in result table
# - Clean non-obstructive OCR overlay
# - Show / Hide overlay
# - Click result row -> move to detection
#
# IMPORTANT VALUE RECOGNITION
# ------------------------------------------------------------
# element_value can contain:
#
#   0.10
#   5.4
#   3.20
#   -
#   Bal
#   Balance
#   Trace
#
# Therefore:
#
#   1. numeric OCR is performed
#   2. text OCR is also performed
#   3. known mapped text such as Bal has priority
#   4. otherwise valid numeric result wins
#   5. then real dash
#   6. then unknown text
#
# CURRENT YOLO CLASSES:
#
# 0 element_symbol
# 1 element_value
# 2 unit
# 3 limit_indicator
# 4 value_range
# 5 sign
#
# ============================================================

import re
import threading
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

MODEL_PATH = Path(
    r"D:\Kaggle\WRtools_003\best.pt"
)

IMAGE_SIZE = 1280

CONFIDENCE = 0.03

IOU_THRESHOLD = 0.45

MAX_DETECTIONS = 2000

PDF_DPI = 300

DEVICE = (
    0
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 2. EXPECTED YOLO CLASSES
# ============================================================

EXPECTED_CLASSES = {
    0: "element_symbol",
    1: "element_value",
    2: "unit",
    3: "limit_indicator",
    4: "value_range",
    5: "sign",
}


# ============================================================
# 3. MANUAL OCR MAPPINGS
#
# EDIT THESE WHEN YOU FIND REPEATING OCR ERRORS.
#
# LEFT  = OCR result / mistake
# RIGHT = final corrected value
#
# Mapping lookup is case-insensitive.
# ============================================================


# ------------------------------------------------------------
# ELEMENT SYMBOL MAPPING
# ------------------------------------------------------------

ELEMENT_MAPPING = {

    # Aluminium
    "al": "Al",
    "a1": "Al",
    "ai": "Al",
    "a|": "Al",

    # Boron
    "b": "B",

    # Carbon
    "c": "C",

    # Cadmium
    "cd": "Cd",
    "cb": "Cd",

    # Chromium
    "cr": "Cr",
    "gr": "Cr",
    "c r": "Cr",

    # Cobalt
    "co": "Co",
    "c0": "Co",

    # Copper
    "cu": "Cu",

    # Iron
    "fe": "Fe",
    "f e": "Fe",

    # Hafnium
    "hf": "Hf",

    # Manganese
    "mn": "Mn",

    # Molybdenum
    "mo": "Mo",
    "m0": "Mo",

    # Niobium
    "nb": "Nb",

    # Nickel
    "ni": "Ni",
    "nl": "Ni",
    "n1": "Ni",

    # Phosphorus
    "p": "P",

    # Lead
    "pb": "Pb",

    # Rhenium
    "re": "Re",

    # Sulfur
    "s": "S",

    # Silicon
    "si": "Si",
    "sl": "Si",

    # Tin
    "sn": "Sn",

    # Tantalum
    "ta": "Ta",

    # Titanium
    "ti": "Ti",

    # Vanadium
    "v": "V",

    # Tungsten
    "w": "W",
    "vv": "W",

    # Zirconium
    "zr": "Zr",

    # Silver
    "ag": "Ag",

    # Bismuth
    "bi": "Bi",
}


# ------------------------------------------------------------
# UNIT MAPPING
# ------------------------------------------------------------

UNIT_MAPPING = {

    "%": "%",

    "percent": "%",
    "percentage": "%",
    "0/0": "%",
    "o/o": "%",

    "ppm": "ppm",
    "prm": "ppm",
    "ppn": "ppm",
    "pom": "ppm",
    "prn": "ppm",

    "ppb": "ppb",

    "wt%": "wt%",
    "wt.%": "wt%",
}


# ------------------------------------------------------------
# LIMIT INDICATOR MAPPING
# ------------------------------------------------------------

LIMIT_MAPPING = {

    "max": "MAX",
    "maximum": "MAX",
    "rnax": "MAX",
    "rnaximum": "MAX",
    "m4x": "MAX",

    "min": "MIN",
    "minimum": "MIN",
    "rnin": "MIN",
    "rninimum": "MIN",
}


# ------------------------------------------------------------
# SIGN MAPPING
# ------------------------------------------------------------

SIGN_MAPPING = {

    "<": "<",
    "‹": "<",
    "≤": "<=",
    "<=": "<=",

    ">": ">",
    "›": ">",
    "≥": ">=",
    ">=": ">=",
}


# ------------------------------------------------------------
# SPECIAL ELEMENT VALUE MAPPING
#
# IMPORTANT:
# Add recurring text values here.
#
# Example:
# OCR sees "Bai" -> final should be "Bal"
# ------------------------------------------------------------

VALUE_MAPPING = {

    "-": "-",
    "–": "-",
    "—": "-",
    "−": "-",

    "bal": "Bal",
    "bai": "Bal",
    "ba1": "Bal",
    "bal.": "Bal",

    "balance": "Balance",
    "banch": "Banch",

    "trace": "Trace",
}


# ------------------------------------------------------------
# RANGE MAPPING
# ------------------------------------------------------------

RANGE_MAPPING = {

    "–": "-",
    "—": "-",
    "−": "-",
}


# ============================================================
# 4. VALID ELEMENTS
# ============================================================

VALID_ELEMENTS = {
    "Ag",
    "Al",
    "B",
    "Bi",
    "C",
    "Cd",
    "Co",
    "Cr",
    "Cu",
    "Fe",
    "Hf",
    "Mn",
    "Mo",
    "Nb",
    "Ni",
    "P",
    "Pb",
    "Re",
    "S",
    "Sb",
    "Si",
    "Sn",
    "Ta",
    "Ti",
    "V",
    "W",
    "Zr",
}


# ============================================================
# 5. OCR SETTINGS
# ============================================================

OCR_SCALE = 4.0

# General classes
OCR_PAD_X = 0.12
OCR_PAD_Y = 0.16

# Numeric values:
# Keep vertical padding small so table lines don't enter crop.
VALUE_PAD_X = 0.08
VALUE_PAD_Y = 0.04

# Ranges
RANGE_PAD_X = 0.08
RANGE_PAD_Y = 0.05

OCR_MIN_CONFIDENCE = 0.15


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

OVERLAY_LABEL_ALPHA = 0.58

OVERLAY_MAX_TEXT_LENGTH = 22


# ============================================================
# 7. DISPLAY COLOURS
# ============================================================

CLASS_COLOURS = {

    "element_symbol":
        (215, 50, 50),

    "element_value":
        (255, 140, 0),

    "unit":
        (30, 170, 60),

    "limit_indicator":
        (145, 70, 200),

    "value_range":
        (220, 40, 160),

    "sign":
        (40, 120, 220),
}


CLASS_SHORT = {

    "element_symbol": "E",

    "element_value": "V",

    "unit": "U",

    "limit_indicator": "L",

    "value_range": "R",

    "sign": "S",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):

    value = str(value)

    value = value.replace(
        "\n",
        " "
    )

    return re.sub(
        r"\s+",
        " ",
        value
    ).strip()


def mapping_key(text):

    return clean_text(
        text
    ).lower()


def apply_mapping(
    text,
    mapping
):

    key = mapping_key(
        text
    )

    return mapping.get(
        key
    )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image_rgb(path):

    data = np.fromfile(
        str(path),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        data,
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
# PDF PAGE -> RGB IMAGE
# ============================================================

def pdf_page_to_rgb(
    pdf_path,
    page_number,
    dpi=PDF_DPI
):

    with pymupdf.open(
        str(pdf_path)
    ) as document:

        if document.page_count == 0:

            raise ValueError(
                "The PDF contains no pages."
            )

        page_number = max(
            0,
            min(
                page_number,
                document.page_count - 1
            )
        )

        page = document[
            page_number
        ]

        scale = (
            dpi / 72.0
        )

        matrix = pymupdf.Matrix(
            scale,
            scale
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

        return image.copy()


# ============================================================
# PAD DETECTION CROP
# ============================================================

def padded_crop(
    image,
    detection
):

    x1 = detection["x1"]
    y1 = detection["y1"]

    x2 = detection["x2"]
    y2 = detection["y2"]

    width = max(
        1,
        x2 - x1
    )

    height = max(
        1,
        y2 - y1
    )

    class_name = detection[
        "class_name"
    ]

    if class_name == "element_value":

        pad_x_fraction = VALUE_PAD_X
        pad_y_fraction = VALUE_PAD_Y

    elif class_name == "value_range":

        pad_x_fraction = RANGE_PAD_X
        pad_y_fraction = RANGE_PAD_Y

    else:

        pad_x_fraction = OCR_PAD_X
        pad_y_fraction = OCR_PAD_Y

    pad_x = max(
        1,
        int(
            width
            * pad_x_fraction
        )
    )

    pad_y = max(
        1,
        int(
            height
            * pad_y_fraction
        )
    )

    crop_x1 = max(
        0,
        x1 - pad_x
    )

    crop_y1 = max(
        0,
        y1 - pad_y
    )

    crop_x2 = min(
        image.shape[1],
        x2 + pad_x
    )

    crop_y2 = min(
        image.shape[0],
        y2 + pad_y
    )

    return image[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]


# ============================================================
# OCR PREPROCESSING
# ============================================================

def preprocess_variants(crop):

    if (
        crop is None
        or crop.size == 0
    ):

        return []

    enlarged = cv2.resize(

        crop,

        None,

        fx=OCR_SCALE,

        fy=OCR_SCALE,

        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_RGB2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(
        gray
    )

    sharpen = cv2.addWeighted(

        clahe,

        1.6,

        cv2.GaussianBlur(
            clahe,
            (0, 0),
            1.0
        ),

        -0.6,

        0
    )

    denoise = cv2.fastNlMeansDenoising(
        sharpen,
        None,
        7,
        7,
        21
    )

    otsu = cv2.threshold(

        denoise,

        0,

        255,

        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU

    )[1]

    inverted_otsu = cv2.bitwise_not(
        otsu
    )

    adaptive = cv2.adaptiveThreshold(

        denoise,

        255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY,

        31,

        9
    )

    return [

        enlarged,
        gray,
        clahe,
        sharpen,
        denoise,
        otsu,
        inverted_otsu,
        adaptive,
    ]


# ============================================================
# STRICT DASH FALLBACK
#
# IMPORTANT:
# Only used AFTER OCR has failed to find useful number/text.
# ============================================================

def strict_dash_fallback(crop):

    if (
        crop is None
        or crop.size == 0
    ):

        return False

    if crop.ndim == 3:

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_RGB2GRAY
        )

    else:

        gray = crop.copy()

    gray = cv2.resize(

        gray,

        None,

        fx=5.0,

        fy=5.0,

        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.copyMakeBorder(

        gray,

        15,
        15,
        15,
        15,

        cv2.BORDER_CONSTANT,

        value=255
    )

    binary = cv2.threshold(

        gray,

        0,

        255,

        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU

    )[1]

    contours, _ = cv2.findContours(

        binary,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )

    image_h, image_w = (
        binary.shape[:2]
    )

    for contour in contours:

        x, y, width, height = (
            cv2.boundingRect(
                contour
            )
        )

        area = (
            width
            * height
        )

        if area < 20:

            continue

        ratio = (
            width
            / max(
                height,
                1
            )
        )

        # Must be strongly horizontal
        if ratio < 3.2:

            continue

        # Must NOT look like a long table border
        if width > (
            image_w
            * 0.65
        ):

            continue

        # Must be short vertically
        if height > (
            image_h
            * 0.22
        ):

            continue

        cx = (
            x
            + width / 2
        )

        cy = (
            y
            + height / 2
        )

        centre_x_ok = (
            abs(
                cx
                - image_w / 2
            )
            < image_w * 0.32
        )

        centre_y_ok = (
            abs(
                cy
                - image_h / 2
            )
            < image_h * 0.32
        )

        if (
            centre_x_ok
            and centre_y_ok
        ):

            return True

    return False


# ============================================================
# NORMALISE ELEMENT
# ============================================================

def normalise_element(text):

    mapped = apply_mapping(
        text,
        ELEMENT_MAPPING
    )

    if mapped is not None:

        return mapped, True

    token = re.sub(
        r"[^A-Za-z]",
        "",
        clean_text(text)
    )

    if not token:

        return clean_text(text), False

    if len(token) == 1:

        candidate = (
            token.upper()
        )

    elif len(token) == 2:

        candidate = (
            token[0].upper()
            + token[1].lower()
        )

    else:

        return clean_text(text), False

    if candidate in VALID_ELEMENTS:

        return candidate, True

    return clean_text(text), False


# ============================================================
# NORMALISE UNIT
# ============================================================

def normalise_unit(text):

    mapped = apply_mapping(
        text,
        UNIT_MAPPING
    )

    if mapped is not None:

        return mapped, True

    token = (
        clean_text(text)
        .lower()
        .replace(
            " ",
            ""
        )
    )

    if "%" in token:

        return "%", True

    if "ppm" in token:

        return "ppm", True

    if "ppb" in token:

        return "ppb", True

    return clean_text(text), False


# ============================================================
# NORMALISE LIMIT
# ============================================================

def normalise_limit(text):

    mapped = apply_mapping(
        text,
        LIMIT_MAPPING
    )

    if mapped is not None:

        return mapped, True

    token = (
        clean_text(text)
        .lower()
    )

    if token.startswith(
        "max"
    ):

        return "MAX", True

    if token.startswith(
        "min"
    ):

        return "MIN", True

    return clean_text(text), False


# ============================================================
# NORMALISE SIGN
# ============================================================

def normalise_sign(text):

    mapped = apply_mapping(
        text,
        SIGN_MAPPING
    )

    if mapped is not None:

        return mapped, True

    value = (
        clean_text(text)
        .replace(
            "≤",
            "<="
        )
        .replace(
            "≥",
            ">="
        )
    )

    if value in {
        "<",
        ">",
        "<=",
        ">="
    }:

        return value, True

    return value, False


# ============================================================
# NORMALISE ELEMENT VALUE
# ============================================================

def normalise_value(text):

    original = clean_text(
        text
    )

    mapped = apply_mapping(
        original,
        VALUE_MAPPING
    )

    if mapped is not None:

        return mapped, True

    value = original.translate(

        str.maketrans({

            "O": "0",
            "o": "0",

            "I": "1",
            "l": "1",
            "|": "1",

            ",": ".",
        })
    )

    value = (
        value
        .replace(
            "−",
            "-"
        )
        .replace(
            "–",
            "-"
        )
        .replace(
            "—",
            "-"
        )
    )

    value = re.sub(
        r"\s+",
        "",
        value
    )

    if value == "-":

        return "-", True

    if re.fullmatch(
        r"[+-]?\d+(?:\.\d+)?",
        value
    ):

        return value, True

    if re.fullmatch(
        r"[<>]=?[+-]?\d+(?:\.\d+)?",
        value
    ):

        return value, True

    if re.fullmatch(
        r"[A-Za-z]+",
        original
    ):

        return original, True

    return value, False


# ============================================================
# NORMALISE RANGE
# ============================================================

def normalise_range(text):

    value = clean_text(
        text
    )

    for old, new in (
        ("–", "-"),
        ("—", "-"),
        ("−", "-"),
    ):

        value = value.replace(
            old,
            new
        )

    value = (
        value
        .replace(
            ",",
            "."
        )
        .replace(
            " ",
            ""
        )
    )

    if re.fullmatch(

        r"[+-]?\d+(?:\.\d+)?"
        r"-"
        r"[+-]?\d+(?:\.\d+)?",

        value
    ):

        return value, True

    return value, False


# ============================================================
# OCR ALLOWLIST
# ============================================================

def get_allowlist(class_name):

    if class_name == "element_symbol":

        return (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
        )

    if class_name == "unit":

        return (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "%."
        )

    if class_name == "limit_indicator":

        return (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
        )

    if class_name == "value_range":

        return (
            "0123456789"
            ".,+-"
        )

    if class_name == "sign":

        return (
            "<>=-"
        )

    return None


# ============================================================
# EASY OCR CANDIDATES
# ============================================================

def run_easyocr_candidates(
    reader,
    variants,
    allowlist,
    text_threshold=0.40,
    low_text=0.20,
    link_threshold=0.25,
):

    candidates = []

    for variant_index, variant in enumerate(
        variants
    ):

        try:

            results = reader.readtext(

                variant,

                detail=1,

                paragraph=False,

                allowlist=allowlist,

                decoder="beamsearch",

                beamWidth=5,

                contrast_ths=0.05,

                adjust_contrast=0.7,

                text_threshold=text_threshold,

                low_text=low_text,

                link_threshold=link_threshold,
            )

            if not results:

                continue

            raw = clean_text(

                " ".join(
                    item[1]
                    for item
                    in results
                )
            )

            confidence = float(

                np.mean(
                    [
                        item[2]
                        for item
                        in results
                    ]
                )
            )

            if raw:

                candidates.append({

                    "raw":
                        raw,

                    "confidence":
                        confidence,

                    "variant":
                        variant_index,
                })

        except Exception:

            continue

    return candidates


# ============================================================
# ELEMENT VALUE OCR
#
# Handles:
#
#   0.10
#   5.4
#   3.20
#   -
#   Bal
#   Balance
#   Trace
#
# Priority:
#
#   known text
#   numeric
#   OCR dash
#   strict dash fallback
#   unknown text
# ============================================================

def read_element_value(
    reader,
    crop
):

    variants = preprocess_variants(
        crop
    )

    if not variants:

        return (
            "",
            0.0,
            False,
            ""
        )


    # ========================================================
    # PASS A - NUMERIC OCR
    # ========================================================

    numeric_results = (
        run_easyocr_candidates(

            reader,

            variants,

            "0123456789.,+-<>",

            text_threshold=0.30,

            low_text=0.10,

            link_threshold=0.16,
        )
    )

    numeric_candidates = []


    for candidate in numeric_results:

        raw = candidate[
            "raw"
        ]

        final, valid = (
            normalise_value(
                raw
            )
        )

        if (
            valid
            and final != "-"
            and (
                re.fullmatch(
                    r"[+-]?\d+(?:\.\d+)?",
                    final
                )
                or
                re.fullmatch(
                    r"[<>]=?[+-]?\d+(?:\.\d+)?",
                    final
                )
            )
        ):

            numeric_candidates.append({

                "raw":
                    raw,

                "final":
                    final,

                "confidence":
                    candidate[
                        "confidence"
                    ],
            })


    # ========================================================
    # PASS B - LETTERS ONLY
    #
    # Important:
    # B cannot become 8 in this pass.
    # ========================================================

    text_results = (
        run_easyocr_candidates(

            reader,

            variants,

            (
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "abcdefghijklmnopqrstuvwxyz"
            ),

            text_threshold=0.28,

            low_text=0.10,

            link_threshold=0.16,
        )
    )


    # ========================================================
    # PRIORITY 1 - KNOWN TEXT
    # ========================================================

    mapped_text_candidates = []


    for candidate in text_results:

        raw = clean_text(
            candidate[
                "raw"
            ]
        )

        mapped = apply_mapping(
            raw,
            VALUE_MAPPING
        )

        if (
            mapped is not None
            and mapped != "-"
        ):

            mapped_text_candidates.append({

                "raw":
                    raw,

                "final":
                    mapped,

                "confidence":
                    candidate[
                        "confidence"
                    ],
            })


    if mapped_text_candidates:

        best = max(

            mapped_text_candidates,

            key=lambda item:
                item[
                    "confidence"
                ]
        )

        return (

            best[
                "final"
            ],

            best[
                "confidence"
            ],

            True,

            best[
                "raw"
            ]
        )


    # ========================================================
    # PRIORITY 2 - NUMERIC RESULT
    # ========================================================

    if numeric_candidates:

        best = max(

            numeric_candidates,

            key=lambda item: (

                item[
                    "confidence"
                ],

                len(
                    item[
                        "final"
                    ]
                )
            )
        )

        return (

            best[
                "final"
            ],

            best[
                "confidence"
            ],

            True,

            best[
                "raw"
            ]
        )


    # ========================================================
    # PRIORITY 3 - REAL OCR DASH
    # ========================================================

    dash_candidates = []

    for candidate in numeric_results:

        raw = (
            clean_text(
                candidate[
                    "raw"
                ]
            )
            .replace(
                "–",
                "-"
            )
            .replace(
                "—",
                "-"
            )
            .replace(
                "−",
                "-"
            )
        )

        if raw == "-":

            dash_candidates.append(
                candidate
            )


    if dash_candidates:

        best = max(

            dash_candidates,

            key=lambda item:
                item[
                    "confidence"
                ]
        )

        return (

            "-",

            best[
                "confidence"
            ],

            True,

            best[
                "raw"
            ]
        )


    # ========================================================
    # PRIORITY 4 - STRICT DASH IMAGE FALLBACK
    # ========================================================

    if strict_dash_fallback(
        crop
    ):

        return (
            "-",
            0.70,
            True,
            "-"
        )


    # ========================================================
    # PRIORITY 5 - UNKNOWN TEXT
    # ========================================================

    if text_results:

        best = max(

            text_results,

            key=lambda item:
                item[
                    "confidence"
                ]
        )

        raw = clean_text(
            best[
                "raw"
            ]
        )

        final, mapped = (
            normalise_value(
                raw
            )
        )

        return (

            final,

            best[
                "confidence"
            ],

            mapped,

            raw
        )


    return (
        "",
        0.0,
        False,
        ""
    )


# ============================================================
# RANGE OCR
# ============================================================

def read_value_range(
    reader,
    crop
):

    variants = preprocess_variants(
        crop
    )

    candidates = (
        run_easyocr_candidates(

            reader,

            variants,

            "0123456789.,+-",

            text_threshold=0.30,

            low_text=0.10,

            link_threshold=0.16,
        )
    )

    if not candidates:

        return (
            "",
            0.0,
            False,
            ""
        )

    processed = []

    for candidate in candidates:

        final, valid = (
            normalise_range(
                candidate[
                    "raw"
                ]
            )
        )

        processed.append({

            "raw":
                candidate[
                    "raw"
                ],

            "final":
                final,

            "valid":
                valid,

            "confidence":
                candidate[
                    "confidence"
                ],
        })

    processed.sort(

        key=lambda item: (

            item[
                "valid"
            ],

            item[
                "confidence"
            ],

            len(
                item[
                    "final"
                ]
            )
        ),

        reverse=True
    )

    best = processed[
        0
    ]

    return (

        best[
            "final"
        ],

        best[
            "confidence"
        ],

        best[
            "valid"
        ],

        best[
            "raw"
        ]
    )


# ============================================================
# OCR ONE CROP
# ============================================================

def read_crop_text(
    reader,
    image,
    detection
):

    class_name = detection[
        "class_name"
    ]

    crop = padded_crop(
        image,
        detection
    )

    if (
        crop is None
        or crop.size == 0
    ):

        return (
            "",
            0.0,
            False,
            ""
        )


    # --------------------------------------------------------
    # element_value
    # --------------------------------------------------------

    if class_name == "element_value":

        return read_element_value(
            reader,
            crop
        )


    # --------------------------------------------------------
    # value_range
    # --------------------------------------------------------

    if class_name == "value_range":

        return read_value_range(
            reader,
            crop
        )


    allowlist = get_allowlist(
        class_name
    )

    candidates = (
        run_easyocr_candidates(

            reader,

            preprocess_variants(
                crop
            ),

            allowlist,

            text_threshold=0.38,

            low_text=0.18,

            link_threshold=0.22,
        )
    )

    if not candidates:

        return (
            "",
            0.0,
            False,
            ""
        )

    raw_text, confidence = max(

        (
            (
                item[
                    "raw"
                ],

                item[
                    "confidence"
                ]
            )

            for item
            in candidates
        ),

        key=lambda item: (

            item[1],

            len(
                item[0]
            )
        )
    )


    if class_name == "element_symbol":

        final_text, mapped = (
            normalise_element(
                raw_text
            )
        )

    elif class_name == "unit":

        final_text, mapped = (
            normalise_unit(
                raw_text
            )
        )

    elif class_name == "limit_indicator":

        final_text, mapped = (
            normalise_limit(
                raw_text
            )
        )

    elif class_name == "sign":

        final_text, mapped = (
            normalise_sign(
                raw_text
            )
        )

    else:

        final_text = raw_text

        mapped = False


    return (

        final_text,

        confidence,

        mapped,

        raw_text
    )


# ============================================================
# SORT READING ORDER
# ============================================================

def sort_reading_order(
    detections
):

    if not detections:

        return detections

    median_height = max(

        1,

        int(
            np.median(
                [
                    d["y2"]
                    - d["y1"]
                    for d
                    in detections
                ]
            )
        )
    )

    return sorted(

        detections,

        key=lambda d: (

            round(
                d["cy"]
                / median_height
            ),

            d["x1"]
        )
    )


# ============================================================
# OVERLAY LABEL
# ============================================================

def make_overlay_label(
    detection
):

    class_name = detection[
        "class_name"
    ]

    text = (
        detection.get(
            "text",
            ""
        )
        or "?"
    )

    if len(text) > OVERLAY_MAX_TEXT_LENGTH:

        text = (
            text[
                :OVERLAY_MAX_TEXT_LENGTH - 3
            ]
            + "..."
        )

    if SHOW_CLASS_PREFIX:

        short = CLASS_SHORT.get(
            class_name,
            "?"
        )

        return (
            f"{short}: {text}"
        )

    return text


# ============================================================
# DRAW TRANSPARENT LABEL
# ============================================================

def draw_transparent_label(
    image,
    label,
    x,
    y,
    colour,
):

    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )

    font_scale = (
        OVERLAY_FONT_SCALE
    )

    thickness = (
        OVERLAY_FONT_THICKNESS
    )

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(

        label,

        font,

        font_scale,

        thickness
    )

    pad = (
        OVERLAY_TEXT_PADDING
    )

    x1 = max(
        0,
        x
    )

    y1 = max(
        0,
        y
    )

    x2 = min(

        image.shape[1] - 1,

        x1
        + text_width
        + pad * 2
    )

    y2 = min(

        image.shape[0] - 1,

        y1
        + text_height
        + baseline
        + pad * 2
    )

    overlay = (
        image.copy()
    )

    cv2.rectangle(

        overlay,

        (
            x1,
            y1
        ),

        (
            x2,
            y2
        ),

        colour,

        -1
    )

    cv2.addWeighted(

        overlay,

        OVERLAY_LABEL_ALPHA,

        image,

        1.0
        - OVERLAY_LABEL_ALPHA,

        0,

        image
    )

    cv2.putText(

        image,

        label,

        (
            x1 + pad,
            y1
            + text_height
            + pad
        ),

        font,

        font_scale,

        (
            255,
            255,
            255
        ),

        thickness,

        cv2.LINE_AA
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# OVERLAP TEST
# ============================================================

def rectangles_overlap(
    a,
    b
):

    ax1, ay1, ax2, ay2 = a

    bx1, by1, bx2, by2 = b

    return not (

        ax2 < bx1
        or bx2 < ax1
        or ay2 < by1
        or by2 < ay1
    )


# ============================================================
# DRAW OCR OVERLAY
# ============================================================

def draw_overlay(
    image,
    detections
):

    output = image.copy()

    occupied_labels = []

    for detection in detections:

        x1 = detection[
            "x1"
        ]

        y1 = detection[
            "y1"
        ]

        x2 = detection[
            "x2"
        ]

        y2 = detection[
            "y2"
        ]

        class_name = detection[
            "class_name"
        ]

        colour = CLASS_COLOURS.get(

            class_name,

            (120, 120, 120)
        )

        cv2.rectangle(

            output,

            (
                x1,
                y1
            ),

            (
                x2,
                y2
            ),

            colour,

            OVERLAY_BOX_THICKNESS
        )

        if not SHOW_OVERLAY_TEXT:

            continue

        label = make_overlay_label(
            detection
        )

        font = (
            cv2.FONT_HERSHEY_SIMPLEX
        )

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(

            label,

            font,

            OVERLAY_FONT_SCALE,

            OVERLAY_FONT_THICKNESS
        )

        label_width = (
            text_width
            + OVERLAY_TEXT_PADDING * 2
        )

        label_height = (
            text_height
            + baseline
            + OVERLAY_TEXT_PADDING * 2
        )

        candidates = [

            (
                x2 + 4,
                y1
            ),

            (
                max(
                    0,
                    x1
                    - label_width
                    - 4
                ),
                y1
            ),

            (
                x1,
                y1
                - label_height
                - 2
            ),

            (
                x1,
                y2 + 2
            ),
        ]

        chosen = None

        for candidate_x, candidate_y in candidates:

            candidate_x = max(
                0,
                candidate_x
            )

            candidate_y = max(
                0,
                candidate_y
            )

            if (
                candidate_x
                + label_width
                >= output.shape[1]
            ):

                candidate_x = max(

                    0,

                    output.shape[1]
                    - label_width
                    - 1
                )

            if (
                candidate_y
                + label_height
                >= output.shape[0]
            ):

                candidate_y = max(

                    0,

                    output.shape[0]
                    - label_height
                    - 1
                )

            rect = (

                candidate_x,
                candidate_y,

                candidate_x
                + label_width,

                candidate_y
                + label_height
            )

            has_overlap = any(

                rectangles_overlap(
                    rect,
                    existing
                )

                for existing
                in occupied_labels
            )

            if not has_overlap:

                chosen = (
                    candidate_x,
                    candidate_y,
                    rect
                )

                break

        if chosen is None:

            candidate_x = max(
                0,
                x2 + 3
            )

            candidate_y = max(
                0,
                y1
            )

            if (
                candidate_x
                + label_width
                >= output.shape[1]
            ):

                candidate_x = max(
                    0,
                    x1
                    - label_width
                    - 3
                )

            rect = (

                candidate_x,
                candidate_y,

                candidate_x
                + label_width,

                candidate_y
                + label_height
            )

            chosen = (
                candidate_x,
                candidate_y,
                rect
            )

        label_x, label_y, rect = (
            chosen
        )

        draw_transparent_label(

            output,

            label,

            label_x,

            label_y,

            colour
        )

        occupied_labels.append(
            rect
        )

    return output


# ============================================================
# GUI
# ============================================================

class OCRInspectionTool:

    def __init__(
        self,
        root,
        model,
        reader
    ):

        self.root = root

        self.model = model

        self.reader = reader

        self.file_path = None

        self.current_image = None

        self.annotated_image = None

        self.detections = []

        self.current_page = 0

        self.pdf_pages = 0

        self.tk_image = None

        self.zoom = 1.0

        self.min_zoom = 0.05

        self.max_zoom = 8.0

        self.busy = False

        self.image_item = None

        root.title(
            "Chemical Document Quick Viewer + YOLO OCR"
        )

        try:

            root.state(
                "zoomed"
            )

        except Exception:

            screen_w = (
                root.winfo_screenwidth()
            )

            screen_h = (
                root.winfo_screenheight()
            )

            root.geometry(
                f"{int(screen_w * 0.95)}x"
                f"{int(screen_h * 0.90)}"
            )

        root.minsize(
            950,
            600
        )

        self._build_gui()


    # ========================================================
    # BUILD GUI
    # ========================================================

    def _build_gui(self):

        toolbar = ttk.Frame(
            self.root,
            padding=5
        )

        toolbar.pack(
            fill=tk.X
        )

        self.open_button = ttk.Button(

            toolbar,

            text="Open PDF / image",

            command=self.upload_file
        )

        self.open_button.pack(
            side=tk.LEFT,
            padx=2
        )

        self.run_button = ttk.Button(

            toolbar,

            text="Run YOLO + OCR",

            command=self.start_inspection
        )

        self.run_button.pack(
            side=tk.LEFT,
            padx=2
        )

        ttk.Separator(
            toolbar,
            orient=tk.VERTICAL
        ).pack(
            side=tk.LEFT,
            fill=tk.Y,
            padx=8
        )

        self.previous_button = ttk.Button(

            toolbar,

            text="◀ Previous",

            command=lambda:
                self.change_page(-1)
        )

        self.previous_button.pack(
            side=tk.LEFT,
            padx=2
        )

        self.next_button = ttk.Button(

            toolbar,

            text="Next ▶",

            command=lambda:
                self.change_page(1)
        )

        self.next_button.pack(
            side=tk.LEFT,
            padx=2
        )

        self.page_label = ttk.Label(

            toolbar,

            text="Page 0 / 0"
        )

        self.page_label.pack(
            side=tk.LEFT,
            padx=8
        )

        ttk.Separator(
            toolbar,
            orient=tk.VERTICAL
        ).pack(
            side=tk.LEFT,
            fill=tk.Y,
            padx=8
        )

        ttk.Button(

            toolbar,

            text="Fit Page",

            command=self.fit_page
        ).pack(
            side=tk.LEFT,
            padx=2
        )

        ttk.Button(

            toolbar,

            text="Fit Width",

            command=self.fit_width
        ).pack(
            side=tk.LEFT,
            padx=2
        )

        ttk.Button(

            toolbar,

            text="100%",

            command=self.reset_zoom
        ).pack(
            side=tk.LEFT,
            padx=2
        )

        ttk.Button(

            toolbar,

            text="−",

            width=3,

            command=lambda:
                self.zoom_button(
                    1 / 1.20
                )
        ).pack(
            side=tk.LEFT,
            padx=1
        )

        ttk.Button(

            toolbar,

            text="+",

            width=3,

            command=lambda:
                self.zoom_button(
                    1.20
                )
        ).pack(
            side=tk.LEFT,
            padx=1
        )

        self.zoom_label = ttk.Label(

            toolbar,

            text="100%"
        )

        self.zoom_label.pack(
            side=tk.LEFT,
            padx=8
        )

        ttk.Separator(
            toolbar,
            orient=tk.VERTICAL
        ).pack(
            side=tk.LEFT,
            fill=tk.Y,
            padx=8
        )

        self.overlay_button = ttk.Button(

            toolbar,

            text="Hide OCR Overlay",

            command=self.toggle_overlay
        )

        self.overlay_button.pack(
            side=tk.LEFT,
            padx=2
        )

        self.status_label = ttk.Label(

            toolbar,

            text="Ready",

            foreground="#174ea6"
        )

        self.status_label.pack(
            side=tk.RIGHT,
            padx=5
        )

        self.file_label = ttk.Label(

            self.root,

            text="No document loaded",

            padding=(
                8,
                2,
                8,
                4
            )
        )

        self.file_label.pack(
            fill=tk.X
        )

        self.main_pane = ttk.PanedWindow(

            self.root,

            orient=tk.HORIZONTAL
        )

        self.main_pane.pack(

            fill=tk.BOTH,

            expand=True,

            padx=5,

            pady=5
        )

        image_frame = ttk.Frame(
            self.main_pane
        )

        result_frame = ttk.Frame(
            self.main_pane,
            width=430
        )

        self.main_pane.add(
            image_frame,
            weight=5
        )

        self.main_pane.add(
            result_frame,
            weight=1
        )


        # ====================================================
        # DOCUMENT CANVAS
        # ====================================================

        self.canvas = tk.Canvas(

            image_frame,

            bg="#383838",

            highlightthickness=0,

            cursor="fleur"
        )

        x_scroll = ttk.Scrollbar(

            image_frame,

            orient=tk.HORIZONTAL,

            command=self.canvas.xview
        )

        y_scroll = ttk.Scrollbar(

            image_frame,

            orient=tk.VERTICAL,

            command=self.canvas.yview
        )

        self.canvas.configure(

            xscrollcommand=x_scroll.set,

            yscrollcommand=y_scroll.set
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
        # MOUSE CONTROLS
        # ====================================================

        self.canvas.bind(
            "<MouseWheel>",
            self.mouse_zoom
        )

        self.canvas.bind(
            "<Shift-MouseWheel>",
            self.horizontal_scroll
        )

        self.canvas.bind(
            "<ButtonPress-1>",
            self.start_pan
        )

        self.canvas.bind(
            "<B1-Motion>",
            self.drag_pan
        )

        self.canvas.bind(
            "<ButtonPress-2>",
            self.start_pan
        )

        self.canvas.bind(
            "<B2-Motion>",
            self.drag_pan
        )

        self.canvas.bind(
            "<Double-Button-1>",
            lambda event:
                self.fit_page()
        )


        # ====================================================
        # RESULT PANEL
        # ====================================================

        ttk.Label(

            result_frame,

            text="YOLO / OCR detections",

            font=(
                "Segoe UI",
                10,
                "bold"
            )

        ).pack(

            anchor="w",

            pady=(
                2,
                5
            )
        )

        columns = (

            "no",

            "class",

            "raw",

            "final",

            "det_conf",

            "ocr_conf",

            "warning"
        )

        table_container = ttk.Frame(
            result_frame
        )

        table_container.pack(
            fill=tk.BOTH,
            expand=True
        )

        self.table = ttk.Treeview(

            table_container,

            columns=columns,

            show="headings"
        )

        headings = {

            "no":
                "#",

            "class":
                "Class",

            "raw":
                "Raw OCR",

            "final":
                "Final",

            "det_conf":
                "YOLO",

            "ocr_conf":
                "OCR",

            "warning":
                "!"
        }

        widths = {

            "no":
                35,

            "class":
                105,

            "raw":
                90,

            "final":
                90,

            "det_conf":
                50,

            "ocr_conf":
                50,

            "warning":
                30
        }

        for column in columns:

            self.table.heading(

                column,

                text=headings[
                    column
                ]
            )

            self.table.column(

                column,

                width=widths[
                    column
                ],

                minwidth=30,

                anchor="w"
            )

        table_y_scroll = ttk.Scrollbar(

            table_container,

            orient=tk.VERTICAL,

            command=self.table.yview
        )

        table_x_scroll = ttk.Scrollbar(

            table_container,

            orient=tk.HORIZONTAL,

            command=self.table.xview
        )

        self.table.configure(

            yscrollcommand=
                table_y_scroll.set,

            xscrollcommand=
                table_x_scroll.set
        )

        self.table.grid(

            row=0,

            column=0,

            sticky="nsew"
        )

        table_y_scroll.grid(

            row=0,

            column=1,

            sticky="ns"
        )

        table_x_scroll.grid(

            row=1,

            column=0,

            sticky="ew"
        )

        table_container.rowconfigure(
            0,
            weight=1
        )

        table_container.columnconfigure(
            0,
            weight=1
        )

        self.table.bind(

            "<<TreeviewSelect>>",

            self.select_detection
        )

        ttk.Label(

            result_frame,

            text=(
                "Wheel: zoom   |   "
                "Drag: move page   |   "
                "Shift+wheel: horizontal"
            ),

            foreground="#666666",

            padding=(0, 5)

        ).pack(
            anchor="w"
        )

        self._set_navigation_state()


    # ========================================================
    # STATUS
    # ========================================================

    def set_status(
        self,
        text,
        colour="#174ea6"
    ):

        self.status_label.config(

            text=text,

            foreground=colour
        )


    # ========================================================
    # OPEN DOCUMENT
    # ========================================================

    def upload_file(self):

        path = filedialog.askopenfilename(

            title=(
                "Select chemical document"
            ),

            filetypes=[

                (
                    "PDF and images",

                    "*.pdf "
                    "*.png "
                    "*.jpg "
                    "*.jpeg "
                    "*.bmp "
                    "*.tif "
                    "*.tiff"
                ),

                (
                    "PDF",
                    "*.pdf"
                ),

                (
                    "Images",
                    "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"
                ),
            ]
        )

        if not path:

            return

        self.file_path = Path(
            path
        )

        self.current_page = 0

        try:

            if (
                self.file_path
                .suffix
                .lower()
                == ".pdf"
            ):

                with pymupdf.open(
                    str(
                        self.file_path
                    )
                ) as document:

                    self.pdf_pages = (
                        document.page_count
                    )

            else:

                self.pdf_pages = 1

            self.load_current_page()

            self.file_label.config(

                text=str(
                    self.file_path
                )
            )

            self.set_status(
                "Document loaded."
            )

        except Exception as error:

            self.file_path = None

            messagebox.showerror(

                "Open failed",

                str(
                    error
                )
            )


    # ========================================================
    # LOAD PAGE
    # ========================================================

    def load_current_page(self):

        if not self.file_path:

            return

        if (
            self.file_path
            .suffix
            .lower()
            == ".pdf"
        ):

            self.current_image = (
                pdf_page_to_rgb(

                    self.file_path,

                    self.current_page
                )
            )

        else:

            self.current_image = (
                load_image_rgb(
                    self.file_path
                )
            )

        self.annotated_image = None

        self.detections = []

        self.clear_table()

        self.page_label.config(

            text=(
                f"Page "
                f"{self.current_page + 1}"
                f" / "
                f"{self.pdf_pages}"
            )
        )

        self._set_navigation_state()

        self.root.after(
            80,
            self.fit_page
        )


    # ========================================================
    # CHANGE PAGE
    # ========================================================

    def change_page(
        self,
        direction
    ):

        if (
            self.busy
            or not self.file_path
        ):

            return

        new_page = (
            self.current_page
            + direction
        )

        if (
            0
            <= new_page
            < self.pdf_pages
        ):

            self.current_page = (
                new_page
            )

            try:

                self.load_current_page()

                self.set_status(
                    "Page loaded."
                )

            except Exception as error:

                messagebox.showerror(

                    "Page error",

                    str(
                        error
                    )
                )


    # ========================================================
    # NAVIGATION STATE
    # ========================================================

    def _set_navigation_state(self):

        previous = (

            tk.NORMAL

            if (
                self.file_path
                and self.current_page > 0
                and not self.busy
            )

            else tk.DISABLED
        )

        following = (

            tk.NORMAL

            if (
                self.file_path
                and
                self.current_page
                < self.pdf_pages - 1
                and not self.busy
            )

            else tk.DISABLED
        )

        self.previous_button.config(
            state=previous
        )

        self.next_button.config(
            state=following
        )


    # ========================================================
    # TOGGLE OVERLAY
    # ========================================================

    def toggle_overlay(self):

        global SHOW_OVERLAY

        SHOW_OVERLAY = (
            not SHOW_OVERLAY
        )

        if SHOW_OVERLAY:

            self.overlay_button.config(
                text="Hide OCR Overlay"
            )

        else:

            self.overlay_button.config(
                text="Show OCR Overlay"
            )

        image = (
            self.get_display_image()
        )

        if image is not None:

            self.show_image(
                image,
                keep_view=True
            )


    # ========================================================
    # START YOLO + OCR
    # ========================================================

    def start_inspection(self):

        if self.current_image is None:

            messagebox.showwarning(

                "No document",

                "Open a PDF or image first."
            )

            return

        if self.busy:

            return

        self.busy = True

        self.open_button.config(
            state=tk.DISABLED
        )

        self.run_button.config(
            state=tk.DISABLED
        )

        self._set_navigation_state()

        self.set_status(
            "Running YOLO..."
        )

        image = (
            self.current_image.copy()
        )

        threading.Thread(

            target=self._process_image,

            args=(image,),

            daemon=True

        ).start()


    # ========================================================
    # PROCESS IMAGE
    # ========================================================

    def _process_image(
        self,
        image
    ):

        try:

            result = self.model.predict(

                source=image,

                imgsz=IMAGE_SIZE,

                conf=CONFIDENCE,

                iou=IOU_THRESHOLD,

                max_det=MAX_DETECTIONS,

                device=DEVICE,

                verbose=False

            )[0]

            detections = []

            if (
                result.boxes
                is not None
            ):

                for box in result.boxes:

                    class_id = int(

                        box.cls[0]
                        .cpu()
                        .item()
                    )

                    coordinates = (

                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .round()
                        .astype(int)
                    )

                    (
                        x1,
                        y1,
                        x2,
                        y2
                    ) = coordinates

                    x1 = max(
                        0,
                        x1
                    )

                    y1 = max(
                        0,
                        y1
                    )

                    x2 = min(
                        image.shape[1],
                        x2
                    )

                    y2 = min(
                        image.shape[0],
                        y2
                    )

                    detections.append({

                        "class_id":
                            class_id,

                        "class_name":
                            str(
                                self.model.names[
                                    class_id
                                ]
                            ),

                        "det_conf":
                            float(
                                box.conf[0]
                                .cpu()
                                .item()
                            ),

                        "x1":
                            x1,

                        "y1":
                            y1,

                        "x2":
                            x2,

                        "y2":
                            y2,

                        "cx":
                            (
                                x1 + x2
                            ) / 2.0,

                        "cy":
                            (
                                y1 + y2
                            ) / 2.0,
                    })

            detections = (
                sort_reading_order(
                    detections
                )
            )

            total = len(
                detections
            )

            for index, detection in enumerate(

                detections,

                start=1
            ):

                if (
                    index == 1
                    or index % 10 == 0
                    or index == total
                ):

                    self.root.after(

                        0,

                        self.set_status,

                        f"OCR "
                        f"{index}"
                        f" / "
                        f"{total}"
                    )

                (
                    final_text,
                    ocr_conf,
                    mapped,
                    raw_text

                ) = read_crop_text(

                    self.reader,

                    image,

                    detection
                )

                detection[
                    "text"
                ] = final_text

                detection[
                    "raw_text"
                ] = raw_text

                detection[
                    "ocr_conf"
                ] = ocr_conf

                detection[
                    "mapped"
                ] = mapped

                detection[
                    "warning"
                ] = (

                    not final_text

                    or
                    ocr_conf
                    < OCR_MIN_CONFIDENCE

                    or
                    not mapped
                )

            overlay = draw_overlay(

                image,

                detections
            )

            self.root.after(

                0,

                self._inspection_complete,

                detections,

                overlay
            )

        except Exception as error:

            self.root.after(

                0,

                self._inspection_failed,

                str(
                    error
                )
            )


    # ========================================================
    # COMPLETE
    # ========================================================

    def _inspection_complete(
        self,
        detections,
        overlay
    ):

        self.detections = (
            detections
        )

        self.annotated_image = (
            overlay
        )

        self.clear_table()

        for index, detection in enumerate(

            detections,

            start=1
        ):

            warning = (
                "⚠"
                if detection[
                    "warning"
                ]
                else ""
            )

            self.table.insert(

                "",

                tk.END,

                iid=str(
                    index - 1
                ),

                values=(

                    index,

                    detection[
                        "class_name"
                    ],

                    detection.get(
                        "raw_text",
                        ""
                    ),

                    detection.get(
                        "text",
                        ""
                    ),

                    f'{detection["det_conf"]:.0%}',

                    f'{detection.get("ocr_conf", 0):.0%}',

                    warning
                )
            )

        self.fit_page()

        self.busy = False

        self.open_button.config(
            state=tk.NORMAL
        )

        self.run_button.config(
            state=tk.NORMAL
        )

        self._set_navigation_state()

        warnings = sum(

            bool(
                d["warning"]
            )

            for d
            in detections
        )

        recognised = sum(

            bool(
                d.get(
                    "text"
                )
            )

            for d
            in detections
        )

        self.set_status(

            f"{len(detections)} boxes | "
            f"{recognised} OCR | "
            f"{warnings} warnings",

            "#137333"
        )


    # ========================================================
    # FAILED
    # ========================================================

    def _inspection_failed(
        self,
        error
    ):

        self.busy = False

        self.open_button.config(
            state=tk.NORMAL
        )

        self.run_button.config(
            state=tk.NORMAL
        )

        self._set_navigation_state()

        self.set_status(

            "Processing failed.",

            "#b3261e"
        )

        messagebox.showerror(

            "YOLO / OCR error",

            error
        )


    # ========================================================
    # CLEAR TABLE
    # ========================================================

    def clear_table(self):

        for item in (
            self.table.get_children()
        ):

            self.table.delete(
                item
            )


    # ========================================================
    # DISPLAY IMAGE SOURCE
    # ========================================================

    def get_display_image(self):

        if (
            SHOW_OVERLAY
            and self.annotated_image
            is not None
        ):

            return self.annotated_image

        return self.current_image


    # ========================================================
    # SHOW IMAGE
    # ========================================================

    def show_image(
        self,
        image,
        keep_view=False
    ):

        if image is None:

            return

        old_x = (
            self.canvas.xview()
        )

        old_y = (
            self.canvas.yview()
        )

        height, width = (
            image.shape[:2]
        )

        shown_width = max(

            1,

            int(
                width
                * self.zoom
            )
        )

        shown_height = max(

            1,

            int(
                height
                * self.zoom
            )
        )

        interpolation = (

            cv2.INTER_AREA

            if self.zoom < 1.0

            else cv2.INTER_CUBIC
        )

        shown = cv2.resize(

            image,

            (
                shown_width,
                shown_height
            ),

            interpolation=interpolation
        )

        self.tk_image = (

            ImageTk.PhotoImage(
                Image.fromarray(
                    shown
                )
            )
        )

        self.canvas.delete(
            "all"
        )

        self.image_item = (
            self.canvas.create_image(

                0,

                0,

                anchor=tk.NW,

                image=self.tk_image
            )
        )

        self.canvas.configure(

            scrollregion=(

                0,

                0,

                shown_width,

                shown_height
            )
        )

        self.zoom_label.config(

            text=(
                f"{self.zoom * 100:.0f}%"
            )
        )

        if keep_view:

            if old_x:

                self.canvas.xview_moveto(
                    old_x[0]
                )

            if old_y:

                self.canvas.yview_moveto(
                    old_y[0]
                )


    # ========================================================
    # FIT PAGE
    # ========================================================

    def fit_page(self):

        image = (
            self.get_display_image()
        )

        if image is None:

            return

        self.root.update_idletasks()

        available_width = max(

            100,

            self.canvas.winfo_width()
            - 20
        )

        available_height = max(

            100,

            self.canvas.winfo_height()
            - 20
        )

        image_height, image_width = (
            image.shape[:2]
        )

        self.zoom = min(

            available_width
            / image_width,

            available_height
            / image_height
        )

        self.zoom = max(

            self.min_zoom,

            min(
                self.zoom,
                self.max_zoom
            )
        )

        self.show_image(
            image
        )

        self.canvas.xview_moveto(
            0
        )

        self.canvas.yview_moveto(
            0
        )


    # ========================================================
    # FIT WIDTH
    # ========================================================

    def fit_width(self):

        image = (
            self.get_display_image()
        )

        if image is None:

            return

        self.root.update_idletasks()

        available_width = max(

            100,

            self.canvas.winfo_width()
            - 20
        )

        image_width = (
            image.shape[1]
        )

        self.zoom = (

            available_width
            / image_width
        )

        self.zoom = max(

            self.min_zoom,

            min(
                self.zoom,
                self.max_zoom
            )
        )

        self.show_image(
            image
        )

        self.canvas.xview_moveto(
            0
        )

        self.canvas.yview_moveto(
            0
        )


    # ========================================================
    # 100%
    # ========================================================

    def reset_zoom(self):

        image = (
            self.get_display_image()
        )

        if image is None:

            return

        self.zoom = 1.0

        self.show_image(
            image
        )


    # ========================================================
    # BUTTON ZOOM
    # ========================================================

    def zoom_button(
        self,
        factor
    ):

        image = (
            self.get_display_image()
        )

        if image is None:

            return

        new_zoom = (

            self.zoom
            * factor
        )

        new_zoom = max(

            self.min_zoom,

            min(
                self.max_zoom,
                new_zoom
            )
        )

        self.zoom = new_zoom

        self.show_image(

            image,

            keep_view=True
        )


    # ========================================================
    # MOUSE ZOOM
    # ========================================================

    def mouse_zoom(
        self,
        event
    ):

        image = (
            self.get_display_image()
        )

        if image is None:

            return "break"

        canvas_x = (
            self.canvas.canvasx(
                event.x
            )
        )

        canvas_y = (
            self.canvas.canvasy(
                event.y
            )
        )

        old_zoom = (
            self.zoom
        )

        if event.delta > 0:

            factor = 1.20

        else:

            factor = (
                1 / 1.20
            )

        new_zoom = (

            old_zoom
            * factor
        )

        new_zoom = max(

            self.min_zoom,

            min(
                self.max_zoom,
                new_zoom
            )
        )

        if abs(
            new_zoom
            - old_zoom
        ) < 0.00001:

            return "break"

        image_x = (
            canvas_x
            / old_zoom
        )

        image_y = (
            canvas_y
            / old_zoom
        )

        self.zoom = (
            new_zoom
        )

        self.show_image(
            image
        )

        new_canvas_x = (
            image_x
            * new_zoom
        )

        new_canvas_y = (
            image_y
            * new_zoom
        )

        bbox = (
            self.canvas.bbox(
                "all"
            )
        )

        if bbox:

            total_width = max(
                1,
                bbox[2]
                - bbox[0]
            )

            total_height = max(
                1,
                bbox[3]
                - bbox[1]
            )

            desired_left = (

                new_canvas_x
                - event.x
            )

            desired_top = (

                new_canvas_y
                - event.y
            )

            self.canvas.xview_moveto(

                max(
                    0.0,

                    min(
                        1.0,

                        desired_left
                        / total_width
                    )
                )
            )

            self.canvas.yview_moveto(

                max(
                    0.0,

                    min(
                        1.0,

                        desired_top
                        / total_height
                    )
                )
            )

        return "break"


    # ========================================================
    # HORIZONTAL SCROLL
    # ========================================================

    def horizontal_scroll(
        self,
        event
    ):

        direction = (

            -3

            if event.delta > 0

            else 3
        )

        self.canvas.xview_scroll(

            direction,

            "units"
        )

        return "break"


    # ========================================================
    # PAN
    # ========================================================

    def start_pan(
        self,
        event
    ):

        self.canvas.scan_mark(

            event.x,

            event.y
        )


    def drag_pan(
        self,
        event
    ):

        self.canvas.scan_dragto(

            event.x,

            event.y,

            gain=1
        )


    # ========================================================
    # SELECT RESULT
    # ========================================================

    def select_detection(
        self,
        _event=None
    ):

        selection = (
            self.table.selection()
        )

        if (
            not selection
            or not self.detections
        ):

            return

        index = int(
            selection[0]
        )

        detection = (
            self.detections[
                index
            ]
        )

        if self.zoom < 0.65:

            self.zoom = 0.90

            self.show_image(
                self.get_display_image()
            )

        target_x = (
            detection["cx"]
            * self.zoom
        )

        target_y = (
            detection["cy"]
            * self.zoom
        )

        bbox = (
            self.canvas.bbox(
                "all"
            )
        )

        if not bbox:

            return

        total_width = max(

            1,

            bbox[2]
        )

        total_height = max(

            1,

            bbox[3]
        )

        view_width = (
            self.canvas.winfo_width()
        )

        view_height = (
            self.canvas.winfo_height()
        )

        x_fraction = (

            target_x
            - view_width / 2

        ) / total_width

        y_fraction = (

            target_y
            - view_height / 2

        ) / total_height

        self.canvas.xview_moveto(

            max(
                0,

                min(
                    1,
                    x_fraction
                )
            )
        )

        self.canvas.yview_moveto(

            max(
                0,

                min(
                    1,
                    y_fraction
                )
            )
        )


# ============================================================
# VERIFY MODEL
# ============================================================

def verify_model_classes(
    model
):

    actual = {

        int(key):
            str(value)

        for key, value
        in model.names.items()
    }

    if actual != EXPECTED_CLASSES:

        raise ValueError(

            "Loaded YOLO model does not match "
            "the expected six-class structure.\n\n"

            f"EXPECTED:\n"
            f"{EXPECTED_CLASSES}\n\n"

            f"ACTUAL:\n"
            f"{actual}\n\n"

            f"Model:\n"
            f"{MODEL_PATH}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "LOADING YOLO MODEL"
    )

    print("=" * 70)

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Device:",
        DEVICE
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(

            f"YOLO model not found:\n"
            f"{MODEL_PATH}"
        )

    model = YOLO(
        str(
            MODEL_PATH
        )
    )

    verify_model_classes(
        model
    )

    print()
    print(
        "YOLO loaded successfully."
    )

    print(
        "Model classes:"
    )

    print("-" * 70)

    for class_id, class_name in (
        model.names.items()
    ):

        print(

            f"{class_id}: "
            f"{class_name}"
        )

    print("-" * 70)

    print()
    print("=" * 70)

    print(
        "LOADING EASYOCR"
    )

    print("=" * 70)

    gpu_available = (
        torch.cuda.is_available()
    )

    print(
        "CUDA available:",
        gpu_available
    )

    print(
        "EasyOCR GPU    :",
        gpu_available
    )

    reader = easyocr.Reader(

        ["en"],

        gpu=gpu_available
    )

    print()
    print(
        "EasyOCR loaded successfully."
    )

    root = tk.Tk()

    app = OCRInspectionTool(

        root,

        model,

        reader
    )

    root.mainloop()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
