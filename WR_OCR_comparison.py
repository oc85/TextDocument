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
# - Separate YOLO, YOLO colours-only and YOLO + OCR buttons
# - EasyOCR loaded only when YOLO + OCR is requested
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
# 4 table             (YOLO only - no OCR)
# 5 headers           (OCR applied in YOLO + OCR mode)
#
# ============================================================

import re
import threading
import time
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
    r"D:\Kaggle\WRtools_005\best.pt"
)

IMAGE_SIZE = 1280

CONFIDENCE = 0.03

# Standard same-class NMS threshold used by YOLO.
# Cross-class overlap is handled separately below.
IOU_THRESHOLD = 0.45

# Non-table categories should not overlap by more than 5%.
# The table category is excluded because it contains the others.
NON_TABLE_MAX_OVERLAP = 0.05

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
    4: "table",
    5: "headers",
}


# OCR is used for text/token classes and header regions.
# table remains a structural YOLO detection only.
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
    "s1": "Si",
    "s|": "Si",

    # Tin
    "sn": "Sn",

    # Tantalum
    "ta": "Ta",

    # Titanium
    "ti": "Ti",

    # Thallium
    "tl": "Tl",

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
    # Complete current periodic table. Using the complete set
    # prevents correctly read symbols such as Se from being
    # rejected merely because they were absent from a short
    # project-specific list.
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
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

# A chemical-symbol candidate must be strictly above 10%.
SYMBOL_MIN_CONFIDENCE = 0.10


# Valid visual-confusion fallbacks used only when the same
# element symbol appears more than once inside one table.
# Every fallback must itself be a real chemical element.
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

# Higher value gives a more opaque, easier-to-read label box.
OVERLAY_LABEL_ALPHA = 0.82

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

    "table":
        (0, 190, 210),

    "headers":
        (90, 95, 230),
}


CLASS_SHORT = {

    "element_symbol": "E",

    "element_value": "V",

    "unit": "U",

    "limit_indicator": "L",

    "table": "T",

    "headers": "H",
}


def rgb_to_hex(
    colour
):

    return "#{:02x}{:02x}{:02x}".format(
        int(colour[0]),
        int(colour[1]),
        int(colour[2])
    )


def format_duration(
    seconds
):

    seconds = max(
        0.0,
        float(seconds)
    )

    minutes = int(
        seconds // 60
    )

    remaining_seconds = (
        seconds
        - minutes * 60
    )

    return (
        f"{minutes:02d}m "
        f"{remaining_seconds:05.2f}s"
    )


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

    # Numeric ranges belong to element_value in the current
    # six-class model.  EasyOCR's numeric pass may return the
    # complete range, for example "26.0 - 30.0".  Keep both
    # numbers and standardise the separator for display.
    range_match = re.fullmatch(

        r"([+-]?\d+(?:\.\d+)?)"
        r"-"
        r"([+-]?\d+(?:\.\d+)?)",

        value
    )

    if range_match:

        return (
            f"{range_match.group(1)} - "
            f"{range_match.group(2)}",
            True
        )

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
# NORMALISE HEADER
# ============================================================

def normalise_header(text):

    original = clean_text(
        text
    )

    if not original:

        return "", False

    # EasyOCR sometimes attaches one or two narrow characters
    # from a nearby table border/cell to the word "Analysis".
    # Correct only that short leading artefact; do not alter
    # legitimate longer headers such as "Chemical Analysis".
    compact = re.sub(
        r"[^a-z]",
        "",
        original.lower()
    )

    if compact.endswith(
        "analysis"
    ):

        prefix = compact[
            :-len("analysis")
        ]

        if len(prefix) <= 2:

            return "Analysis", True

    return original, True


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
                or
                re.fullmatch(
                    r"[+-]?\d+(?:\.\d+)?"
                    r"\s+-\s+"
                    r"[+-]?\d+(?:\.\d+)?",
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

                # Prefer a complete numeric range over an OCR
                # candidate containing only one endpoint.
                bool(
                    re.fullmatch(
                        r"[+-]?\d+(?:\.\d+)?"
                        r"\s+-\s+"
                        r"[+-]?\d+(?:\.\d+)?",
                        item["final"]
                    )
                ),

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

    # The complete table region is detected by YOLO only.
    # Headers are allowed through and are read by EasyOCR.
    if class_name in NO_OCR_CLASSES:

        return (
            "",
            0.0,
            True,
            ""
        )

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

    # --------------------------------------------------------
    # ELEMENT SYMBOL CANDIDATES
    #
    # Keep the best confidence for every valid alternative.
    # These alternatives are used later if the same symbol is
    # recognised more than once inside one detected table.
    # --------------------------------------------------------

    if class_name == "element_symbol":

        best_by_symbol = {}

        for candidate in candidates:

            if (
                float(candidate["confidence"])
                <= SYMBOL_MIN_CONFIDENCE
            ):

                continue

            candidate_text, candidate_valid = (
                normalise_element(
                    candidate["raw"]
                )
            )

            if not candidate_valid:

                continue

            previous = best_by_symbol.get(
                candidate_text
            )

            if (
                previous is None
                or candidate["confidence"]
                > previous["confidence"]
            ):

                best_by_symbol[candidate_text] = {
                    "text": candidate_text,
                    "raw": candidate["raw"],
                    "confidence": float(
                        candidate["confidence"]
                    ),
                }

        symbol_alternatives = sorted(
            best_by_symbol.values(),
            key=lambda item:
                item["confidence"],
            reverse=True
        )

        detection["ocr_alternatives"] = (
            symbol_alternatives
        )

        if symbol_alternatives:

            best_symbol = symbol_alternatives[0]

            return (
                best_symbol["text"],
                best_symbol["confidence"],
                True,
                best_symbol["raw"]
            )

        # Candidates existed, but none was both a valid element
        # and above the required 10% confidence threshold.
        rejected_best = max(
            candidates,
            key=lambda item:
                item["confidence"]
        )

        return (
            "",
            float(rejected_best["confidence"]),
            False,
            rejected_best["raw"]
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

    elif class_name == "headers":

        final_text, mapped = (
            normalise_header(
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
# RESOLVE DUPLICATE ELEMENT SYMBOLS INSIDE EACH TABLE
#
# Example:
# - two boxes initially read as Ti
# - the higher-confidence Ti remains Ti
# - the lower-confidence duplicate can use its next valid
#   unused candidate, for example Tl
# ============================================================

def containing_table_key(
    detection,
    tables
):

    matching_tables = []

    for table_index, table in enumerate(tables):

        if (
            table["x1"] <= detection["cx"] <= table["x2"]
            and table["y1"] <= detection["cy"] <= table["y2"]
        ):

            table_area = max(
                1,
                (table["x2"] - table["x1"])
                * (table["y2"] - table["y1"])
            )

            matching_tables.append({
                "confidence": float(
                    table.get("det_conf", 0.0)
                ),
                "area": table_area,
                "index": table_index,
            })

    if not matching_tables:

        # Symbols outside a detected table are treated as one
        # page-level group rather than being discarded.
        return "page"

    # Prefer the most confident containing table. This avoids
    # assigning two symbols to different nested duplicate table
    # boxes merely because one table box is slightly smaller.
    best_table = max(
        matching_tables,
        key=lambda item: (
            item["confidence"],
            item["area"]
        )
    )

    return best_table["index"]


def resolve_duplicate_element_symbols(
    detections
):

    raw_tables = [
        detection
        for detection in detections
        if detection["class_name"] == "table"
    ]

    # Collapse nested or near-duplicate detections of the same
    # physical table. Keep the highest-confidence table box.
    raw_tables.sort(
        key=lambda detection:
            detection.get("det_conf", 0.0),
        reverse=True
    )

    tables = []

    for table in raw_tables:

        represents_existing_table = any(
            smaller_box_overlap_ratio(
                table,
                existing_table
            ) >= 0.60
            for existing_table in tables
        )

        if not represents_existing_table:

            tables.append(
                table
            )

    symbols_by_table = {}

    for detection in detections:

        if (
            detection["class_name"] != "element_symbol"
            or not detection.get("text")
        ):

            continue

        table_key = containing_table_key(
            detection,
            tables
        )

        symbols_by_table.setdefault(
            table_key,
            []
        ).append(
            detection
        )

    replacements = 0

    for table_symbols in symbols_by_table.values():

        used_symbols = {
            detection["text"]
            for detection in table_symbols
            if detection.get("text")
        }

        symbols_by_text = {}

        for detection in table_symbols:

            symbols_by_text.setdefault(
                detection["text"],
                []
            ).append(
                detection
            )

        for duplicated_symbol, duplicates in (
            symbols_by_text.items()
        ):

            if len(duplicates) <= 1:

                continue

            duplicates.sort(
                key=lambda detection:
                    detection.get("ocr_conf", 0.0),
                reverse=True
            )

            # The strongest occurrence keeps the original
            # symbol. Resolve each weaker duplicate in order.
            for duplicate in duplicates[1:]:

                replacement = None

                preferred_fallbacks = (
                    DUPLICATE_SYMBOL_FALLBACKS.get(
                        duplicated_symbol,
                        []
                    )
                )

                # For a known visual pair such as Ti/Tl, prefer
                # the configured counterpart when OCR supplied
                # it as an alternative candidate.
                for fallback_text in preferred_fallbacks:

                    for alternative in duplicate.get(
                        "ocr_alternatives",
                        []
                    ):

                        if (
                            alternative["text"] == fallback_text
                            and fallback_text not in used_symbols
                            and alternative["confidence"]
                            > SYMBOL_MIN_CONFIDENCE
                        ):

                            replacement = alternative
                            break

                    if replacement is not None:

                        break

                if replacement is None:

                    for alternative in duplicate.get(
                        "ocr_alternatives",
                        []
                    ):

                        alternative_text = alternative["text"]

                        if (
                            alternative_text != duplicated_symbol
                            and alternative_text not in used_symbols
                            and alternative["confidence"]
                            > SYMBOL_MIN_CONFIDENCE
                        ):

                            replacement = alternative
                            break

                if replacement is None:

                    # If OCR did not expose a separate candidate,
                    # use a configured valid chemical-symbol
                    # fallback for a known visual ambiguity.
                    for fallback_text in preferred_fallbacks:

                        if fallback_text in used_symbols:

                            continue

                        inferred_confidence = max(
                            SYMBOL_MIN_CONFIDENCE + 0.001,
                            float(
                                duplicate.get(
                                    "ocr_conf",
                                    SYMBOL_MIN_CONFIDENCE + 0.001
                                )
                            ) * 0.85
                        )

                        replacement = {
                            "text": fallback_text,
                            "raw": duplicate.get(
                                "raw_text",
                                duplicated_symbol
                            ),
                            "confidence": inferred_confidence,
                            "inferred": True,
                        }

                        break

                if replacement is None:

                    # Keep the duplicate visible, but flag it
                    # because no safe distinct alternative was
                    # found in the OCR candidate list.
                    duplicate["warning"] = True
                    duplicate["duplicate_symbol"] = True
                    continue

                duplicate["duplicate_resolved_from"] = (
                    duplicate["text"]
                )

                duplicate["text"] = replacement["text"]
                duplicate["raw_text"] = replacement["raw"]
                duplicate["ocr_conf"] = replacement["confidence"]
                duplicate["mapped"] = True
                duplicate["inferred_symbol"] = replacement.get(
                    "inferred",
                    False
                )
                duplicate["warning"] = (
                    replacement["confidence"]
                    < OCR_MIN_CONFIDENCE
                    or duplicate["inferred_symbol"]
                )

                used_symbols.add(
                    replacement["text"]
                )

                replacements += 1

    return replacements


# ============================================================
# NON-TABLE OVERLAP FILTER
#
# The table region is allowed to contain every other class.
# For every other category, overlap greater than 5% of the
# smaller box is treated as a conflict. The higher-confidence
# detection is kept.
# ============================================================

def smaller_box_overlap_ratio(
    first,
    second
):

    intersection_x1 = max(
        first["x1"],
        second["x1"]
    )

    intersection_y1 = max(
        first["y1"],
        second["y1"]
    )

    intersection_x2 = min(
        first["x2"],
        second["x2"]
    )

    intersection_y2 = min(
        first["y2"],
        second["y2"]
    )

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    if intersection_area <= 0:

        return 0.0

    first_area = max(
        1,
        (first["x2"] - first["x1"])
        * (first["y2"] - first["y1"])
    )

    second_area = max(
        1,
        (second["x2"] - second["x1"])
        * (second["y2"] - second["y1"])
    )

    return (
        intersection_area
        / min(first_area, second_area)
    )


def filter_non_table_overlaps(
    detections,
    maximum_overlap=NON_TABLE_MAX_OVERLAP
):

    table_detections = [
        detection
        for detection in detections
        if detection["class_name"] == "table"
    ]

    non_table_detections = [
        detection
        for detection in detections
        if detection["class_name"] != "table"
    ]

    # Highest-confidence candidates are considered first.
    non_table_detections.sort(
        key=lambda detection:
            detection["det_conf"],
        reverse=True
    )

    kept = []

    for candidate in non_table_detections:

        conflicts = any(
            smaller_box_overlap_ratio(
                candidate,
                existing
            ) > maximum_overlap
            for existing in kept
        )

        if not conflicts:

            kept.append(
                candidate
            )

    # Tables bypass the overlap filter and remain available
    # even when they contain all of the kept token boxes.
    return (
        table_detections
        + kept
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

    # In YOLO-only mode every detection is labelled by class,
    # without a question mark or OCR placeholder.
    if not detection.get(
        "ocr_applied",
        False
    ):

        short = CLASS_SHORT.get(
            class_name,
            "?"
        )

        if SHOW_CLASS_PREFIX:

            return (
                f"{short}: {class_name}"
            )

        return class_name

    # The table class does not have OCR text.
    # Headers reach the OCR-text section below.
    if class_name in NO_OCR_CLASSES:

        short = CLASS_SHORT.get(
            class_name,
            "?"
        )

        if SHOW_CLASS_PREFIX:

            return (
                f"{short}: {class_name}"
            )

        return class_name

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

    # YOLO + OCR mode shows only the recognised result.
    # Examples: C, 0.25, %, MAX.
    # Do not add E:, V:, U: or L: prefixes.
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
    detections,
    show_box_text=True
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

        if (
            not SHOW_OVERLAY_TEXT
            or not show_box_text
        ):

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
            "Chemical Document Viewer - YOLO / YOLO + OCR"
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

        self.yolo_button = ttk.Button(

            toolbar,

            text="Run YOLO Only",

            command=lambda:
                self.start_inspection(
                    use_ocr=False,
                    show_box_text=True
                )
        )

        self.yolo_button.pack(
            side=tk.LEFT,
            padx=2
        )

        self.colour_button = ttk.Button(

            toolbar,

            text="YOLO Colours Only",

            command=lambda:
                self.start_inspection(
                    use_ocr=False,
                    show_box_text=False
                )
        )

        self.colour_button.pack(
            side=tk.LEFT,
            padx=2
        )

        self.run_button = ttk.Button(

            toolbar,

            text="Run YOLO + OCR",

            command=lambda:
                self.start_inspection(
                    use_ocr=True,
                    show_box_text=True
                )
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

            text="Hide Detection Overlay",

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

            text="YOLO detections and OCR results",

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

        # ====================================================
        # YOLO BOX COLOUR LEGEND
        # ====================================================

        legend_frame = ttk.LabelFrame(
            result_frame,
            text="YOLO box colours",
            padding=5
        )

        legend_frame.pack(
            fill=tk.X,
            pady=(0, 7)
        )

        self.colour_legend = tk.Text(
            legend_frame,
            height=6,
            wrap=tk.NONE,
            relief=tk.FLAT,
            borderwidth=0,
            background="#f7f7f7",
            cursor="arrow",
            font=("Segoe UI", 9)
        )

        self.colour_legend.pack(
            fill=tk.X
        )

        legend_descriptions = {
            "element_symbol": "Element symbol",
            "element_value": "Element value",
            "unit": "Unit",
            "limit_indicator": "Limit indicator",
            "table": "Complete table region",
            "headers": "Header region",
        }

        for class_id, class_name in EXPECTED_CLASSES.items():

            colour = CLASS_COLOURS.get(
                class_name,
                (80, 80, 80)
            )

            tag_name = f"legend_{class_name}"

            self.colour_legend.tag_configure(
                tag_name,
                foreground=rgb_to_hex(colour),
                font=("Segoe UI", 9, "bold")
            )

            description = legend_descriptions.get(
                class_name,
                class_name
            )

            self.colour_legend.insert(
                tk.END,
                f"■  {class_id}  {description}\n",
                tag_name
            )

        self.colour_legend.config(
            state=tk.DISABLED
        )

        # ====================================================
        # PROCESSING TIME INFORMATION
        # ====================================================

        timing_frame = ttk.LabelFrame(
            result_frame,
            text="Processing time",
            padding=5
        )

        timing_frame.pack(
            fill=tk.X,
            pady=(0, 7)
        )

        self.timing_text = tk.Text(
            timing_frame,
            height=12,
            wrap=tk.NONE,
            relief=tk.FLAT,
            borderwidth=0,
            background="#f7f7f7",
            cursor="arrow",
            font=("Consolas", 8)
        )

        self.timing_text.pack(
            fill=tk.X
        )

        self.timing_text.insert(
            tk.END,
            "Run YOLO or YOLO + OCR to display timing."
        )

        self.timing_text.config(
            state=tk.DISABLED
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
    # UPDATE PROCESSING TIME PANEL
    # ========================================================

    def update_timing_display(
        self,
        timing_info
    ):

        lines = [
            f"YOLO:       {format_duration(timing_info['yolo_seconds'])}",
            f"Whole OCR:  {format_duration(timing_info['ocr_total_seconds'])}",
            f"OCR work:   {format_duration(timing_info['ocr_recognition_seconds'])}",
            f"OCR load:   {format_duration(timing_info['ocr_load_seconds'])}",
            f"Total:      {format_duration(timing_info['total_seconds'])}",
            "",
            "OCR average by category:",
        ]

        category_times = timing_info.get(
            "ocr_category_times",
            {}
        )

        category_order = [
            class_name
            for class_name in EXPECTED_CLASSES.values()
            if class_name in OCR_CLASSES
        ]

        category_found = False

        for class_name in category_order:

            category = category_times.get(
                class_name
            )

            if not category or category["count"] == 0:

                continue

            category_found = True

            average_seconds = (
                category["total_seconds"]
                / category["count"]
            )

            lines.append(
                f"{class_name:16s} "
                f"n={category['count']:3d}  "
                f"total={format_duration(category['total_seconds'])}  "
                f"avg={format_duration(average_seconds)}"
            )

        if not category_found:

            lines.append(
                "No OCR categories processed."
            )

        self.timing_text.config(
            state=tk.NORMAL
        )

        self.timing_text.delete(
            "1.0",
            tk.END
        )

        self.timing_text.insert(
            tk.END,
            "\n".join(lines)
        )

        self.timing_text.config(
            state=tk.DISABLED
        )


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
                text="Hide Detection Overlay"
            )

        else:

            self.overlay_button.config(
                text="Show Detection Overlay"
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

    def start_inspection(
        self,
        use_ocr,
        show_box_text=True
    ):

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

        self.yolo_button.config(
            state=tk.DISABLED
        )

        self.colour_button.config(
            state=tk.DISABLED
        )

        self._set_navigation_state()

        if use_ocr:

            mode_text = "YOLO + OCR"

        elif show_box_text:

            mode_text = "YOLO only"

        else:

            mode_text = "YOLO colours only"

        self.set_status(
            f"Running {mode_text}..."
        )

        image = (
            self.current_image.copy()
        )

        threading.Thread(

            target=self._process_image,

            args=(
                image,
                use_ocr,
                show_box_text
            ),

            daemon=True

        ).start()


    # ========================================================
    # PROCESS IMAGE
    # ========================================================

    def _process_image(
        self,
        image,
        use_ocr,
        show_box_text
    ):

        try:

            processing_started = time.perf_counter()

            yolo_started = time.perf_counter()

            result = self.model.predict(

                source=image,

                imgsz=IMAGE_SIZE,

                conf=CONFIDENCE,

                iou=IOU_THRESHOLD,

                max_det=MAX_DETECTIONS,

                device=DEVICE,

                # Keep YOLO NMS class-aware so the table
                # detection cannot suppress contained classes.
                agnostic_nms=False,

                verbose=False

            )[0]

            yolo_seconds = (
                time.perf_counter()
                - yolo_started
            )

            ocr_load_seconds = 0.0

            ocr_category_times = {
                class_name: {
                    "count": 0,
                    "total_seconds": 0.0,
                }
                for class_name in OCR_CLASSES
            }

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

            detections_before_filter = len(
                detections
            )

            detections = filter_non_table_overlaps(
                detections
            )

            removed_overlaps = (
                detections_before_filter
                - len(detections)
            )

            detections = (
                sort_reading_order(
                    detections
                )
            )

            total = len(
                detections
            )

            ocr_detections = [

                detection

                for detection
                in detections

                if (
                    use_ocr
                    and detection["class_name"]
                    in OCR_CLASSES
                )
            ]

            ocr_total = len(
                ocr_detections
            )

            # Load EasyOCR only when the user selects the
            # YOLO + OCR button. YOLO-only inspection starts
            # without paying the OCR loading cost.
            if (
                use_ocr
                and ocr_total > 0
                and self.reader is None
            ):

                self.root.after(
                    0,
                    self.set_status,
                    "Loading EasyOCR..."
                )

                ocr_load_started = time.perf_counter()

                self.reader = easyocr.Reader(
                    ["en"],
                    gpu=torch.cuda.is_available()
                )

                ocr_load_seconds = (
                    time.perf_counter()
                    - ocr_load_started
                )

            for index, detection in enumerate(

                ocr_detections,

                start=1
            ):

                if (
                    index == 1
                    or index % 10 == 0
                    or index == ocr_total
                ):

                    self.root.after(

                        0,

                        self.set_status,

                        f"OCR "
                        f"{index}"
                        f" / "
                        f"{ocr_total}"
                    )

                ocr_item_started = time.perf_counter()

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

                ocr_item_seconds = (
                    time.perf_counter()
                    - ocr_item_started
                )

                detection["ocr_duration_seconds"] = (
                    ocr_item_seconds
                )

                category_timing = ocr_category_times[
                    detection["class_name"]
                ]

                category_timing["count"] += 1
                category_timing["total_seconds"] += (
                    ocr_item_seconds
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
                    "ocr_applied"
                ] = True

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

            duplicate_symbol_replacements = 0

            if use_ocr:

                duplicate_symbol_replacements = (
                    resolve_duplicate_element_symbols(
                        detections
                    )
                )

            # In YOLO-only mode no class is passed to OCR.
            # In YOLO + OCR mode, only the complete table region
            # bypasses OCR. Header regions are now read by OCR.
            for detection in detections:

                if (
                    not use_ocr
                    or detection["class_name"]
                    in NO_OCR_CLASSES
                ):

                    detection["text"] = ""
                    detection["raw_text"] = ""
                    detection["ocr_conf"] = None
                    detection["mapped"] = True
                    detection["warning"] = False
                    detection["ocr_applied"] = False

                detection["removed_overlaps"] = (
                    removed_overlaps
                )

                detection["duplicate_symbol_replacements"] = (
                    duplicate_symbol_replacements
                )

            overlay = draw_overlay(

                image,

                detections,

                show_box_text=show_box_text
            )

            ocr_recognition_seconds = sum(
                category["total_seconds"]
                for category in ocr_category_times.values()
            )

            ocr_total_seconds = (
                ocr_load_seconds
                + ocr_recognition_seconds
            )

            total_seconds = (
                time.perf_counter()
                - processing_started
            )

            timing_info = {
                "yolo_seconds": yolo_seconds,
                "ocr_recognition_seconds": ocr_recognition_seconds,
                "ocr_load_seconds": ocr_load_seconds,
                "ocr_total_seconds": ocr_total_seconds,
                "total_seconds": total_seconds,
                "ocr_category_times": ocr_category_times,
            }

            self.root.after(

                0,

                self._inspection_complete,

                detections,

                overlay,

                use_ocr,

                show_box_text,

                timing_info
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
        overlay,
        use_ocr,
        show_box_text,
        timing_info
    ):

        self.detections = (
            detections
        )

        self.annotated_image = (
            overlay
        )

        self.update_timing_display(
            timing_info
        )

        self.clear_table()

        for index, detection in enumerate(

            detections,

            start=1
        ):

            is_structural = (
                detection["class_name"]
                in STRUCTURAL_CLASSES
            )

            ocr_applied = detection.get(
                "ocr_applied",
                False
            )

            warning = (
                "⚠"
                if detection[
                    "warning"
                ]
                else ""
            )

            raw_display = (
                "—"
                if not ocr_applied
                else detection.get("raw_text", "")
            )

            final_display = (
                detection.get("text", "")
                if ocr_applied
                else "YOLO region"
                if is_structural
                else "YOLO box"
            )

            ocr_display = (
                "—"
                if not ocr_applied
                else f'{detection.get("ocr_conf", 0):.0%}'
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

                    raw_display,

                    final_display,

                    f'{detection["det_conf"]:.0%}',

                    ocr_display,

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

        self.yolo_button.config(
            state=tk.NORMAL
        )

        self.colour_button.config(
            state=tk.NORMAL
        )

        self._set_navigation_state()

        warnings = sum(

            bool(
                d["warning"]
            )

            for d
            in detections

            if d["class_name"]
            in OCR_CLASSES
        )

        recognised = sum(

            bool(
                d.get(
                    "text"
                )
            )

            for d
            in detections

            if d["class_name"]
            in OCR_CLASSES
        )

        ocr_total = sum(

            d["class_name"] in OCR_CLASSES

            for d
            in detections
        )

        structural_total = sum(

            d["class_name"] in STRUCTURAL_CLASSES

            for d
            in detections
        )

        removed_overlaps = (
            detections[0].get(
                "removed_overlaps",
                0
            )
            if detections
            else 0
        )

        duplicate_symbol_replacements = (
            detections[0].get(
                "duplicate_symbol_replacements",
                0
            )
            if detections
            else 0
        )

        if use_ocr:

            status_text = (
                f"{len(detections)} boxes | "
                f"{recognised}/{ocr_total} OCR | "
                f"{structural_total} regions | "
                f"{warnings} warnings | "
                f"{duplicate_symbol_replacements} symbols corrected | "
                f"{removed_overlaps} overlaps removed"
            )

        elif show_box_text:

            status_text = (
                f"YOLO only | "
                f"{len(detections)} boxes | "
                f"{structural_total} regions | "
                f"{removed_overlaps} overlaps removed"
            )

        else:

            status_text = (
                f"YOLO colours only | "
                f"{len(detections)} boxes | "
                f"{structural_total} regions | "
                f"{removed_overlaps} overlaps removed"
            )

        self.set_status(
            status_text,
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

        self.yolo_button.config(
            state=tk.NORMAL
        )

        self.colour_button.config(
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
        "OCR LOADING MODE"
    )

    print("=" * 70)

    print(
        "EasyOCR will load only when "
        "Run YOLO + OCR is selected."
    )

    reader = None

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
