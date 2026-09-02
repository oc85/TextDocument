# ============================================================
# OCR EFFECTIVENESS INSPECTION TOOL (UPGRADED V3)
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

MODEL_PATH = r"D:\Kaggle\WRtools\best_doc.pt"

IMAGE_SIZE = 1280
CONFIDENCE = 0.10
MAX_DETECTIONS = 1000
PDF_DPI = 300
DEVICE = 0 if torch.cuda.is_available() else "cpu"

ALL_ELEMENTS = [
    "Al", "B", "Cb", "Co", "Cr", "Cu", "Fe", "Mn",
    "Mo", "Ni", "P", "S", "Si", "Ta", "Ti", "V",
    "W", "Zr", "Bi", "Pb", "Sn", "Ag", "Sb", "Cd"
]


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading YOLO model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLO model not found:\n{MODEL_PATH}")

model = YOLO(MODEL_PATH)
print("YOLO loaded.")

print("Loading OCR...")
ocr_reader = easyocr.Reader(["en"], gpu=False)
print("OCR loaded.")


# ============================================================
# HELPER FUNCTIONS & TARGETED FIXES
# ============================================================

def load_image_rgb(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not open image:\n{path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def pdf_page_to_rgb(pdf_path, page_number=0, dpi=PDF_DPI):
    document = fitz.open(pdf_path)
    try:
        if len(document) == 0:
            raise ValueError("PDF contains no pages.")
        page_number = max(0, min(page_number, len(document) - 1))
        page = document[page_number]
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8)
        image = image.reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image
    finally:
        document.close()


def clean_ocr_text(text):
    text = str(text).strip()
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def crop_looks_like_dash(crop):
    if crop is None or crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop.copy()
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        if aspect > 1.8 and w > (gray.shape[1] * 0.15):
            return True
    return False


def normalize_element_text(text):
    t = clean_ocr_text(text).lower()
    
    mapping = {
        "a1": "Al", "al": "Al", "1l": "Al",
        "3": "B", "b": "B",
        "cb": "Cb", "co": "Co", "cr": "Cr", "cu": "Cu",
        "fe": "Fe", "mn": "Mn", "mo": "Mo", "ni": "Ni",
        "p": "P", "s": "S", "si": "Si", "ta": "Ta",
        "ti": "Ti", "v": "V", 
        "w": "W", "t": "W",
        "zr": "Zr", "zx": "Zr",
        "bi": "Bi", "pb": "Pb", "sn": "Sn",
        "49": "Ag", "a9": "Ag", "a4": "Ag", 
        "4g": "Ag", "ag": "Ag",
        "sb": "Sb", "cd": "Cd"
    }
    return mapping.get(t, text)


def normalize_unit_text(text, class_name=""):
    """Enforces that unit categories can only be % or ppm. If it starts with p or P, it becomes ppm."""
    t = clean_ocr_text(text)
    if not t:
        return text
    
    # Rule: If first letter is p or P -> automatically ppm
    if t.startswith('p') or t.startswith('P'):
        return "ppm"
        
    if "%" in t or "percent" in t.lower():
        return "%"
        
    # Strict fallback for unit category boxes
    if "unit" in class_name.lower() or "ppm" in class_name.lower():
        if 'p' in t.lower():
            return "ppm"
        return "%"
        
    return t


def read_crop_text(image, box):
    class_name = box.get("class_name", "")
    x1, y1 = max(0, int(box["x1"])), max(0, int(box["y1"]))
    x2, y2 = min(image.shape[1], int(box["x2"])), min(image.shape[0], int(box["y2"]))

    if x2 <= x1 or y2 <= y1:
        return ""

    tight_crop = image[y1:y2, x1:x2]
    if tight_crop.size == 0:
        return ""

    if "value" in class_name or "limit" in class_name:
        if crop_looks_like_dash(tight_crop):
            return "-"

    crop = cv2.resize(tight_crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    allowlist = None
    if "value" in class_name:
        allowlist = "0123456789.-<>Balancebal"
    elif "unit" in class_name or "ppm" in class_name.lower():
        allowlist = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ%-"

    try:
        if allowlist:
            results = ocr_reader.readtext(crop, detail=0, paragraph=False, allowlist=allowlist)
        else:
            results = ocr_reader.readtext(crop, detail=0, paragraph=False)
            
        if results:
            raw_text = clean_ocr_text(" ".join(results))
            if "element" in class_name:
                return normalize_element_text(raw_text)
            elif "unit" in class_name or "ppm" in class_name.lower():
                return normalize_unit_text(raw_text, class_name)
            return raw_text
    except Exception:
        pass

    if "value" in class_name and crop_looks_like_dash(tight_crop):
        return "-"

    return ""


def run_yolo(image):
    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        max_det=MAX_DETECTIONS,
        device=DEVICE,
        verbose=False
    )
    result = results[0]
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0].cpu().item())
        class_name = model.names[class_id]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        detections.append({
            "class_name": class_name,
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "cx": float((x1 + x2) / 2),
            "cy": float((y1 + y2) / 2)
        })

    return detections


def draw_ocr_overlay(image, detections):
    output = image.copy()

    for d in detections:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        text = d.get("text", "")
        class_name = d.get("class_name", "")

        if "value" in class_name:
            box_color = (0, 140, 255)    # Orange
        elif "unit" in class_name:
            box_color = (0, 200, 0)      # Green
        elif "element" in class_name:
            box_color = (255, 0, 0)      # Red
        else:
            box_color = (120, 0, 255)    # Purple

        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2)

        display_text = f"{class_name}: {text}" if text else f"[{class_name}]"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.40
        thickness = 1

        (text_w, text_h), _ = cv2.getTextSize(display_text, font, font_scale, thickness)
        label_y1 = max(y1 - text_h - 6, 0)
        label_y2 = label_y1 + text_h + 6
        label_x2 = min(x1 + text_w + 6, output.shape[1])

        cv2.rectangle(output, (x1, label_y1), (label_x2, label_y2), box_color, -1)
        cv2.putText(output, display_text, (x1 + 3, label_y2 - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return output


# ============================================================
# INSPECTION TOOL GUI
# ============================================================

class OCRInspectionTool:

    def __init__(self, root):
        self.root = root
        self.root.title("OCR Effectiveness Inspection Tool (V3)")
        self.root.geometry("1500x950")

        self.file_path = None
        self.current_image = None
        self.annotated_image = None
        self.tk_image = None
        self.current_page = 0
        self.pdf_pages = 0

        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_panel = ttk.Frame(main_frame)
        control_panel.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(control_panel, text="Upload PDF / Image", command=self.upload_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_panel, text="Run OCR & Visualize", command=self.run_inspection).pack(side=tk.LEFT, padx=5)

        self.file_label = ttk.Label(control_panel, text="No document loaded", font=("Segoe UI", 10))
        self.file_label.pack(side=tk.LEFT, padx=15)

        self.status_label = ttk.Label(control_panel, text="Ready", font=("Segoe UI", 10, "bold"), foreground="blue")
        self.status_label.pack(side=tk.RIGHT, padx=5)

        canvas_container = ttk.Frame(main_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_container, bg="#333333")
        x_scroll = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        y_scroll = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        canvas_container.rowconfigure(0, weight=1)
        canvas_container.columnconfigure(0, weight=1)

    def upload_file(self):
        path = filedialog.askopenfilename(
            title="Select document",
            filetypes=[("PDF and images", "*.pdf *.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if not path:
            return

        self.file_path = path
        self.current_page = 0
        self.file_label.config(text=os.path.basename(path))

        extension = os.path.splitext(path)[1].lower()

        try:
            if extension == ".pdf":
                doc = fitz.open(path)
                self.pdf_pages = len(doc)
                doc.close()
                self.current_image = pdf_page_to_rgb(path, 0)
            else:
                self.pdf_pages = 1
                self.current_image = load_image_rgb(path)

            self.annotated_image = None
            self.show_image(self.current_image)
            self.status_label.config(text="Document loaded. Click 'Run OCR & Visualize'.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_inspection(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please upload a document first.")
            return

        try:
            self.status_label.config(text="Detecting regions with YOLO...")
            self.root.update_idletasks()

            detections = run_yolo(self.current_image)

            self.status_label.config(text=f"Extracted {len(detections)} boxes. Running targeted OCR...")
            self.root.update_idletasks()

            for det in detections:
                det["text"] = read_crop_text(self.current_image, det)

            self.status_label.config(text="Rendering OCR text overlay...")
            self.root.update_idletasks()

            self.annotated_image = draw_ocr_overlay(self.current_image, detections)
            self.show_image(self.annotated_image)

            self.status_label.config(text=f"Inspection complete! Processed {len(detections)} boxes.")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="Processing failed.")

    def show_image(self, image):
        if image is None:
            return

        h, w = image.shape[:2]
        pil_image = Image.fromarray(image)
        self.tk_image = ImageTk.PhotoImage(pil_image)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.configure(scrollregion=(0, 0, w, h))


if __name__ == "__main__":
    root = tk.Tk()
    app = OCRInspectionTool(root)
    root.mainloop()
