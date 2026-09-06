PDF / image
    ↓
YOLO detector
    ↓
detect semantic categories:
- element_symbol
- element_value
- unit
- limit_indicator
- value_range
- sign
    ↓
class-specific recognition



YOLO
│
├── element_symbol
│      → constrained OCR
│      → element dictionary
│      → manual mappings
│
├── element_value
│      → numeric OCR
│      → text OCR in parallel
│      → dash fallback
│
├── unit
│      → OCR
│      → mapping
│
├── limit_indicator
│      → OCR
│      → mapping
│
├── value_range
│      → range OCR
│      → validation / parsing
│
└── sign
       → small classifier
