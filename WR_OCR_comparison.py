def normalise_value(text):
    original = clean_text(text)
    mapped = apply_mapping(original, VALUE_MAPPING)
    if mapped is not None:
        return mapped, True

    # Strip trailing units to extract pure numeric values (e.g. "17.7 %" -> "17.7")
    clean_val = re.sub(r"\s*(?:%|wt%|ppm|ppb)$", "", original, flags=re.IGNORECASE)

    value = clean_val.translate(str.maketrans({
        "O": "0", "o": "0",
        "I": "1", "l": "1", "|": "1",
        ",": ".",
    }))
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = re.sub(r"\s+", "", value)

    if value == "-":
        return "-", True

    if re.fullmatch(r"\d{4,5}1", value) and not original.endswith("1"):
        value = value[:-1]

    range_match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)-([+-]?\d+(?:\.\d+)?)", value)
    if range_match:
        return f"{range_match.group(1)} - {range_match.group(2)}", True

    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        return value, True

    if re.fullmatch(r"[<>]=?[+-]?\d+(?:\.\d+)?", value):
        return value, True

    if re.fullmatch(r"[A-Za-z]+", original):
        return original, True

    return value, False
