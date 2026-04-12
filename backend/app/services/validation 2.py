import re

VALID_RANGES = {
    "gender": (0, 1),
    "own_car": (0, 1),
    "own_realty": (0, 1),
    "children": (0, 19),
    "income": (10000, 2000000),
    "income_type": (0, 4),
    "education": (0, 4),
    "family_status": (0, 4),
    "housing": (0, 5),
    "occupation": (0, 18),
    "work_phone": (0, 1),
    "phone": (0, 1),
    "email": (0, 1),
    "family_members": (1, 20),
    "age": (18, 70),
    "years_employed": (0, 50),
}

REQUIRED_FIELDS = list(VALID_RANGES.keys())


def sanitize_string(s):
    if not isinstance(s, str):
        return s
    s = re.sub(r"<[^>]+>", "", s)
    s = s.strip()
    return s


def validate_application_input(data):
    errors = []
    cleaned = {}

    for field in REQUIRED_FIELDS:
        val = data.get(field)
        if val is None:
            errors.append(f"Missing field: {field}")
            continue

        try:
            val = float(val)
        except (ValueError, TypeError):
            errors.append(f"Invalid value for {field}: must be numeric")
            continue

        lo, hi = VALID_RANGES[field]
        if val < lo or val > hi:
            errors.append(f"{field} must be between {lo} and {hi}")
            continue

        if field in ("gender", "own_car", "own_realty", "children", "income_type",
                      "education", "family_status", "housing", "occupation",
                      "work_phone", "phone", "email", "family_members"):
            cleaned[field] = int(val)
        else:
            cleaned[field] = val

    return cleaned, errors
