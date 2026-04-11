"""Feature column order, categorical code maps, and UI labels for the credit model."""

FEATURE_ORDER = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
    "AMT_INCOME_TOTAL", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
    "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL", "CNT_FAM_MEMBERS",
    "AGE_YEARS", "YEARS_EMPLOYED",
    "CREDIT_SCORE",
]

INCOME_TYPES = {
    0: "Working",
    1: "Commercial Associate",
    2: "Pensioner",
    3: "State Servant",
    4: "Student",
}

EDUCATION_TYPES = {
    0: "Lower Secondary",
    1: "Secondary / Special",
    2: "Incomplete Higher",
    3: "Higher Education",
    4: "Academic Degree",
}

FAMILY_STATUS = {
    0: "Single / Not Married",
    1: "Married",
    2: "Separated",
    3: "Civil Marriage",
    4: "Widow",
}

HOUSING_TYPES = {
    0: "With Parents",
    1: "Rented Apartment",
    2: "Municipal Apartment",
    3: "Co-op Apartment",
    4: "House / Apartment (owned)",
    5: "Office Apartment",
}

OCCUPATION_TYPES = {
    0: "Laborers",
    1: "Core Staff",
    2: "Sales Staff",
    3: "Managers",
    4: "Drivers",
    5: "High Skill Tech Staff",
    6: "Accountants",
    7: "Medicine Staff",
    8: "Cooking Staff",
    9: "Security Staff",
    10: "Cleaning Staff",
    11: "Private Service Staff",
    12: "Low-skill Laborers",
    13: "Waiters / Barmen Staff",
    14: "Secretaries",
    15: "Realty Agents",
    16: "HR Staff",
    17: "IT Staff",
    18: "Other / Unknown",
}

FEATURE_LABELS = {
    "CODE_GENDER": "Gender",
    "FLAG_OWN_CAR": "Car Ownership",
    "FLAG_OWN_REALTY": "Property Ownership",
    "CNT_CHILDREN": "Number of Children",
    "AMT_INCOME_TOTAL": "Annual Income",
    "NAME_INCOME_TYPE": "Income Type",
    "NAME_EDUCATION_TYPE": "Education Level",
    "NAME_FAMILY_STATUS": "Family Status",
    "NAME_HOUSING_TYPE": "Housing Type",
    "OCCUPATION_TYPE": "Occupation",
    "FLAG_WORK_PHONE": "Has Work Phone",
    "FLAG_PHONE": "Has Home Phone",
    "FLAG_EMAIL": "Has Email",
    "CNT_FAM_MEMBERS": "Family Members",
    "AGE_YEARS": "Age",
    "YEARS_EMPLOYED": "Years Employed",
    "CREDIT_SCORE": "Credit Score",
}
