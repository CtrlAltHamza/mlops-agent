"""Columns shared by training and the inference API."""

FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "contract_type",
    "support_calls",
]
TARGET = "churn"
REQUIRED_COLUMNS = ["customer_id", *FEATURES, TARGET]
