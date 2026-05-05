"""Bridge to the existing category classifier from feature 1.

The classifier is optional here because this repository currently does not
contain feature 1's trained model file. If `RESUME_CLASSIFIER_PATH` points to a
joblib object with `.predict`, this module uses it. Otherwise the API can still
accept a manually supplied `predicted_category` for demos.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
from dotenv import load_dotenv


VALID_CATEGORIES = {"HR", "IT", "Business-Dev", "Finance", "Sales"}


class CategoryPredictor:
    def __init__(self, model_path: str | None = None) -> None:
        load_dotenv()
        raw_path = model_path if model_path is not None else os.getenv("RESUME_CLASSIFIER_PATH", "")
        self.model_path = Path(raw_path).expanduser() if raw_path else None
        self.model = None
        if self.model_path and self.model_path.exists():
            self.model = joblib.load(self.model_path)

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, resume_text: str) -> str | None:
        if not self.model:
            return None
        prediction = self.model.predict([resume_text])[0]
        category = str(prediction).strip()
        return category if category in VALID_CATEGORIES else category


@lru_cache(maxsize=1)
def get_category_predictor() -> CategoryPredictor:
    return CategoryPredictor()


def resolve_resume_category(
    *,
    resume_text: str,
    supplied_category: str | None = None,
) -> tuple[str | None, str]:
    """Return `(category, source)` for scoring transparency."""
    if supplied_category:
        return supplied_category, "request"

    predictor = get_category_predictor()
    predicted = predictor.predict(resume_text)
    if predicted:
        return predicted, "classifier"
    return None, "unavailable"
