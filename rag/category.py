"""Bridge to the resume category classifier from feature 1.

The API first tries the saved PyTorch models in `saved_models/`. The older
`RESUME_CLASSIFIER_PATH` joblib bridge remains as a fallback for demos or
alternate classifiers.
"""

from __future__ import annotations

import os
from functools import lru_cache
from importlib import import_module
from pathlib import Path

import joblib
from dotenv import load_dotenv


VALID_CATEGORIES = {"HR", "INFORMATION-TECHNOLOGY", "BUSINESS-DEVELOPMENT", "FINANCE", "SALES"}
CATEGORY_ALIASES = {
    "business development": "BUSINESS-DEVELOPMENT",
    "business-development": "BUSINESS-DEVELOPMENT",
    "business_dev": "BUSINESS-DEVELOPMENT",
    "business-dev": "BUSINESS-DEVELOPMENT",
    "finance": "FINANCE",
    "fin": "FINANCE",
    "hr": "HR",
    "human resources": "HR",
    "information technology": "INFORMATION-TECHNOLOGY",
    "information-technology": "INFORMATION-TECHNOLOGY",
    "it": "INFORMATION-TECHNOLOGY",
    "sales": "SALES",
}


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
        return normalise_category(str(prediction))


class SavedModelCategoryPredictor:
    def __init__(self) -> None:
        load_dotenv()
        self._classifier = None
        self._load_error: Exception | None = None
        try:
            load_model = import_module("load_model")
            model_dir = os.getenv("RESUME_CLASSIFIER_MODEL_DIR") or None
            model_name = os.getenv("RESUME_CLASSIFIER_MODEL") or None
            self._classifier = load_model.get_saved_classifier(model_dir, model_name)
        except Exception as exc:
            self._load_error = exc

    @property
    def available(self) -> bool:
        return self._classifier is not None

    def predict(self, resume_text: str) -> str | None:
        if not self._classifier:
            return None
        prediction = self._classifier.predict(resume_text)
        return normalise_category(prediction)


@lru_cache(maxsize=1)
def get_category_predictor() -> CategoryPredictor:
    return CategoryPredictor()


@lru_cache(maxsize=1)
def get_saved_model_predictor() -> SavedModelCategoryPredictor:
    return SavedModelCategoryPredictor()


def normalise_category(category: str | None) -> str | None:
    if not category:
        return None
    key = " ".join(str(category).strip().replace("_", " ").split()).casefold()
    key = key.replace("/", " ")
    return CATEGORY_ALIASES.get(key, str(category).strip().upper())


def resolve_resume_category(
    *,
    resume_text: str,
    supplied_category: str | None = None,
) -> tuple[str | None, str]:
    """Return `(category, source)` for scoring transparency."""
    if supplied_category:
        return normalise_category(supplied_category), "request"

    saved_predictor = get_saved_model_predictor()
    predicted = saved_predictor.predict(resume_text)
    if predicted:
        return predicted, "saved_model"

    predictor = get_category_predictor()
    predicted = predictor.predict(resume_text)
    if predicted:
        return predicted, "classifier"
    return None, "unavailable"
