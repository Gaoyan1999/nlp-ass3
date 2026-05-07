"""Load and run the saved resume category models."""

from __future__ import annotations

import json
import os
import pickle
import re
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        filter_sizes: tuple[int, ...] = (3, 4, 5),
        num_filters: int = 64,
        dropout: float = 0.4,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=fs) for fs in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def extract_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))
            pooled.append(torch.max(h, dim=2).values)
        return torch.cat(pooled, dim=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(input_ids)
        return self.fc(self.dropout(features))


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.4,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def extract_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(x)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(input_ids)
        return self.fc(self.dropout(features))


class SavedResumeClassifier:
    """Inference wrapper for the saved category prediction models."""

    def __init__(self, model_dir: str | Path | None = None, model_name: str | None = None) -> None:
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR).expanduser()
        if not self.model_dir.exists():
            raise FileNotFoundError(f"No saved_models directory found at {self.model_dir}.")

        self.metadata = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))
        with (self.model_dir / "vocab.pkl").open("rb") as handle:
            self.vocab: dict[str, int] = pickle.load(handle)

        self.label_names = [str(label) for label in self.metadata["label_names"]]
        self.num_classes = int(self.metadata["num_classes"])
        self.max_seq_len = int(self.metadata["max_seq_len"])
        requested_model = model_name or os.getenv("RESUME_CLASSIFIER_MODEL")
        self.model_name = self._resolve_model_name(
            requested_model or self.metadata.get("best_model_name", "BiLSTM")
        )
        self.tokenizer = None
        self.model = self._load_model(self.model_name)

    def predict(self, text: str) -> str:
        """Return the raw saved-model label, for example `information-technology`."""
        label, _confidence = self.predict_with_confidence(text)
        return label

    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        if self.model_name == "DistilBERT":
            logits = self._predict_transformer_logits(text)
        else:
            input_ids = self._encode_text(text).to(DEVICE)
            with torch.inference_mode():
                logits = self.model(input_ids)

        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction_id = torch.max(probabilities, dim=1)
        return self.label_names[int(prediction_id.item())], float(confidence.item())

    def _resolve_model_name(self, model_name: str) -> str:
        normalised = model_name.strip().casefold()
        aliases = {
            "bilstm": "BiLSTM",
            "bi-lstm": "BiLSTM",
            "textcnn": "TextCNN",
            "text-cnn": "TextCNN",
            "distilbert": "DistilBERT",
            "distilbert-base-uncased": "DistilBERT",
        }
        resolved = aliases.get(normalised)
        if not resolved:
            raise ValueError(f"Unsupported saved classifier model: {model_name}")
        if resolved == "DistilBERT" and not TRANSFORMERS_AVAILABLE:
            fallback = self.metadata.get("best_model_name", "BiLSTM")
            if str(fallback).strip().casefold() in {"distilbert", "distilbert-base-uncased"}:
                return "BiLSTM"
            return self._resolve_model_name(fallback)
        return resolved

    def _load_model(self, model_name: str) -> nn.Module:
        if model_name == "BiLSTM":
            model = BiLSTMClassifier(
                len(self.vocab),
                embed_dim=128,
                hidden_dim=128,
                num_classes=self.num_classes,
            ).to(DEVICE)
            checkpoint = torch.load(self.model_dir / "bilstm.pt", map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
        elif model_name == "TextCNN":
            model = TextCNN(len(self.vocab), embed_dim=128, num_classes=self.num_classes).to(DEVICE)
            checkpoint = torch.load(self.model_dir / "textcnn.pt", map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            if (
                not TRANSFORMERS_AVAILABLE
                or AutoModelForSequenceClassification is None
                or AutoTokenizer is None
            ):
                raise RuntimeError("transformers is required to load DistilBERT.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir / "distilbert")
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir / "distilbert"
            ).to(DEVICE)

        model.eval()
        return model

    def _predict_transformer_logits(self, text: str) -> torch.Tensor:
        if not TRANSFORMERS_AVAILABLE or self.tokenizer is None:
            raise RuntimeError("transformers is required to run DistilBERT.")
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=int(self.metadata["transformer_max_len"]),
            return_tensors="pt",
        )
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
        with torch.inference_mode():
            return self.model(**encoded).logits

    def _encode_text(self, text: str) -> torch.Tensor:
        tokens = _tokenise(text)[: self.max_seq_len]
        unk_id = self.vocab.get("<UNK>", 1)
        pad_id = self.vocab.get("<PAD>", 0)
        ids = [self.vocab.get(token, unk_id) for token in tokens]
        ids.extend([pad_id] * (self.max_seq_len - len(ids)))
        return torch.tensor([ids], dtype=torch.long)


def _tokenise(text: str) -> list[str]:
    text = re.sub(r"<[^>]+>", " ", text.casefold())
    return re.findall(r"[a-z0-9+#.]+", text)


@lru_cache(maxsize=1)
def get_saved_classifier(
    model_dir: str | Path | None = None,
    model_name: str | None = None,
) -> SavedResumeClassifier:
    return SavedResumeClassifier(model_dir=model_dir, model_name=model_name)


def predict_category(text: str) -> str:
    return get_saved_classifier().predict(text)


if __name__ == "__main__":
    classifier = get_saved_classifier()
    print(f"Loaded saved models from: {classifier.model_dir.resolve()}")
    print(f"Using model: {classifier.model_name}")
