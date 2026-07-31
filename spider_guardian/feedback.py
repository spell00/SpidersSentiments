"""Feedback model utilities."""

from __future__ import annotations

from typing import Iterable

import logging
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


class FeedbackModel:
    """Online-learning classifier that detects hostile replies."""

    def __init__(self) -> None:
        self.vectorizer = HashingVectorizer(n_features=2**14, alternate_sign=False, norm="l2")
        self.classifier = SGDClassifier(loss="log_loss")
        self._is_initialized = False

    def update(self, texts: Iterable[str], labels: Iterable[int]) -> None:
        texts = list(texts)
        if not texts:
            return
        X = self.vectorizer.transform(texts)
        y = np.fromiter(labels, dtype=int)
        if not self._is_initialized:
            self.classifier.partial_fit(X, y, classes=np.array([0, 1]))
            self._is_initialized = True
        else:
            self.classifier.partial_fit(X, y)

    def predict_hostility(self, text: str) -> float:
        if not self._is_initialized:
            return 0.5
        X = self.vectorizer.transform([text])
        return float(self.classifier.predict_proba(X)[0][1])
