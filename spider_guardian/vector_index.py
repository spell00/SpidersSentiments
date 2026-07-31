"""Vector index utilities for contextual retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import logging
import numpy as np

try:
    from datasets import load_from_disk
except Exception:  # pragma: no cover - optional dependency
    load_from_disk = None  # type: ignore

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


@dataclass
class VectorIndex:
    """Lightweight wrapper around a sentence-transformer + nearest-neighbour index."""

    embedder_name: str
    embedder: SentenceTransformer = field(init=False)
    index: Optional[NearestNeighbors] = field(default=None, init=False)
    embeddings: Optional[np.ndarray] = field(default=None, init=False)
    documents: List[str] = field(default_factory=list, init=False)
    labels: List[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.embedder = SentenceTransformer(self.embedder_name)

    def _load_dataset_any(self, dataset_path: str):
        if load_from_disk is None:
            raise ImportError("Install datasets: pip install datasets")
        dataset = load_from_disk(dataset_path)
        if hasattr(dataset, "column_names"):
            return dataset
        for key in ("train", "validation", "test"):
            if key in dataset:
                return dataset[key]
        return next(iter(dataset.values()))

    def build(self, dataset_path: str) -> None:
        logging.info("Loading dataset from %s", dataset_path)
        dataset = self._load_dataset_any(dataset_path)
        if "text" not in dataset.column_names or "label" not in dataset.column_names:
            raise ValueError("Dataset must contain 'text' and 'label' columns")
        texts = list(dataset["text"])
        raw_labels = dataset["label"]
        labels: List[int] = []
        for lbl in raw_labels:
            if isinstance(lbl, (list, tuple, np.ndarray)):
                labels.append(int(np.argmax(lbl)))
            else:
                labels.append(int(lbl))
        self.documents = texts
        self.labels = labels
        logging.info("Computing embeddings with %s", self.embedder_name)
        embeds = self.embedder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32,
            normalize_embeddings=True,
        )
        logging.info("Fitting nearest neighbors")
        self.index = NearestNeighbors(metric="cosine")
        self.index.fit(embeds)
        self.embeddings = embeds

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if not self.index or self.embeddings is None:
            raise RuntimeError("Vector index not built. Call build().")
        q = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = self.index.kneighbors(q, n_neighbors=min(top_k, len(self.documents)))
        return [
            (self.documents[int(idx)], 1.0 - float(dist))
            for dist, idx in zip(distances[0], indices[0])
        ]
