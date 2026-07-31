"""Thin client for interacting with the iNaturalist API."""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from PIL import Image


@dataclass
class INatPrediction:
    """Container for the top iNaturalist computer-vision result."""

    taxon: Dict[str, Any]
    score: float


class INaturalistClient:
    """Lightweight helper around the public iNaturalist API."""

    def __init__(
        self,
        base_url: str = "https://api.inaturalist.org/v1",
        *,
        access_token: Optional[str] = None,
        timeout: int = 20,
        preferred_place_id: Optional[str] = None,
        locale: str = "en",
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.preferred_place_id = preferred_place_id
        self.locale = locale
        self.access_token = access_token or os.environ.get("INAT_ACCESS_TOKEN")
        self.session = session or requests.Session()
        self.session.headers.setdefault(
            "User-Agent",
            os.environ.get(
                "INAT_USER_AGENT",
                "SpiderGuardianBot/1.0 (+https://github.com/spell00/SpidersSentiments)",
            ),
        )
        accept_language = os.environ.get("INAT_ACCEPT_LANGUAGE")
        if accept_language:
            self.session.headers.setdefault("Accept-Language", accept_language)
        if self.access_token:
            self.session.headers["Authorization"] = f"Bearer {self.access_token}"
        self.enabled = True
        self._taxon_cache: Dict[int, Dict[str, Any]] = {}

    @classmethod
    def from_env(cls) -> Optional["INaturalistClient"]:
        """Build a client from environment variables, handling opt-out flags."""

        disable_flag = os.environ.get("SPIDER_GUARDIAN_DISABLE_INAT", "")
        if disable_flag.lower() in {"1", "true", "yes"}:
            logging.info("iNaturalist client disabled via SPIDER_GUARDIAN_DISABLE_INAT=%s", disable_flag)
            return None

        timeout = 20
        timeout_str = os.environ.get("INAT_TIMEOUT")
        if timeout_str:
            try:
                timeout = int(timeout_str)
            except ValueError:
                logging.warning("Invalid INAT_TIMEOUT value '%s'; falling back to %s", timeout_str, timeout)

        preferred_place_id = os.environ.get("INAT_PREFERRED_PLACE_ID") or None
        locale = os.environ.get("INAT_LOCALE", "en")

        try:
            return cls(
                access_token=os.environ.get("INAT_ACCESS_TOKEN"),
                timeout=timeout,
                preferred_place_id=preferred_place_id,
                locale=locale,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logging.warning("Unable to initialise iNaturalist client: %s", exc)
            return None

    @property
    def is_available(self) -> bool:
        """Return whether the client is currently usable."""

        return self.enabled

    def identify_taxon(
        self,
        image: Image.Image,
        *,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        place_id: Optional[int] = None,
    ) -> Optional[INatPrediction]:
        """Return the top taxonomy prediction for an image, if possible."""

        if not self.enabled:
            return None

        try:
            result = self._score_image(image, lat=lat, lng=lng, place_id=place_id)
        except Exception as exc:  # pragma: no cover - network interaction
            logging.warning("iNaturalist score_image failed: %s", exc)
            self.enabled = False
            return None

        if not result:
            return None

        taxon = result.get("taxon") or {}
        taxon_id = taxon.get("id")
        if taxon_id:
            detailed = self._get_taxon_detail(taxon_id)
            if detailed:
                taxon = detailed

        score = self._extract_score(result)
        return INatPrediction(taxon=taxon, score=score)

    def _score_image(
        self,
        image: Image.Image,
        *,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        place_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Call the computer vision endpoint and return the top result."""

        endpoint = f"{self.base_url}/computervision/score_image"
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)

        files = {"image": ("image.jpg", buffer.getvalue(), "image/jpeg")}
        data: Dict[str, Any] = {}
        if lat is not None and lng is not None:
            data["lat"] = lat
            data["lng"] = lng
        if place_id is not None:
            data["place_id"] = place_id

        response = self.session.post(endpoint, files=files, data=data, timeout=self.timeout)

        if response.status_code in {401, 403}:
            logging.warning("iNaturalist API denied access (status %s); disabling client", response.status_code)
            self.enabled = False
            return None
        if response.status_code == 429:
            logging.warning("iNaturalist API rate-limited the request (429)")
            return None

        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") or []
        if not results:
            return None
        return results[0]

    def _get_taxon_detail(self, taxon_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve rich taxonomy information for a specific taxon id."""

        if taxon_id in self._taxon_cache:
            return self._taxon_cache[taxon_id]

        endpoint = f"{self.base_url}/taxa/{taxon_id}"
        params = {
            "include": "ancestors",
            "all_names": "true",
            "locale": self.locale,
        }
        if self.preferred_place_id:
            params["preferred_place_id"] = self.preferred_place_id

        response = self.session.get(endpoint, params=params, timeout=self.timeout)
        if response.status_code == 404:
            return None
        response.raise_for_status()

        payload = response.json()
        results = payload.get("results") or []
        if not results:
            return None

        taxon = results[0]
        self._taxon_cache[taxon_id] = taxon
        return taxon

    @staticmethod
    def _extract_score(result: Dict[str, Any]) -> float:
        """Pull out the most useful confidence score from a result blob."""

        for key in ("vision_score", "score", "combined_score", "frequency_score"):
            value = result.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0
