"""Unit tests for the lightweight iNaturalist client."""

from __future__ import annotations

import unittest
from unittest import mock

from PIL import Image

from spider_guardian.inat_client import INaturalistClient, INatPrediction


class FakeResponse:
    """Minimal response object used to simulate requests responses."""

    def __init__(self, *, status_code: int = 200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise RuntimeError(f"HTTP error {self.status_code}")


class FakeSession:
    """Simple session stub that records calls for assertions."""

    def __init__(self, post_response: FakeResponse, get_response: FakeResponse):
        self.headers = {}
        self._post_response = post_response
        self._get_response = get_response
        self.post_calls = []
        self.get_calls = []

    def post(self, url, files=None, data=None, timeout=None):
        self.post_calls.append((url, files, data, timeout))
        if isinstance(self._post_response, Exception):
            raise self._post_response
        return self._post_response

    def get(self, url, params=None, timeout=None):
        self.get_calls.append((url, params, timeout))
        if isinstance(self._get_response, Exception):
            raise self._get_response
        return self._get_response


class INaturalistClientTests(unittest.TestCase):
    """Tests for the iNaturalist client without performing network calls."""

    def setUp(self):
        # Minimal image buffer used across tests
        self.image = Image.new("RGB", (4, 4), color="white")

        self.sample_taxon = {
            "id": 123,
            "name": "Latrodectus mactans",
            "rank": "species",
            "preferred_common_name": "Southern Black Widow",
            "ancestors": [
                {"rank": "kingdom", "name": "Animalia"},
                {"rank": "phylum", "name": "Arthropoda"},
                {"rank": "class", "name": "Arachnida"},
                {"rank": "order", "name": "Araneae"},
                {"rank": "family", "name": "Theridiidae"},
                {"rank": "genus", "name": "Latrodectus"},
            ],
        }

        self.score_payload = {
            "results": [
                {
                    "taxon": {"id": 123},
                    "vision_score": 0.87,
                }
            ]
        }

        self.taxon_payload = {
            "results": [self.sample_taxon]
        }

    def _build_client(self):
        post_response = FakeResponse(json_data=self.score_payload)
        get_response = FakeResponse(json_data=self.taxon_payload)
        session = FakeSession(post_response, get_response)
        client = INaturalistClient(session=session, timeout=5)
        return client, session

    def test_identify_taxon_returns_prediction_with_lineage(self):
        client, session = self._build_client()

        prediction = client.identify_taxon(self.image)
        self.assertIsInstance(prediction, INatPrediction)

        self.assertEqual(len(session.post_calls), 1)
        self.assertGreaterEqual(len(session.post_calls[0][1]["image"][1]), 10)

        self.assertEqual(prediction.score, 0.87)
        self.assertEqual(prediction.taxon["id"], 123)

        # The detailed taxon response should populate lineage fields.
        self.assertEqual(prediction.taxon["name"], "Latrodectus mactans")
        self.assertEqual(prediction.taxon["ancestors"][0]["name"], "Animalia")

        # The detail request should happen once even across repeated calls thanks to caching.
        second = client.identify_taxon(self.image)
        self.assertEqual(second.score, prediction.score)
        self.assertEqual(len(session.get_calls), 1)

    def test_identify_taxon_handles_score_errors(self):
        client, _session = self._build_client()

        with mock.patch.object(client, "_score_image", side_effect=RuntimeError("boom")):
            result = client.identify_taxon(self.image)

        self.assertIsNone(result)
        self.assertFalse(client.is_available)


if __name__ == "__main__":
    unittest.main()
