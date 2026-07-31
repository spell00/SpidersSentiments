"""Opt-in integration tests that hit the real iNaturalist API."""

from __future__ import annotations

import io
import os
import unittest

import requests
from PIL import Image

from spider_guardian.inat_client import INaturalistClient


class INaturalistClientLiveTests(unittest.TestCase):
    """Integration tests for the iNaturalist client against the real API."""

    SAMPLE_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/3/3f/Latrodectus_mactans_1.jpg"

    @unittest.skipUnless(
        os.environ.get("RUN_INAT_LIVE_TESTS") == "1",
        "Set RUN_INAT_LIVE_TESTS=1 to enable live iNaturalist API tests.",
    )
    def test_score_image_live(self):
        token = os.environ.get("INAT_ACCESS_TOKEN")
        if not token:
            self.skipTest("INAT_ACCESS_TOKEN must be set to run live iNaturalist tests")

        client = INaturalistClient.from_env()
        if client is None or not client.is_available:
            self.skipTest("iNaturalist client not available via environment settings")

        response = requests.get(self.SAMPLE_IMAGE_URL, timeout=30)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        result = client._score_image(image)

        self.assertIsNotNone(result, "iNaturalist returned no classifications")
        self.assertIsInstance(result, dict)

        taxon = result.get("taxon") or {}
        self.assertIn("id", taxon)
        self.assertTrue(taxon.get("id"), "Taxon ID missing from iNaturalist response")

        score = INaturalistClient._extract_score(result)
        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
