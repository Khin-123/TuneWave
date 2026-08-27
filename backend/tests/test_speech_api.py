import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from app import create_app


class SpeechEnhancementApiTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.test_directory = Path(self.temporary_directory.name)
        self.app = create_app(
            {
                "TESTING": True,
                "UPLOAD_FOLDER": self.test_directory / "uploads",
                "OUTPUT_FOLDER": self.test_directory / "outputs",
            }
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def make_noisy_speech_wav() -> io.BytesIO:
        sample_rate = 16_000
        time = np.arange(int(0.4 * sample_rate)) / sample_rate
        generator = np.random.default_rng(42)
        speech = 0.22 * np.sin(2 * np.pi * 190 * time)
        speech += 0.08 * np.sin(2 * np.pi * 380 * time)
        noisy = speech + generator.normal(0.0, 0.04, time.size)
        buffer = io.BytesIO()
        sf.write(buffer, noisy, sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return buffer

    def test_enhance_speech_returns_playable_visualization_and_download_urls(self):
        response = self.client.post(
            "/api/enhance-speech",
            data={"file": (self.make_noisy_speech_wav(), "noisy speech.wav")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["original"]["filename"], "noisy speech.wav")
        self.assertEqual(payload["original"]["sample_rate"], 16_000)
        self.assertEqual(payload["original"]["channels"], 1)
        self.assertEqual(payload["processing"]["noise_estimation"], "Automatic")

        urls = (
            payload["original"]["audio_url"],
            payload["original"]["waveform_url"],
            payload["original"]["spectrogram_url"],
            payload["enhanced"]["audio_url"],
            payload["enhanced"]["waveform_url"],
            payload["enhanced"]["spectrogram_url"],
        )
        for url in urls:
            served = self.client.get(url)
            try:
                self.assertEqual(served.status_code, 200)
                if url.endswith(".png"):
                    self.assertEqual(served.mimetype, "image/png")
                    self.assertTrue(served.data.startswith(b"\x89PNG\r\n\x1a\n"))
                else:
                    self.assertEqual(served.mimetype, "audio/wav")
                    self.assertGreater(len(served.data), 44)
            finally:
                served.close()

        download = self.client.get(payload["enhanced"]["download_url"])
        self.assertEqual(download.status_code, 200)
        self.assertIn("attachment", download.headers["Content-Disposition"])
        self.assertIn("enhanced_speech.wav", download.headers["Content-Disposition"])
        download.close()

    def test_enhance_speech_rejects_missing_file(self):
        response = self.client.post("/api/enhance-speech")
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])

    def test_enhance_speech_hides_internal_processing_errors(self):
        with patch("app.enhance_speech_audio_file", side_effect=RuntimeError("private details")):
            response = self.client.post(
                "/api/enhance-speech",
                data={"file": (self.make_noisy_speech_wav(), "noisy.wav")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"], "Speech enhancement failed.")


if __name__ == "__main__":
    unittest.main()
