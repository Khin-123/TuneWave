import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from app import create_app


class SeparationApiTestCase(unittest.TestCase):
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
    def make_wav(channels: int = 2) -> io.BytesIO:
        sample_rate = 8_000
        time = np.arange(2_400) / sample_rate
        left = 0.2 * np.sin(2 * np.pi * 440 * time)
        audio = left if channels == 1 else np.column_stack(
            (left, left + 0.1 * np.sin(2 * np.pi * 880 * time))
        )
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return buffer

    def test_separate_returns_metadata_and_playable_urls(self):
        response = self.client.post(
            "/api/separate",
            data={"file": (self.make_wav(), "stereo song.wav")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(
            payload["original"],
            {
                "filename": "stereo song.wav",
                "sample_rate": 8_000,
                "duration": 0.3,
                "channels": 2,
                "samples": 2_400,
            },
        )

        vocal_response = self.client.get(payload["vocal_audio_url"])
        instrumental_response = self.client.get(payload["instrumental_audio_url"])
        self.assertEqual(vocal_response.status_code, 200)
        self.assertEqual(instrumental_response.status_code, 200)
        self.assertEqual(vocal_response.mimetype, "audio/wav")
        self.assertGreater(len(vocal_response.data), 44)

        download_response = self.client.get(
            f'{payload["vocal_audio_url"]}?download=1'
        )
        self.assertEqual(download_response.status_code, 200)
        self.assertIn("attachment", download_response.headers["Content-Disposition"])
        self.assertIn("vocals.wav", download_response.headers["Content-Disposition"])
        vocal_response.close()
        instrumental_response.close()
        download_response.close()

    def test_separate_rejects_mono_audio(self):
        response = self.client.post(
            "/api/separate",
            data={"file": (self.make_wav(channels=1), "mono.wav")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])
        self.assertIn("mono audio is not supported", response.get_json()["error"])

    def test_output_route_rejects_unknown_files(self):
        response = self.client.get(f'/api/outputs/{"a" * 32}/unknown.wav')

        self.assertEqual(response.status_code, 404)
        self.assertFalse(response.get_json()["success"])


if __name__ == "__main__":
    unittest.main()
