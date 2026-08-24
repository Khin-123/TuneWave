import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from app import create_app
from dsp.audio_utils import load_audio


class AudioInfoTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.upload_folder = Path(self.temporary_directory.name)
        self.app = create_app(
            {
                "TESTING": True,
                "UPLOAD_FOLDER": self.upload_folder,
            }
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def make_stereo_wav() -> io.BytesIO:
        sample_rate = 8_000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        stereo = np.column_stack(
            (
                0.25 * np.sin(2 * np.pi * 220 * time),
                0.25 * np.sin(2 * np.pi * 440 * time),
            )
        )
        buffer = io.BytesIO()
        sf.write(buffer, stereo, sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return buffer

    def test_load_audio_preserves_stereo_channels(self):
        wav_path = self.upload_folder / "stereo.wav"
        wav_path.write_bytes(self.make_stereo_wav().getvalue())

        audio, sample_rate = load_audio(wav_path, preserve_channels=True)

        self.assertEqual(sample_rate, 8_000)
        self.assertEqual(audio.shape, (2, 8_000))

    def test_audio_info_endpoint_returns_metadata(self):
        response = self.client.post(
            "/api/audio-info",
            data={"file": (self.make_stereo_wav(), "test tone.wav")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {
                "success": True,
                "filename": "test tone.wav",
                "sample_rate": 8_000,
                "duration": 1.0,
                "channels": 2,
                "samples": 8_000,
            },
        )

    def test_audio_info_endpoint_rejects_unsupported_extension(self):
        response = self.client.post(
            "/api/audio-info",
            data={"file": (io.BytesIO(b"not audio"), "notes.txt")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])


if __name__ == "__main__":
    unittest.main()
