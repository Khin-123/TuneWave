import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        self.separator_patch = patch('app.separate_music_audio_file', side_effect=self.fake_separator)
        self.mock_separator = self.separator_patch.start()

    def tearDown(self):
        self.separator_patch.stop()
        self.temporary_directory.cleanup()

    @staticmethod
    def fake_separator(input_path, output_directory, **_kwargs):
        audio, sample_rate = sf.read(input_path, always_2d=True, dtype='float32')
        if audio.shape[1] != 2:
            raise ValueError('mono audio is not supported')
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        vocals = 0.5 * audio
        instrumental = audio - vocals
        paths = {}
        for name, stem in {'vocals': vocals, 'instrumental': instrumental}.items():
            path = output_directory / f'{name}.wav'
            sf.write(path, stem, sample_rate, subtype='PCM_16')
            paths[name] = path
        return {
            'sample_rate': sample_rate,
            'samples': audio.shape[0],
            'channels': 2,
            'paths': paths,
        }

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
        self.assertEqual(payload['original']['filename'], 'stereo song.wav')
        self.assertEqual(payload['original']['sample_rate'], 8_000)
        self.assertEqual(payload['original']['duration'], 0.3)
        self.assertEqual(payload['original']['channels'], 2)
        self.assertEqual(payload['original']['samples'], 2_400)
        self.assertIn('audio_url', payload['original'])
        self.assertIn('download_url', payload['vocals'])
        self.assertIn('download_url', payload['instrumental'])

        visualization_urls = []
        for source_name in ('original', 'vocals', 'instrumental'):
            for visualization_name in ('waveform', 'spectrogram'):
                url = payload[source_name][f'{visualization_name}_url']
                self.assertIn(
                    f'{source_name}_{visualization_name}.png',
                    url,
                )
                visualization_urls.append(url)

        vocal_response = self.client.get(payload["vocal_audio_url"])
        instrumental_response = self.client.get(payload["instrumental_audio_url"])
        original_response = self.client.get(payload['original']['audio_url'])
        self.assertEqual(vocal_response.status_code, 200)
        self.assertEqual(instrumental_response.status_code, 200)
        self.assertEqual(original_response.status_code, 200)
        self.assertEqual(vocal_response.mimetype, "audio/wav")
        self.assertGreater(len(vocal_response.data), 44)
        self.assertGreater(len(original_response.data), 44)

        for visualization_url in visualization_urls:
            visualization_response = self.client.get(visualization_url)
            self.assertEqual(visualization_response.status_code, 200)
            self.assertEqual(visualization_response.mimetype, 'image/png')
            self.assertTrue(visualization_response.data.startswith(b'\x89PNG\r\n\x1a\n'))
            visualization_response.close()

        job_directory = next((self.test_directory / 'outputs').iterdir())
        expected_visualizations = {
            f'{source}_{visualization}.png'
            for source in ('original', 'vocals', 'instrumental')
            for visualization in ('waveform', 'spectrogram')
        }
        self.assertTrue(expected_visualizations.issubset(
            {path.name for path in job_directory.iterdir()}
        ))

        download_response = self.client.get(
            payload['vocals']['download_url']
        )
        self.assertEqual(download_response.status_code, 200)
        self.assertIn("attachment", download_response.headers["Content-Disposition"])
        self.assertIn("vocals.wav", download_response.headers["Content-Disposition"])
        vocal_response.close()
        instrumental_response.close()
        original_response.close()
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

    def test_separate_returns_safe_error_when_processing_fails(self):
        self.mock_separator.side_effect = RuntimeError('private model details')
        response = self.client.post(
            "/api/separate",
            data={"file": (self.make_wav(), "stereo.wav")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"], "Music separation failed.")

    def test_separate_rejects_missing_file(self):
        response = self.client.post("/api/separate")
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])

    def test_output_route_rejects_unknown_files(self):
        response = self.client.get(f'/api/outputs/{"a" * 32}/unknown.wav')

        self.assertEqual(response.status_code, 404)
        self.assertFalse(response.get_json()["success"])


if __name__ == "__main__":
    unittest.main()
