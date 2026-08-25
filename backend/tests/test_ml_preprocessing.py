import unittest

import numpy as np

from dsp.ml_preprocessing import (
    MagnitudeNormalizationStats,
    PreprocessingConfig,
    extract_stereo_features,
    iter_stereo_chunks,
    normalize_magnitude,
    prepare_model_audio,
    preprocess_stereo_chunks,
    stereo_istft,
    stereo_stft,
)


class MlPreprocessingTestCase(unittest.TestCase):
    @staticmethod
    def make_stereo_audio(sample_rate=8_000, sample_count=1_100):
        time = np.arange(sample_count, dtype=np.float64) / sample_rate
        left = 0.4 * np.sin(2 * np.pi * 440 * time)
        right = 0.3 * np.sin(2 * np.pi * 660 * time + 0.2)
        return np.vstack((left, right)).astype(np.float32)

    def setUp(self):
        self.config = PreprocessingConfig(
            sample_rate=8_000,
            n_fft=256,
            hop_length=64,
            chunk_seconds=0.04,
            chunk_hop_seconds=0.02,
        )

    def test_prepare_model_audio_resamples_and_preserves_stereo(self):
        original = self.make_stereo_audio(sample_rate=4_000, sample_count=400)

        prepared, sample_rate = prepare_model_audio(
            original,
            sample_rate=4_000,
            target_sample_rate=8_000,
        )

        self.assertEqual(sample_rate, 8_000)
        self.assertEqual(prepared.shape, (2, 800))
        self.assertEqual(prepared.dtype, np.float32)
        self.assertTrue(np.isfinite(prepared).all())

    def test_chunks_have_fixed_shape_and_cover_final_samples(self):
        original = self.make_stereo_audio(sample_count=700)
        chunks = tuple(iter_stereo_chunks(original, self.config))

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk.audio.shape == (2, 320) for chunk in chunks))
        self.assertEqual(chunks[0].start_sample, 0)
        self.assertEqual(chunks[-1].start_sample, 380)
        self.assertEqual(chunks[-1].valid_samples, 320)

        short_audio = original[:, :100]
        short_chunk = tuple(iter_stereo_chunks(short_audio, self.config))[0]
        self.assertEqual(short_chunk.valid_samples, 100)
        np.testing.assert_array_equal(short_chunk.audio[:, 100:], 0.0)

    def test_stereo_stft_and_istft_reconstruct_both_channels(self):
        original = self.make_stereo_audio(sample_count=1_173)

        spectra = stereo_stft(original, self.config.sample_rate, self.config)
        reconstructed = stereo_istft(
            spectra,
            self.config.sample_rate,
            self.config,
            length=original.shape[1],
        )

        self.assertEqual(spectra.shape[0], 2)
        self.assertEqual(spectra.shape[1], self.config.n_fft // 2 + 1)
        self.assertEqual(reconstructed.shape, original.shape)
        np.testing.assert_allclose(reconstructed, original, atol=1e-6)

    def test_feature_shapes_keep_complex_spectra_separate_from_normalized_input(self):
        original = self.make_stereo_audio(sample_count=self.config.chunk_samples)
        features = extract_stereo_features(
            original,
            self.config.sample_rate,
            self.config,
        )

        expected_shape = (2, self.config.n_fft // 2 + 1, features.magnitude.shape[-1])
        self.assertEqual(features.complex_stft.shape, expected_shape)
        self.assertEqual(features.magnitude.shape, expected_shape)
        self.assertEqual(features.normalized_magnitude.shape, expected_shape)
        self.assertTrue(np.iscomplexobj(features.complex_stft))
        self.assertTrue(np.issubdtype(features.normalized_magnitude.dtype, np.floating))
        self.assertTrue(np.isfinite(features.normalized_magnitude).all())
        self.assertFalse(np.shares_memory(features.magnitude, features.normalized_magnitude))

    def test_normalization_can_reuse_frozen_statistics(self):
        magnitude = np.ones((2, 5, 7), dtype=np.float32)
        stats = MagnitudeNormalizationStats(
            mean=np.full((2, 1, 1), 0.5, dtype=np.float32),
            std=np.full((2, 1, 1), 0.25, dtype=np.float32),
        )

        normalized, used_stats = normalize_magnitude(magnitude, stats=stats)

        self.assertEqual(normalized.shape, magnitude.shape)
        np.testing.assert_array_equal(used_stats.mean, stats.mean)
        np.testing.assert_array_equal(used_stats.std, stats.std)
        np.testing.assert_allclose(normalized, (np.log1p(magnitude) - 0.5) / 0.25)

    def test_full_chunk_preprocessing_returns_matching_chunks_and_features(self):
        original = self.make_stereo_audio(sample_count=700)

        sample_rate, chunks, features = preprocess_stereo_chunks(
            original,
            sample_rate=self.config.sample_rate,
            config=self.config,
        )

        self.assertEqual(sample_rate, self.config.sample_rate)
        self.assertEqual(len(chunks), len(features))
        self.assertTrue(
            all(
                feature.normalized_magnitude.shape[0] == 2
                for feature in features
            )
        )


if __name__ == "__main__":
    unittest.main()
