import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from dsp.stft_utils import istft, stft
from speech.wiener_enhancer import (
    SpeechEnhancementConfig,
    adaptive_subtraction_factor,
    compute_snr_db,
    compute_wiener_gain,
    enhance_audio,
    enhance_audio_file,
    estimate_noise_power,
    select_low_energy_noise_frames,
    smooth_wiener_gain,
    spectral_subtraction,
    speech_stft_parameters,
)


class SpeechEnhancementTestCase(unittest.TestCase):
    @staticmethod
    def synthetic_speech(sample_rate: int = 16_000, duration: float = 1.0):
        generator = np.random.default_rng(1234)
        samples = int(round(sample_rate * duration))
        time = np.arange(samples, dtype=np.float64) / sample_rate
        clean = (
            0.25 * np.sin(2 * np.pi * 180 * time)
            + 0.12 * np.sin(2 * np.pi * 360 * time)
            + 0.06 * np.sin(2 * np.pi * 720 * time)
        )
        envelope = np.clip((time - 0.2) / 0.05, 0.0, 1.0)
        clean *= envelope
        noise = generator.normal(0.0, 0.06, samples)
        return clean.astype(np.float32), (clean + noise).astype(np.float32), noise

    def test_mono_duration_and_finite_output_at_common_rates(self):
        for sample_rate in (8_000, 16_000, 44_100, 48_000):
            with self.subTest(sample_rate=sample_rate):
                _, noisy, _ = self.synthetic_speech(sample_rate, duration=0.6)
                enhanced, diagnostics = enhance_audio(noisy, sample_rate)
                self.assertEqual(enhanced.shape, noisy.shape)
                self.assertTrue(np.isfinite(enhanced).all())
                self.assertEqual(diagnostics["sample_rate"], sample_rate)
                self.assertEqual(diagnostics["channels"], 1)
                self.assertGreater(diagnostics["frames"], 0)
                self.assertEqual(diagnostics["noise_estimation"], "auto")
                self.assertGreaterEqual(diagnostics["noise_frames"], 8)
                self.assertLessEqual(diagnostics["noise_frames_percent"], 20.0)

    def test_default_stft_parameters_are_25_ms_with_75_percent_overlap(self):
        frame_length, hop_length = speech_stft_parameters(
            16_000,
            SpeechEnhancementConfig(),
        )
        self.assertEqual(frame_length, 400)
        self.assertEqual(hop_length, 100)

    def test_silence_remains_finite_silence(self):
        silence = np.zeros(8_000, dtype=np.float32)
        enhanced, diagnostics = enhance_audio(silence, 16_000)
        np.testing.assert_array_equal(enhanced, silence)
        self.assertEqual(diagnostics["output_peak"], 0.0)

    def test_synthetic_noise_only_prefix_is_attenuated(self):
        sample_rate = 16_000
        _, noisy, _ = self.synthetic_speech(sample_rate)
        enhanced, _ = enhance_audio(
            noisy,
            sample_rate,
            SpeechEnhancementConfig(
                noise_estimation="initial",
                noise_duration_seconds=0.2,
            ),
        )
        prefix_samples = int(0.15 * sample_rate)
        input_prefix_power = float(np.mean(noisy[:prefix_samples] ** 2))
        output_prefix_power = float(np.mean(enhanced[:prefix_samples] ** 2))
        self.assertLess(output_prefix_power, input_prefix_power)
        self.assertFalse(np.allclose(enhanced, noisy))

    def test_stereo_is_processed_per_channel_and_preserved(self):
        sample_rate = 16_000
        _, noisy, _ = self.synthetic_speech(sample_rate)
        stereo = np.vstack((noisy, 0.8 * noisy + 0.01)).astype(np.float32)
        enhanced, diagnostics = enhance_audio(stereo, sample_rate)
        self.assertEqual(enhanced.shape, stereo.shape)
        self.assertEqual(diagnostics["channels"], 2)
        self.assertTrue(np.isfinite(enhanced).all())
        self.assertFalse(np.array_equal(enhanced[0], enhanced[1]))

    def test_shared_clipping_protection(self):
        time = np.arange(8_000) / 8_000
        loud = (2.0 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
        enhanced, diagnostics = enhance_audio(loud, 8_000)
        self.assertLessEqual(float(np.max(np.abs(enhanced))), 0.99 + 1e-6)
        self.assertLessEqual(diagnostics["shared_output_gain"], 1.0)

    def test_wiener_gain_is_finite_and_bounded(self):
        noisy_power = np.array([[0.0, 1.0, 4.0], [0.5, 2.0, 8.0]])
        noise_power = np.array([[1.0], [0.5]])
        gain = compute_wiener_gain(
            noisy_power,
            noise_power,
            min_gain=0.10,
        )
        self.assertTrue(np.isfinite(gain).all())
        self.assertTrue(np.all(gain >= 0.10))
        self.assertTrue(np.all(gain <= 1.0))
        expected = np.maximum(noisy_power - noise_power, 0.0)
        expected /= expected + noise_power + 1e-10
        np.testing.assert_allclose(gain, np.clip(expected, 0.10, 1.0))

        one_dimensional_gain = compute_wiener_gain(
            noisy_power,
            np.array([1.0, 0.5]),
        )
        np.testing.assert_allclose(one_dimensional_gain, gain)

    def test_low_energy_selector_uses_requested_fraction_and_minimum(self):
        energy = np.array([9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0, 10.0])
        selected = select_low_energy_noise_frames(
            energy,
            frame_percent=0.2,
            minimum_frames=3,
        )
        np.testing.assert_array_equal(selected, np.array([1, 3, 5]))

        short_selected = select_low_energy_noise_frames(
            energy[:4],
            frame_percent=0.15,
            minimum_frames=8,
        )
        np.testing.assert_array_equal(short_selected, np.arange(4))

    def test_gain_smoothing_is_moderate_finite_and_bounded(self):
        gain = np.ones((7, 7), dtype=np.float64)
        gain[3, 3] = 0.10
        smoothed = smooth_wiener_gain(gain, min_gain=0.10)
        self.assertTrue(np.isfinite(smoothed).all())
        self.assertTrue(np.all(smoothed >= 0.10))
        self.assertTrue(np.all(smoothed <= 1.0))
        self.assertGreater(smoothed[3, 3], gain[3, 3])
        self.assertLess(smoothed[3, 2], gain[3, 2])
        self.assertLess(float(np.var(smoothed)), float(np.var(gain)))

    def test_initial_noise_estimation_remains_available(self):
        sample_rate = 16_000
        _, noisy, _ = self.synthetic_speech(sample_rate)
        _, diagnostics = enhance_audio(
            noisy,
            sample_rate,
            SpeechEnhancementConfig(
                noise_estimation="initial",
                noise_duration_seconds=0.25,
            ),
        )
        self.assertEqual(diagnostics["noise_estimation"], "initial")
        self.assertAlmostEqual(
            diagnostics["noise_estimation_duration_seconds"],
            0.25,
        )

    def test_requested_quality_diagnostics_are_finite_percentages(self):
        sample_rate = 16_000
        _, noisy, _ = self.synthetic_speech(sample_rate)
        _, diagnostics = enhance_audio(noisy, sample_rate)
        for key in ("input_rms", "output_rms", "mean_wiener_gain"):
            self.assertTrue(np.isfinite(diagnostics[key]))
        self.assertGreaterEqual(diagnostics["mean_wiener_gain"], 0.10)
        self.assertLessEqual(diagnostics["mean_wiener_gain"], 1.0)
        for key in (
            "noise_frames_percent",
            "bins_attenuated_over_3_db_percent",
            "bins_attenuated_over_6_db_percent",
        ):
            self.assertGreaterEqual(diagnostics[key], 0.0)
            self.assertLessEqual(diagnostics[key], 100.0)
        self.assertGreaterEqual(
            diagnostics["bins_attenuated_over_3_db_percent"],
            diagnostics["bins_attenuated_over_6_db_percent"],
        )

    def test_rejects_empty_nonfinite_and_multichannel_audio(self):
        invalid_inputs = (
            np.array([], dtype=np.float32),
            np.array([0.0, np.nan], dtype=np.float32),
            np.zeros((3, 100), dtype=np.float32),
        )
        for audio in invalid_inputs:
            with self.subTest(shape=audio.shape):
                with self.assertRaises(ValueError):
                    enhance_audio(audio, 16_000)

    def test_noise_percentile_is_robust_to_one_loud_frame(self):
        spectrum = np.ones((4, 10), dtype=np.complex128)
        spectrum[:, 3] = 100.0
        estimate = estimate_noise_power(spectrum, 0, 10, percentile=20.0)
        np.testing.assert_allclose(estimate, 1.0)

    def test_file_enhancement_writes_expected_mono_wavs(self):
        sample_rate = 16_000
        _, noisy, _ = self.synthetic_speech(sample_rate, duration=0.4)
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "noisy.wav"
            sf.write(input_path, noisy, sample_rate, subtype="PCM_16")
            result = enhance_audio_file(input_path, directory / "outputs")
            self.assertEqual(
                set(result["paths"]),
                {"original_noisy", "enhanced_speech"},
            )
            for path in result["paths"].values():
                info = sf.info(path)
                self.assertTrue(path.is_file())
                self.assertEqual(info.samplerate, sample_rate)
                self.assertEqual(info.channels, 1)
                self.assertEqual(info.frames, noisy.size)


    def test_auto_noise_estimation_ranks_low_energy_frames_over_full_recording(self):
        sample_rate = 16_000
        _, noisy, _ = self.synthetic_speech(sample_rate, duration=1.0)
        _, diagnostics = enhance_audio(noisy, sample_rate)
        self.assertEqual(diagnostics["noise_estimation"], "auto")
        self.assertEqual(diagnostics["noise_estimation_method"], "low_energy_frames")
        self.assertIsNone(diagnostics["noise_estimation_duration_seconds"])
        self.assertGreaterEqual(diagnostics["noise_frames_selected"], 8)
        self.assertLess(diagnostics["noise_frames_selected"], diagnostics["number_of_frames"])

    def test_spectral_subtraction_obeys_configured_floor(self):
        noisy_magnitude = np.array([[10.0, 2.0], [1.0, 0.2]])
        noise_magnitude = np.array([[8.0], [0.5]])
        beta = 2.0
        spectral_floor = 0.10
        enhanced = spectral_subtraction(
            noisy_magnitude,
            noise_magnitude,
            beta=beta,
            spectral_floor=spectral_floor,
        )
        expected = np.maximum(
            noisy_magnitude - beta * noise_magnitude,
            spectral_floor * noisy_magnitude,
        )
        np.testing.assert_allclose(enhanced, expected)
        self.assertTrue(np.all(enhanced[noisy_magnitude > 0] > 0))

    def test_fft_ifft_reconstruction_sanity_for_speech_frames(self):
        sample_rate = 16_000
        samples = int(0.37 * sample_rate) + 17
        time = np.arange(samples) / sample_rate
        original = 0.4 * np.sin(2 * np.pi * 440 * time)
        spectrum = stft(original, sample_rate, n_fft=400, hop_length=100)
        reconstructed = istft(
            spectrum,
            sample_rate,
            n_fft=400,
            hop_length=100,
            length=samples,
        )
        np.testing.assert_allclose(reconstructed, original, atol=1e-10)

    def test_synthetic_noisy_speech_has_measurable_error_reduction(self):
        sample_rate = 16_000
        clean, noisy, _ = self.synthetic_speech(sample_rate, duration=1.0)
        enhanced, diagnostics = enhance_audio(noisy, sample_rate)
        noisy_error = float(np.mean((noisy - clean) ** 2))
        enhanced_error = float(np.mean((enhanced - clean) ** 2))
        self.assertLess(enhanced_error, noisy_error * 0.75)
        self.assertGreater(diagnostics["subtraction_attenuated_bins_percent"], 0.0)

    def test_output_level_compensation_reports_and_restores_level(self):
        sample_rate = 16_000
        _, noisy, _ = self.synthetic_speech(sample_rate, duration=1.0)
        enhanced, diagnostics = enhance_audio(noisy, sample_rate)
        self.assertTrue(np.isfinite(enhanced).all())
        self.assertLess(diagnostics["output_rms_before_compensation"], diagnostics["input_rms"])
        self.assertGreater(diagnostics["output_rms_after_compensation"], diagnostics["output_rms_before_compensation"])
        self.assertGreater(diagnostics["compensation_gain"], 1.0)
        self.assertGreaterEqual(
            diagnostics["output_rms_after_compensation"],
            diagnostics["input_rms"] * 0.90,
        )

    def test_stereo_file_preserves_channel_count_sample_rate_and_duration(self):
        sample_rate = 16_000
        _, noisy, _ = self.synthetic_speech(sample_rate, duration=0.4)
        stereo = np.column_stack((noisy, 0.65 * noisy)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "stereo_noisy.wav"
            sf.write(input_path, stereo, sample_rate, subtype="PCM_16")
            result = enhance_audio_file(input_path, directory / "outputs")
            info = sf.info(result["paths"]["enhanced_speech"])
            self.assertEqual(info.samplerate, sample_rate)
            self.assertEqual(info.channels, 2)
            self.assertEqual(info.frames, stereo.shape[0])
            enhanced, _ = enhance_audio(stereo.T, sample_rate)
            self.assertEqual(enhanced.shape, stereo.T.shape)
            self.assertTrue(np.isfinite(enhanced).all())
            self.assertLessEqual(float(np.max(np.abs(enhanced))), 0.99 + 1e-6)
    def test_adaptive_beta_is_stronger_in_noise_and_weaker_in_speech(self):
        local_snr_db = np.array([-10.0, 0.0, 7.5, 15.0, 30.0])
        beta = adaptive_subtraction_factor(local_snr_db)
        self.assertTrue(np.isfinite(beta).all())
        self.assertGreaterEqual(float(np.min(beta)), 0.8)
        self.assertLessEqual(float(np.max(beta)), 1.8)
        self.assertGreater(beta[0], beta[-1])
        self.assertGreater(beta[1], beta[3])
        np.testing.assert_allclose(beta[[0, 1, 3, 4]], [1.8, 1.8, 0.8, 0.8])

    def test_adaptive_diagnostics_include_snr_beta_and_noise_tracking(self):
        sample_rate = 16_000
        clean, noisy, _ = self.synthetic_speech(sample_rate, duration=1.0)
        _, diagnostics = enhance_audio(noisy, sample_rate, reference_audio=clean)
        for key in (
            "mean_local_snr",
            "beta_min",
            "beta_mean",
            "beta_max",
            "spectral_floor",
            "input_rms",
            "output_rms_after_compensation",
            "processing_time",
            "synthetic_input_snr",
            "synthetic_output_snr",
            "snr_improvement",
        ):
            self.assertIsNotNone(diagnostics[key])
            self.assertTrue(np.isfinite(diagnostics[key]))
        self.assertGreaterEqual(diagnostics["beta_min"], 0.8)
        self.assertLessEqual(diagnostics["beta_max"], 1.8)
        self.assertGreater(diagnostics["beta_max"], diagnostics["beta_min"])
        self.assertGreater(diagnostics["number_of_noise_dominant_frames"], 0)
        self.assertGreater(diagnostics["snr_improvement"], 0.0)
        self.assertAlmostEqual(
            diagnostics["synthetic_input_snr"],
            compute_snr_db(clean, noisy),
            places=6,
        )

    def test_speech_activity_guard_reports_active_frames(self):
        sample_rate = 16_000
        clean, noisy, _ = self.synthetic_speech(sample_rate, duration=1.0)
        _, diagnostics = enhance_audio(
            noisy,
            sample_rate,
            SpeechEnhancementConfig(speech_activity_margin_db=6.0),
            reference_audio=clean,
        )
        self.assertGreater(diagnostics["number_of_speech_active_frames"], 0)
        self.assertGreaterEqual(
            diagnostics["number_of_speech_active_frames"],
            diagnostics["number_of_noise_dominant_frames"],
        )
        self.assertEqual(diagnostics["speech_activity_margin_db"], 6.0)

    def test_speech_activity_margin_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            SpeechEnhancementConfig(speech_activity_margin_db=-1.0)

    def test_adaptive_beats_fixed_beta_on_same_synthetic_input(self):
        sample_rate = 16_000
        clean, noisy, _ = self.synthetic_speech(sample_rate, duration=1.0)
        fixed, fixed_diagnostics = enhance_audio(
            noisy,
            sample_rate,
            SpeechEnhancementConfig(
                beta=1.5,
                adaptive_beta=False,
                noise_tracking_alpha=0.0,
            ),
            reference_audio=clean,
        )
        adaptive, adaptive_diagnostics = enhance_audio(
            noisy,
            sample_rate,
            reference_audio=clean,
        )
        self.assertEqual(fixed.shape, adaptive.shape)
        self.assertGreater(
            adaptive_diagnostics["synthetic_output_snr"],
            fixed_diagnostics["synthetic_output_snr"] + 0.1,
        )
        self.assertLess(adaptive_diagnostics["beta_min"], adaptive_diagnostics["beta_max"])
if __name__ == "__main__":
    unittest.main()
