"""Classical DSP speech-enhancement modules for TuneWave."""

from .wiener_enhancer import (
    SpeechEnhancementConfig,
    apply_mild_wiener_postfilter,
    compute_wiener_gain,
    enhance_audio,
    enhance_audio_file,
    estimate_noise_magnitude_from_frames,
    estimate_noise_power,
    estimate_noise_power_from_frames,
    select_low_energy_noise_frames,
    smooth_wiener_gain,
    spectral_subtract,
    spectral_subtraction,
    speech_stft_parameters,
)

__all__ = [
    "SpeechEnhancementConfig",
    "apply_mild_wiener_postfilter",
    "compute_wiener_gain",
    "enhance_audio",
    "enhance_audio_file",
    "estimate_noise_magnitude_from_frames",
    "estimate_noise_power",
    "estimate_noise_power_from_frames",
    "select_low_energy_noise_frames",
    "smooth_wiener_gain",
    "spectral_subtract",
    "spectral_subtraction",
    "speech_stft_parameters",
]