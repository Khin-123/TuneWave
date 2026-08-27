"""Classical DSP speech-enhancement modules for TuneWave."""

from .wiener_enhancer import (
    SpeechEnhancementConfig,
    compute_wiener_gain,
    enhance_audio,
    enhance_audio_file,
    estimate_noise_power,
    estimate_noise_power_from_frames,
    select_low_energy_noise_frames,
    smooth_wiener_gain,
    speech_stft_parameters,
)

__all__ = [
    "SpeechEnhancementConfig",
    "compute_wiener_gain",
    "enhance_audio",
    "enhance_audio_file",
    "estimate_noise_power",
    "estimate_noise_power_from_frames",
    "select_low_energy_noise_frames",
    "smooth_wiener_gain",
    "speech_stft_parameters",
]
