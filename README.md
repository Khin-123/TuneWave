# TuneWave

TuneWave is a university project combining classical digital signal
processing with pretrained machine learning for music separation. The planned
speech-enhancement path uses classical DSP.

## Final architecture

### Music separation

~~~
Stereo 44.1 kHz audio
→ 30-second chunks with 2-second overlap
→ Hann-window STFT
→ complex mixture spectrum and magnitude
→ pretrained Open-Unmix UMXL vocals model
→ residual multichannel Wiener reconstruction
→ instrumental complex spectrum = mixture − vocals
→ ISTFT
→ weighted overlap-add and shared clipping protection
→ vocals.wav + instrumental.wav
~~~

Open-Unmix supplies learned vocal-spectrum estimates. TuneWave owns the
stereo handling, STFT, magnitude/phase analysis, Wiener reconstruction,
mixture-consistent residual, ISTFT, overlap-add, and output protection.
The production path loads only the vocals UMXL target; it does not load
drums, bass, or other models. UMXL weights are pretrained and downloaded by
Open-Unmix on first use. They are distributed under CC BY-NC-SA 4.0.

The former classical Mid/Side separator in
'backend/dsp/source_separation.py' is retained as a DSP-only baseline for
comparison. The former custom U-Net/training experiment has been removed from
the application tree.

### Speech enhancement

~~~
Noisy speech
→ Hann/STFT
→ noise estimation
→ Wiener filtering
→ ISTFT
→ enhanced speech
~~~

Speech enhancement is specified as a classical-DSP path and does not use
Open-Unmix. No speech-enhancement Flask endpoint is currently present in this
repository.

## Requirements

- Python 3.10+
- Node.js 20.19+ or 22.12+
- 'backend/requirements.txt'
- Matching PyTorch and TorchAudio builds
- FFmpeg available on PATH when decoding formats unsupported by libsndfile

Install Python dependencies:

~~~
cd backend
python -m pip install -r requirements.txt
~~~

The application does not require MUSDB18. TuneWave does not train UMXL; the old
compressed MUSDB18 training workflow and Kaggle instructions are no longer part
of the final system. CPU inference is supported, though a short input is
recommended for faster demonstrations.

## Run Flask and React

From the repository root:

~~~
cd backend
python app.py
~~~

In a second terminal:

~~~
cd frontend
npm install
npm run dev
~~~

Flask runs at http://localhost:5000; Vite normally runs at
http://localhost:5173 and proxies /api requests to Flask.

## Music API

'POST /api/separate' accepts a stereo WAV, MP3, or FLAC upload in the
multipart field 'file'. It returns playback/download URLs for 'original',
'vocals', and 'instrumental', including waveform and spectrogram image URLs.
Each request receives an isolated output directory under
'backend/outputs/<job-id>/'.

The production implementation is
'backend/separation/openunmix_dsp_separator.py'. It supports CPU and uses
CUDA automatically when available. The cached model key includes model name
and device, so UMXL and UMXHQ weights cannot be accidentally mixed.

Run a bounded verification:

~~~
python test_openunmix_chunked_separator.py --input test_audio/test_song.wav --output-dir outputs/openunmix_umxl_test --chunk-seconds 30 --overlap-seconds 2 --device cpu --niter 2
~~~

The local 'test_audio/' directory is intentionally ignored. Place your own
short, stereo 44.1 kHz WAV/MP3/FLAC there (for example 'test_song.wav') before
running this command; do not commit commercial music.

The verifier reports chunk count, model-load/processing time, consistency
errors, shared gain, and validates stereo 44.1 kHz output.

## Tests

~~~
cd backend
python -m pytest tests/test_source_separation.py tests/test_openunmix_chunking.py tests/test_separation_api.py -q
~~~

Build the frontend:

~~~
cd frontend
npm run build
~~~

## Project structure

~~~
backend/
  app.py
  dsp/
    audio_utils.py
    audio_visualization.py
    mid_side.py
    source_separation.py       # DSP-only baseline
    stft_utils.py
  separation/
    openunmix_separator.py     # pretrained API proof of concept
    openunmix_dsp_separator.py # final production UMXL pipeline
  tests/
frontend/
  src/
~~~

Generated uploads and outputs are ignored by Git. Pretrained weights are
cached by PyTorch/Open-Unmix outside the source tree and are not committed.

## Limitations

Open-Unmix inference is compute-intensive on CPU, especially for long songs.
Chunking bounds peak spectral memory and overlap-add reduces boundary
discontinuities, but a long request can still take substantial time.
Separation quality depends on the pretrained UMXL model and may include vocal
or instrumental leakage. The system currently requires stereo 44.1 kHz input.
