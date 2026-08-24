# TuneWave

TuneWave is a university project for classical DSP-based audio source separation and speech enhancement. The current modules provide a React/Vite frontend, a Flask API, and basic audio upload and metadata extraction. DSP processing is intentionally not implemented yet.

## Prerequisites

- Node.js 18 or newer
- Python 3.10 or newer

## Run the backend

From the `tunewave` directory:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

The API runs at `http://localhost:5000`. Verify it directly at `http://localhost:5000/api/health`.

## Run the frontend

In a second terminal, from the `tunewave` directory:

```powershell
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite (normally `http://localhost:5173`). With the backend running, the page displays `Backend: connected`.

## Module 1 API

### `GET /api/health`

Response:

```json
{
  "status": "ok"
}
```

## Module 2 API

### `POST /api/audio-info`

Send a WAV, MP3, or FLAC file as the `file` field in `multipart/form-data`:

```powershell
curl.exe -X POST -F "file=@C:\path\to\audio.wav" http://localhost:5000/api/audio-info
```

The response contains the original filename, sample rate, duration in seconds, channel count, and number of samples per channel.

## Run backend tests

With the backend virtual environment activated:

```powershell
cd backend
python -m unittest discover -s tests -v
```

## Module 3 DSP utilities

`backend/dsp/stft_utils.py` provides reusable functions for:

- Standard Hann window generation
- Overlapping audio framing
- Complex STFT calculation
- Magnitude and phase extraction
- ISTFT overlap-add reconstruction with exact output-length handling

The utilities process one real-valued audio channel at a time. Run the backend
test command above to verify generated sine-wave and short-signal STFT/ISTFT
reconstruction.

## Module 4 mid-side processing

`backend/dsp/mid_side.py` provides reversible classical stereo mid-side
processing:

- `encode_mid_side(stereo_audio)` converts channel-first left/right audio into
  separate mid and side signals.
- `decode_mid_side(mid, side)` reconstructs channel-first stereo audio.

Mono input is rejected because mid-side processing requires distinct left and
right channels. The backend test suite verifies the equations, reconstruction,
and invalid-input handling.

## Module 5 vocal and instrumental separation

`backend/dsp/source_separation.py` implements classical stereo separation with
this pipeline:

```text
Stereo audio -> Mid/Side -> STFT -> Soft spectral masks -> ISTFT -> Stems
```

The vocal mask favors mid-dominant time-frequency regions. Its complementary
instrumental mask receives the remaining mid content, while the full side
signal is retained in the instrumental stem. `separate_audio_file()` writes
`vocals.wav` and `instrumental.wav` to `backend/outputs` by default.

This approach assumes vocals are centered. Centered instruments can leak into
the vocal stem, and off-center vocals can leak into the instrumental stem.

## Module 6 music-separation API and UI

### `POST /api/separate`

Send stereo WAV, MP3, or FLAC audio in the multipart field named `file`. The
route calls `dsp/source_separation.py` and returns original audio information
plus playback URLs for the generated vocal and instrumental stems.

Generated files are stored in a unique subdirectory under `backend/outputs`.
The returned URLs support inline playback, and appending `?download=1` serves a
stem as a download.

The React music-separation flow provides original preview, processing status,
result players, and download buttons. Start Flask and Vite using the commands
above, then open the Vite URL to test the complete flow.
