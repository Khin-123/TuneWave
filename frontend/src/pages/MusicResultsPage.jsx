import { useEffect, useState } from 'react'


function formatDuration(duration) {
  const roundedDuration = Math.round(duration * 100) / 100
  const minutes = Math.floor(roundedDuration / 60)
  const seconds = roundedDuration - minutes * 60
  return `${minutes}:${seconds.toFixed(2).padStart(5, '0')}`
}


function downloadUrl(audioUrl) {
  return `${audioUrl}${audioUrl.includes('?') ? '&' : '?'}download=1`
}


function MusicResultsPage({ result, onStartOver }) {
  const [originalAudioUrl, setOriginalAudioUrl] = useState('')

  useEffect(() => {
    const objectUrl = URL.createObjectURL(result.originalFile)
    setOriginalAudioUrl(objectUrl)
    return () => URL.revokeObjectURL(objectUrl)
  }, [result.originalFile])

  const { original } = result

  return (
    <section className="results-page" aria-labelledby="results-heading">
      <div className="results-header">
        <div>
          <h2 id="results-heading">Separation results</h2>
          <p>{original.filename}</p>
        </div>
        <button type="button" className="secondary-button" onClick={onStartOver}>
          Separate another file
        </button>
      </div>

      <dl className="result-metadata">
        <div><dt>Duration</dt><dd>{formatDuration(original.duration)}</dd></div>
        <div><dt>Sample rate</dt><dd>{original.sample_rate.toLocaleString()} Hz</dd></div>
        <div><dt>Channels</dt><dd>{original.channels}</dd></div>
        <div><dt>Samples</dt><dd>{original.samples.toLocaleString()}</dd></div>
      </dl>

      <article className="audio-card original-card">
        <h3>Original</h3>
        {originalAudioUrl && (
          <audio controls preload="metadata" src={originalAudioUrl}>
            Your browser does not support audio playback.
          </audio>
        )}
      </article>

      <div className="stem-grid">
        <article className="audio-card">
          <h3>Vocals</h3>
          <audio controls preload="metadata" src={result.vocal_audio_url}>
            Your browser does not support audio playback.
          </audio>
          <a
            className="download-button"
            href={downloadUrl(result.vocal_audio_url)}
            download="vocals.wav"
          >
            Download vocals
          </a>
        </article>

        <article className="audio-card">
          <h3>Instrumental</h3>
          <audio controls preload="metadata" src={result.instrumental_audio_url}>
            Your browser does not support audio playback.
          </audio>
          <a
            className="download-button"
            href={downloadUrl(result.instrumental_audio_url)}
            download="instrumental.wav"
          >
            Download instrumental
          </a>
        </article>
      </div>
    </section>
  )
}


export default MusicResultsPage
