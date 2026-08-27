function formatDuration(duration) {
  const seconds = Number(duration)
  if (!Number.isFinite(seconds)) return 'Unavailable'
  return `${seconds.toFixed(2)} s`
}


function Visualization({ label, src, alt }) {
  if (!src) return null

  return (
    <figure className="visualization">
      <figcaption>{label}</figcaption>
      <div className="visualization-frame">
        <img src={src} alt={alt} loading="lazy" />
      </div>
    </figure>
  )
}


function SpeechAudioCard({ title, source, download, className = '' }) {
  return (
    <article className={`audio-card ${className}`.trim()}>
      <h3>{title}</h3>
      {source.audio_url && (
        <audio controls preload="metadata" src={source.audio_url}>
          Your browser does not support audio playback.
        </audio>
      )}
      <div className="visualization-grid">
        <Visualization
          label="Waveform"
          src={source.waveform_url}
          alt={`${title} waveform over time`}
        />
        <Visualization
          label="Spectrogram"
          src={source.spectrogram_url}
          alt={`${title} time-frequency spectrogram`}
        />
      </div>
      {download && source.download_url && (
        <a className="download-button" href={source.download_url} download>
          Download Enhanced Audio
        </a>
      )}
    </article>
  )
}


function SpeechEnhancementResultsPage({ result, onStartOver }) {
  const { original, enhanced, processing = {} } = result
  const sampleRate = processing.sample_rate || original.sample_rate
  const duration = processing.duration ?? original.duration

  return (
    <section className="results-page" aria-labelledby="speech-results-heading">
      <div className="results-header">
        <div>
          <h2 id="speech-results-heading">Speech enhancement result</h2>
          <p>{original.filename}</p>
        </div>
        <button type="button" className="secondary-button" onClick={onStartOver}>
          Enhance another file
        </button>
      </div>

      <dl className="result-metadata">
        <div><dt>Sample rate</dt><dd>{sampleRate ? `${(sampleRate / 1000).toFixed(1)} kHz` : 'Unavailable'}</dd></div>
        <div><dt>Duration</dt><dd>{formatDuration(duration)}</dd></div>
        <div><dt>Noise estimation</dt><dd>{processing.noise_estimation || 'Automatic'}</dd></div>
        <div><dt>Processing time</dt><dd>{formatDuration(processing.processing_time)}</dd></div>
      </dl>

      <div className="stem-grid speech-comparison-grid">
        <SpeechAudioCard
          title="Original"
          source={original}
          className="original-card"
        />
        <SpeechAudioCard
          title="Enhanced Speech"
          source={enhanced}
          className="enhanced-card"
          download
        />
      </div>
    </section>
  )
}


export default SpeechEnhancementResultsPage
