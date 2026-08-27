function formatDuration(duration) {
  const roundedDuration = Math.round(duration * 100) / 100
  const minutes = Math.floor(roundedDuration / 60)
  const seconds = roundedDuration - minutes * 60
  return `${minutes}:${seconds.toFixed(2).padStart(5, '0')}`
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


function AudioResultCard({ title, source, downloadLabel, className = '' }) {
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
          alt={`${title} audio waveform over time`}
        />
        <Visualization
          label="Spectrogram"
          src={source.spectrogram_url}
          alt={`${title} audio frequency spectrogram`}
        />
      </div>
      {source.download_url && (
        <a className="download-button" href={source.download_url} download>
          {downloadLabel}
        </a>
      )}
    </article>
  )
}


function MusicResultsPage({ result, onStartOver }) {
  const { original, vocals, instrumental } = result

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

      <AudioResultCard
        title="Original"
        source={original}
        className="original-card"
      />

      <div className="stem-grid">
        <AudioResultCard
          title="Vocals"
          source={vocals}
          downloadLabel="Download Vocals"
        />
        <AudioResultCard
          title="Instrumental"
          source={instrumental}
          downloadLabel="Download Instrumental"
        />
      </div>
    </section>
  )
}


export default MusicResultsPage
