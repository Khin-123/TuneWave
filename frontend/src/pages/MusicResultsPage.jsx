import AudioResultCard from '../components/AudioResultCard.jsx'

function formatDuration(duration) {
  const roundedDuration = Math.round(duration * 100) / 100
  const minutes = Math.floor(roundedDuration / 60)
  const seconds = roundedDuration - minutes * 60
  return minutes + ':' + seconds.toFixed(2).padStart(5, '0')
}

function MusicResultsPage({ result, onStartOver }) {
  const { original, vocals, instrumental } = result

  return (
    <section className="results-page music-results-page" aria-labelledby="results-heading">
      <div className="results-header">
        <div>
          <span className="page-kicker success-kicker">
            <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
              <path d="m5.5 10.2 2.8 2.8 6.2-6.3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            Results ready
          </span>
          <h2 id="results-heading">Separation Complete</h2>
          <p>Listen, compare, and download your tracks.</p>
          <p className="result-filename"><span className="file-type-icon" aria-hidden="true">WAV</span>{original.filename}</p>
        </div>
        <button type="button" className="secondary-button" onClick={onStartOver}>
          <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
            <path d="M10 4v12M4 10h12" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
          </svg>
          Separate Another File
        </button>
      </div>

      <dl className="result-metadata compact-metadata">
        <div><dt>Duration</dt><dd>{formatDuration(original.duration)}</dd></div>
        <div><dt>Sample rate</dt><dd>{original.sample_rate.toLocaleString()} Hz</dd></div>
        <div><dt>Channels</dt><dd>{original.channels}</dd></div>
      </dl>

      <div className="music-track-layout">
        <AudioResultCard title="Original" source={original} className="original-card" downloadLabel="Download original" />
        <div className="stem-grid">
          <AudioResultCard title="Vocals" source={vocals} downloadLabel="Download vocals" />
          <AudioResultCard title="Instrumental" source={instrumental} downloadLabel="Download instrumental" />
        </div>
      </div>
    </section>
  )
}

export default MusicResultsPage
