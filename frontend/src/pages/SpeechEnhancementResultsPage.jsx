import AudioResultCard from '../components/AudioResultCard.jsx'
import WorkflowSteps from '../components/WorkflowSteps.jsx'

function formatDuration(duration) {
  const seconds = Number(duration)
  if (!Number.isFinite(seconds)) return 'Unavailable'
  return seconds.toFixed(2) + ' s'
}

function SpeechEnhancementResultsPage({ result, onStartOver }) {
  const { original, enhanced, processing = {} } = result
  const sampleRate = processing.sample_rate || original.sample_rate
  const duration = processing.duration ?? original.duration

  return (
    <section className="results-page" aria-labelledby="speech-results-heading">
      <div className="results-header">
        <div>
          <span className="page-kicker success-kicker"><span aria-hidden="true">✓</span> Enhancement complete</span>
          <h2 id="speech-results-heading">Your clearer speech</h2>
          <p className="result-filename"><span className="file-type-icon" aria-hidden="true">WAV</span>{original.filename}</p>
        </div>
        <button type="button" className="secondary-button" onClick={onStartOver}>
          <span aria-hidden="true">＋</span> Enhance another
        </button>
      </div>

      <WorkflowSteps currentStep={3} />

      <div className="result-summary">
        <div>
          <span className="summary-label">Two tracks ready</span>
          <strong>Compare the original recording with the enhanced version.</strong>
        </div>
        <dl className="result-metadata">
          <div><dt>Sample rate</dt><dd>{sampleRate ? (sampleRate / 1000).toFixed(1) + ' kHz' : 'Unavailable'}</dd></div>
          <div><dt>Duration</dt><dd>{formatDuration(duration)}</dd></div>
          <div><dt>Processing time</dt><dd>{formatDuration(processing.processing_time)}</dd></div>
        </dl>
      </div>

      <div className="result-section-heading">
        <span>01</span>
        <div>
          <h3>Compare your audio</h3>
          <p>Play either version and inspect the signal visually.</p>
        </div>
      </div>
      <div className="stem-grid speech-comparison-grid">
        <AudioResultCard title="Original" source={original} className="original-card" downloadLabel="Download original" />
        <AudioResultCard title="Enhanced Speech" source={enhanced} className="enhanced-card" downloadLabel="Download enhanced" />
      </div>
    </section>
  )
}

export default SpeechEnhancementResultsPage
