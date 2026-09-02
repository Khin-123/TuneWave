import { useState } from 'react'

function AudioResultCard({ title, source = {}, downloadLabel = `Download ${title}`, className = '' }) {
  const availableView = source.waveform_url ? 'waveform' : 'spectrogram'
  const [view, setView] = useState(availableView)
  const image = view === 'waveform' ? source.waveform_url : source.spectrogram_url

  return (
    <article className={`audio-card ${className}`.trim()}>
      <div className="audio-card-header">
        <div className="audio-card-title">
          <span className="track-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none">
              <path d="M7 18V6.5L18 4v11.5M7 18a2.5 2.5 0 1 1-2.5-2.5A2.5 2.5 0 0 1 7 18Zm11-2.5a2.5 2.5 0 1 1-2.5-2.5 2.5 2.5 0 0 1 2.5 2.5Z" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </span>
          <div>
            <h3>{title}</h3>
            <span className="track-type">{title === 'Original' ? 'Source audio' : 'Separated track'}</span>
          </div>
        </div>
        {source.download_url && (
          <a className="download-button" href={source.download_url} download>
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path d="M12 4v10m0 0 4-4m-4 4-4-4M5 19h14" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            {downloadLabel}
          </a>
        )}
      </div>

      {source.audio_url ? (
        <audio controls preload="metadata" src={source.audio_url}>
          Your browser does not support audio playback.
        </audio>
      ) : (
        <p className="audio-unavailable">Audio preview unavailable.</p>
      )}

      <div className="visualization-panel">
        <div className="visualization-toolbar">
          <div className="visualization-tabs" role="tablist" aria-label={`${title} visualizations`}>
            <button
              type="button"
              role="tab"
              aria-selected={view === 'waveform'}
              className={view === 'waveform' ? 'is-active' : ''}
              onClick={() => setView('waveform')}
              disabled={!source.waveform_url}
            >
              Waveform
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={view === 'spectrogram'}
              className={view === 'spectrogram' ? 'is-active' : ''}
              onClick={() => setView('spectrogram')}
              disabled={!source.spectrogram_url}
            >
              Spectrogram
            </button>
          </div>
        </div>
        {image ? (
          <figure className="visualization">
            <img src={image} alt={`${title} ${view} visualization`} loading="lazy" />
          </figure>
        ) : (
          <div className="visualization-empty">Visualization unavailable for this track.</div>
        )}
      </div>
    </article>
  )
}

export default AudioResultCard

