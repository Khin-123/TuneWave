import { useEffect, useRef, useState } from 'react'
import { separateAudio, enhanceSpeech } from '../services/api.js'

export function Mark() {
  return <svg viewBox="0 0 40 40" fill="none" aria-hidden="true"><path d="M4 20h5l4-12 7 24 7-24 4 12h5" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/></svg>
}
const copy = {
  music: { title: 'Find every layer.', description: 'Turn one song into vocals and instrumental. Listen closer. Make it yours.', action: 'Separate audio', working: 'Separating audio...', formats: ['wav','mp3','flac'], tracks: ['Original','Vocals','Instrumental'] },
  speech: { title: 'Let your voice through.', description: 'Reduce the background noise. Bring the conversation into focus.', action: 'Enhance speech', working: 'Enhancing speech...', formats: ['wav','mp3','flac'], tracks: ['Original','Enhanced Speech'] },
}
export function UploadStudio({ mode, onComplete, onBusy = () => {} }) {
  const c = copy[mode]
  const input = useRef(null)
  const preview = useRef(null)
  const mounted = useRef(true)
  const [file, setFile] = useState(null)
  const [url, setUrl] = useState('')
  const [drag, setDrag] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false; onBusy(false) }
  }, [])
  useEffect(() => {
    if (!file) { setUrl(''); return }
    const objectUrl = URL.createObjectURL(file)
    setUrl(objectUrl)
    return () => URL.revokeObjectURL(objectUrl)
  }, [file])
  function select(next) {
    if (busy || !next) return
    if (!c.formats.includes(next.name.split('.').pop().toLowerCase())) {
      setError('Choose a ' + c.formats.join(', ').toUpperCase() + ' audio file.')
      return
    }
    setError(''); setFile(next)
  }
  async function process() {
    if (!file || busy) return
    preview.current?.pause()
    setBusy(true); onBusy(true); setError('')
    try {
      const result = await (mode === 'music' ? separateAudio(file) : enhanceSpeech(file))
      if (mounted.current) onComplete(result)
    } catch (e) {
      if (mounted.current) setError(e.name === 'TypeError' ? 'Unable to connect. Please try again when the service is available.' : e.message || 'Processing failed. Please try again.')
    } finally { if (mounted.current) { setBusy(false); onBusy(false) } }
  }
  return <section className="upload-layout">
    <div className="intro">
      <span className="eyebrow">{mode === 'music' ? 'Music Separation' : 'Speech Enhancement'}</span>
      <h1>{c.title}</h1><p className="lead">{c.description}</p>
      <div className="sound-art" aria-hidden="true">{Array.from({length: 39}, (_, i) => <i key={i} style={{height: (12 + Math.abs(Math.sin(i * .72) * Math.cos(i * .17)) * 85) + '%'}} />)}</div>
      <div className="deliverables"><span className="eyebrow">What you get</span>{c.tracks.map((t,i) => <div key={t}><span className="track-number">0{i+1}</span><span>{t}</span><span className="muted">{i ? 'Processed' : 'Your source'}</span></div>)}</div>
    </div>
    <div className="upload-surface">
      <div className="surface-heading"><span>01 / {busy ? 'Processing' : 'Upload audio'}</span><span className="muted">{c.formats.join(' / ').toUpperCase()}</span></div>
      <input ref={input} type="file" className="sr-only" aria-label="Choose audio file" disabled={busy} accept={c.formats.map(f => '.'+f).join(',')} onChange={e => { select(e.target.files[0]); e.target.value = '' }} />
      {busy ? <div className="processing" role="status"><div className="processing-bars" aria-hidden="true">{[0,1,2,3,4].map(i => <i key={i} style={{animationDelay: i * .13 + 's'}} />)}</div><h2>{c.working}</h2><p>This can take a moment. Keep this page open.</p><small>{file.name}</small></div> :
      file ? <div className="file-ready"><span className="ready-label">Ready to process</span><h2>{file.name}</h2><p>{(file.size / 1024 / 1024).toFixed(2)} MB</p><audio ref={preview} controls src={url} preload="metadata" aria-label="Selected audio preview"/><div className="file-actions"><button onClick={() => input.current.click()}>Change file</button><button onClick={() => { setFile(null); setError('') }}>Remove</button></div></div> :
      <div className={'drop-zone' + (drag ? ' dragging' : '')} onDragOver={e => {e.preventDefault(); setDrag(true)}} onDragLeave={e => { if (!e.currentTarget.contains(e.relatedTarget)) setDrag(false) }} onDrop={e => {e.preventDefault();setDrag(false);select(e.dataTransfer.files[0])}}>
        <div className="upload-symbol"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/></svg></div>
        <h2>Drop your audio here</h2><p>or choose a file from your device</p><button className="secondary" onClick={() => input.current.click()}>Browse files</button>
      </div>}
      {error && <p className="error" role="alert">{error}</p>}
      <div className="upload-bottom"><button className="primary" disabled={!file || busy} onClick={process}>{busy ? c.working : c.action}<span aria-hidden="true">↗</span></button><p>{mode === 'music' ? 'Best with a stereo music recording.' : 'MP3, WAV, or FLAC speech recording.'}</p></div>
    </div>
  </section>
}
export function ResultsStudio({ mode, result, onStartOver }) {
  const tracks = mode === 'music' ? [['Original',result.original],['Vocals',result.vocals],['Instrumental',result.instrumental]] : [['Original',result.original],['Enhanced Speech',result.enhanced]]
  const [active, setActive] = useState(0)
  const [view, setView] = useState('waveform')
  const [visuals, setVisuals] = useState({})
  const players = useRef([])
  const heading = useRef(null)
  const [title, source] = tracks[active]
  const image = source[view + '_url']
  useEffect(() => { heading.current?.focus() }, [])
  useEffect(() => {
    const urls = [...new Set(tracks.flatMap(([, track]) => [track.waveform_url, track.spectrogram_url]).filter(Boolean))]
    const preloaders = []
    let cancelled = false

    setVisuals(Object.fromEntries(urls.map(url => [url, 'loading'])))
    urls.forEach(url => {
      const preloader = new Image()
      preloader.onload = () => {
        if (!cancelled) setVisuals(current => ({ ...current, [url]: 'loaded' }))
      }
      preloader.onerror = () => {
        if (!cancelled) setVisuals(current => ({ ...current, [url]: 'error' }))
      }
      preloader.src = url
      preloaders.push(preloader)
    })

    return () => {
      cancelled = true
      preloaders.forEach(preloader => {
        preloader.onload = null
        preloader.onerror = null
      })
    }
  }, [mode, result])
  function play(index) {
    players.current.forEach((p,i) => { if (i !== index) p?.pause() })
  }
  return <section className="results-studio">
    <header className="results-heading"><div><span className="eyebrow ready-label">Processing complete</span><h1 ref={heading} tabIndex="-1">{mode === 'music' ? 'Your mix, unpacked.' : 'A clearer conversation.'}</h1><p className="filename">{result.original.filename}</p></div><button className="secondary" onClick={onStartOver}>New {mode === 'music' ? 'separation' : 'enhancement'} <span aria-hidden="true">＋</span></button></header>
    <div className="listening-workspace">
      <div className="track-list"><div className="surface-heading"><span>Your tracks</span><span className="muted">{tracks.length} tracks</span></div>
      {tracks.map(([name, track],i) => <article className={'track-row' + (i === active ? ' selected' : '')} key={name}>
        <div className="track-heading"><button className="track-select" aria-pressed={i === active} onClick={() => setActive(i)}><span className="track-number">0{i+1}</span><strong>{name}</strong></button>{track.download_url && <a className="download" href={track.download_url} download aria-label={'Download '+name}>↓ <span>Download</span></a>}</div>
        {track.audio_url ? <audio ref={el => {players.current[i]=el}} controls preload="metadata" src={track.audio_url} aria-label={name + ' playback'} onPlay={() => play(i)}/> : <p className="muted">Audio unavailable.</p>}
      </article>)}
      <p className="list-hint">Select a track to inspect it. Only one plays at a time.</p></div>
      <div className="inspector"><div className="surface-heading"><span>Sound explorer</span><span className="muted">02 / Results</span></div><div className="inspector-heading"><h2>{title}</h2><div className="view-switch" role="group" aria-label="Visualization type">{['waveform','spectrogram'].map(v => <button key={v} aria-pressed={view === v} onClick={() => setView(v)}>{v === 'waveform' ? 'Waveform' : 'Spectrogram'}</button>)}</div></div>
        <div className="plot" aria-live="polite" aria-busy={Boolean(image && visuals[image] !== 'loaded' && visuals[image] !== 'error')}>
          {image && visuals[image] === 'loaded' && <a href={image} target="_blank" rel="noreferrer" aria-label={'Open '+title+' '+view+' full size'}><img key={image} src={image} alt={title+' '+view} onError={() => setVisuals(current => ({ ...current, [image]: 'error' }))}/></a>}
          {image && visuals[image] !== 'loaded' && visuals[image] !== 'error' && <div className="visualization-skeleton"><span>Loading {view}...</span></div>}
          {(!image || visuals[image] === 'error') && <div className="visualization-error"><strong>Visualization unavailable</strong><span>The image could not be loaded.</span></div>}
        </div>
        <div className="plot-caption"><span>{view === 'waveform' ? 'Amplitude over time' : 'Frequency over time'}</span>{image && visuals[image] === 'loaded' && <a href={image} target="_blank" rel="noreferrer">Open full size ↗</a>}</div>
        <div className="inspector-note">Select a track on the left to explore its sound.</div>
      </div>
    </div>
  </section>
}
