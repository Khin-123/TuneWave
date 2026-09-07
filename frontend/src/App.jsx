import { useEffect, useState } from 'react'
import { Mark, UploadStudio, ResultsStudio } from './components/Studio.jsx'
import { checkBackend } from './services/api.js'
const paths = { music: '/music', speech: '/speech-enhancement' }
function valid(result, mode) {
  return (mode === 'music' ? ['original','vocals','instrumental'] : ['original','enhanced']).every(k => typeof result?.[k]?.audio_url === 'string' && result[k].audio_url.length > 0)
}
function readRoute() {
  const path = window.location.pathname.replace(/\/+$/, '')
  const mode = path.startsWith(paths.speech) ? 'speech' : 'music'
  const result = window.history.state?.result || window.history.state?.separationResult
  const isResults = path === paths[mode] + '/results' && valid(result, mode)
  return { mode, result: isResults ? result : null }
}
export default function App() {
  const [route, setRoute] = useState(readRoute)
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('Connecting')
  useEffect(() => {
    const controller = new AbortController()
    checkBackend(controller.signal).then(r => setStatus(r.status === 'ok' ? 'Ready' : 'Unavailable')).catch(e => {if(e.name !== 'AbortError') setStatus('Unavailable')})
    function sync() {
      const next = readRoute()
      window.history.replaceState(next.result ? { result: next.result } : {}, '', paths[next.mode] + (next.result ? '/results' : ''))
      setRoute(next)
    }
    sync()
    window.addEventListener('popstate', sync)
    return () => { controller.abort(); window.removeEventListener('popstate',sync) }
  }, [])
  function navigate(mode, result = null) {
    window.history.pushState(result ? {result} : {}, '', paths[mode] + (result ? '/results' : ''))
    setRoute({mode,result})
    window.scrollTo(0,0)
  }
  return <div className="app">
    <a className="skip-link" href="#workspace">Skip to workspace</a>
    <header className="topbar"><a className="brand" href="/music" onClick={e => {e.preventDefault();if(!busy) navigate('music')}} aria-label="TuneWave home"><Mark/><span>TuneWave</span></a>
      <nav aria-label="Audio tools">{Object.entries(paths).map(([mode,path]) => <a href={path} key={mode} aria-current={route.mode === mode ? 'page' : undefined} aria-disabled={busy} onClick={e => {if(e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return; e.preventDefault();if(!busy) navigate(mode)}}>{mode === 'music' ? 'Music Separation' : 'Speech Enhancement'}</a>)}</nav>
      <span className={'service-status '+status.toLowerCase()}><i/>{status}</span>
    </header>
    <main id="workspace" className="workspace">{route.result ? <ResultsStudio key={route.mode + window.location.pathname} mode={route.mode} result={route.result} onStartOver={() => navigate(route.mode)}/> : <UploadStudio key={route.mode} mode={route.mode} onBusy={setBusy} onComplete={result => navigate(route.mode,result)}/>}</main>
    <footer className="app-footer"><span>Made for a closer listen.</span><span>TuneWave / Audio workspace</span></footer>
  </div>
}
