import { useEffect, useState } from 'react'
import MusicResultsPage from './pages/MusicResultsPage.jsx'
import MusicSeparationPage from './pages/MusicSeparationPage.jsx'
import SpeechEnhancementPage from './pages/SpeechEnhancementPage.jsx'
import SpeechEnhancementResultsPage from './pages/SpeechEnhancementResultsPage.jsx'
import { checkBackend as checkBackendRequest } from './services/api.js'

function moduleFromPath() {
  return window.location.pathname === '/speech-enhancement' ? 'speech' : 'music'
}

const BACKEND_LABELS = {
  checking: 'Checking backend',
  connected: 'Backend connected',
  unavailable: 'Backend unavailable',
}

function App() {
  const [backendStatus, setBackendStatus] = useState('checking')
  const [activeModule, setActiveModule] = useState(moduleFromPath)
  const [separationResult, setSeparationResult] = useState(null)
  const [speechResult, setSpeechResult] = useState(null)

  useEffect(() => {
    const controller = new AbortController()

    async function checkBackend() {
      try {
        const data = await checkBackendRequest(controller.signal)
        setBackendStatus(data.status === 'ok' ? 'connected' : 'unavailable')
      } catch (error) {
        if (error.name !== 'AbortError') {
          setBackendStatus('unavailable')
        }
      }
    }

    checkBackend()

    return () => controller.abort()
  }, [])

  useEffect(() => {
    function handleNavigation() {
      setActiveModule(moduleFromPath())
    }

    window.addEventListener('popstate', handleNavigation)
    return () => window.removeEventListener('popstate', handleNavigation)
  }, [])

  function navigateTo(module) {
    const path = module === 'speech' ? '/speech-enhancement' : '/'
    if (window.location.pathname !== path) {
      window.history.pushState({}, '', path)
    }
    setActiveModule(module)
  }

  let currentPage
  if (activeModule === 'speech') {
    currentPage = speechResult ? (
      <SpeechEnhancementResultsPage
        result={speechResult}
        onStartOver={() => setSpeechResult(null)}
      />
    ) : (
      <SpeechEnhancementPage onEnhanced={setSpeechResult} />
    )
  } else {
    currentPage = separationResult ? (
      <MusicResultsPage
        result={separationResult}
        onStartOver={() => setSeparationResult(null)}
      />
    ) : (
      <MusicSeparationPage onSeparated={setSeparationResult} />
    )
  }

  return (
    <main className="app">
      <header className="app-header">
        <div className="header-summary">
          <div className="brand-lockup">
            <h1>TuneWave</h1>
          </div>
          <nav className="module-navigation" aria-label="TuneWave modules">
            <button
              type="button"
              className={activeModule === 'music' ? 'is-active' : ''}
              aria-current={activeModule === 'music' ? 'page' : undefined}
              onClick={() => navigateTo('music')}
            >
              Music Separation
            </button>
            <button
              type="button"
              className={activeModule === 'speech' ? 'is-active' : ''}
              aria-current={activeModule === 'speech' ? 'page' : undefined}
              onClick={() => navigateTo('speech')}
            >
              Speech Enhancement
            </button>
          </nav>
          <span className={`backend-status is-${backendStatus}`}>
            <span className="status-dot" aria-hidden="true" />
            {BACKEND_LABELS[backendStatus]}
          </span>
        </div>
      </header>
      {currentPage}
    </main>
  )
}

export default App
