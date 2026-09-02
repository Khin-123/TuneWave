import { useEffect, useState } from 'react'
import MusicResultsPage from './pages/MusicResultsPage.jsx'
import MusicSeparationPage from './pages/MusicSeparationPage.jsx'
import SpeechEnhancementPage from './pages/SpeechEnhancementPage.jsx'
import SpeechEnhancementResultsPage from './pages/SpeechEnhancementResultsPage.jsx'
import { checkBackend as checkBackendRequest } from './services/api.js'

const ROUTES = {
  music: '/music',
  musicResults: '/music/results',
  speech: '/speech-enhancement',
}

function routeFromPath() {
  const path = window.location.pathname.replace(/\/+$/, '') || '/'

  if (path === ROUTES.musicResults) return 'music-results'
  if (path === ROUTES.speech) return 'speech'
  return 'music'
}

function isValidSeparationResult(result) {
  return Boolean(result?.original && result?.vocals && result?.instrumental)
}

const BACKEND_LABELS = {
  checking: 'Checking backend',
  connected: 'Backend connected',
  unavailable: 'Backend unavailable',
}

function App() {
  const [backendStatus, setBackendStatus] = useState('checking')
  const [currentRoute, setCurrentRoute] = useState(routeFromPath)
  const [separationResult, setSeparationResult] = useState(() => {
    const historyResult = window.history.state?.separationResult
    return isValidSeparationResult(historyResult) ? historyResult : null
  })
  const [speechResult, setSpeechResult] = useState(null)
  const activeModule = currentRoute === 'speech' ? 'speech' : 'music'

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
    const route = routeFromPath()

    if (route === 'music-results' && !isValidSeparationResult(window.history.state?.separationResult)) {
      window.history.replaceState({}, '', ROUTES.music)
      setCurrentRoute('music')
    } else if (window.location.pathname === '/' || !Object.values(ROUTES).includes(window.location.pathname)) {
      window.history.replaceState({}, '', ROUTES.music)
      setCurrentRoute('music')
    }

    function handleNavigation(event) {
      const nextRoute = routeFromPath()

      if (nextRoute === 'music-results') {
        const result = event.state?.separationResult
        if (!isValidSeparationResult(result)) {
          window.history.replaceState({}, '', ROUTES.music)
          setSeparationResult(null)
          setCurrentRoute('music')
          return
        }
        setSeparationResult(result)
      }

      setCurrentRoute(nextRoute)
    }

    window.addEventListener('popstate', handleNavigation)
    return () => window.removeEventListener('popstate', handleNavigation)
  }, [])

  function navigateTo(module) {
    const path = module === 'speech' ? ROUTES.speech : ROUTES.music
    if (window.location.pathname !== path) {
      window.history.pushState({}, '', path)
    }
    setCurrentRoute(module)
    window.scrollTo(0, 0)
  }

  function showSeparationResults(result) {
    setSeparationResult(result)
    window.history.pushState({ separationResult: result }, '', ROUTES.musicResults)
    setCurrentRoute('music-results')
    window.scrollTo(0, 0)
  }

  function startAnotherSeparation() {
    setSeparationResult(null)
    window.history.pushState({}, '', ROUTES.music)
    setCurrentRoute('music')
    window.scrollTo(0, 0)
  }

  let currentPage
  if (currentRoute === 'speech') {
    currentPage = speechResult ? (
      <SpeechEnhancementResultsPage
        result={speechResult}
        onStartOver={() => setSpeechResult(null)}
      />
    ) : (
      <SpeechEnhancementPage onEnhanced={setSpeechResult} />
    )
  } else if (currentRoute === 'music-results' && isValidSeparationResult(separationResult)) {
    currentPage = (
      <MusicResultsPage
        result={separationResult}
        onStartOver={startAnotherSeparation}
      />
    )
  } else {
    currentPage = <MusicSeparationPage onSeparated={showSeparationResults} />
  }

  return (
    <main className="app">
      <header className="app-header">
        <div className="header-summary">
          <div className="brand-lockup">
            <svg className="brand-logo" viewBox="0 0 164 36" role="img" aria-label="TuneWave">
              <defs>
                <linearGradient id="tunewave-logo-accent" x1="4" y1="4" x2="31" y2="32" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#a59cff" />
                  <stop offset="1" stopColor="#6756e8" />
                </linearGradient>
              </defs>
              <rect x="1" y="2" width="32" height="32" rx="10" fill="url(#tunewave-logo-accent)" />
              <path d="M7 19h3l2.2-6 3.1 12 3.2-15 2.8 11 2.2-6 2 4H29" fill="none" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
              <text x="42" y="24.5" fill="currentColor" fontFamily="Inter, ui-sans-serif, system-ui, sans-serif" fontSize="18" fontWeight="760" letterSpacing="-0.7">TuneWave</text>
            </svg>
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
      <div className="app-content">{currentPage}</div>
    </main>
  )
}

export default App
