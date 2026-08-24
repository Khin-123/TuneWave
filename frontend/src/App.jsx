import { useEffect, useState } from 'react'
import MusicResultsPage from './pages/MusicResultsPage.jsx'
import MusicSeparationPage from './pages/MusicSeparationPage.jsx'

function App() {
  const [backendStatus, setBackendStatus] = useState('checking')
  const [separationResult, setSeparationResult] = useState(null)

  useEffect(() => {
    const controller = new AbortController()

    async function checkBackend() {
      try {
        const response = await fetch('/api/health', { signal: controller.signal })

        if (!response.ok) {
          throw new Error(`Health check failed with status ${response.status}`)
        }

        const data = await response.json()
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

  return (
    <main className="app">
      <header className="app-header">
        <h1>TuneWave</h1>
        <p>DSP-Based Audio Processing</p>
        <small>Backend: {backendStatus}</small>
      </header>
      {separationResult ? (
        <MusicResultsPage
          result={separationResult}
          onStartOver={() => setSeparationResult(null)}
        />
      ) : (
        <MusicSeparationPage onSeparated={setSeparationResult} />
      )}
    </main>
  )
}

export default App
