import { useEffect, useRef, useState } from 'react'
import { ErrorMessage, ProcessingMessage } from '../components/StatusMessage.jsx'
import { separateAudio as separateAudioRequest } from '../services/api.js'

const ACCEPTED_EXTENSIONS = ['wav', 'mp3', 'flac']

function acceptsAudioFile(file) {
  const extension = file.name.split('.').pop()?.toLowerCase()
  return ACCEPTED_EXTENSIONS.includes(extension)
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB'
}

function MusicSeparationPage({ onSeparated }) {
  const inputRef = useRef(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl('')
      return undefined
    }

    const objectUrl = URL.createObjectURL(selectedFile)
    setPreviewUrl(objectUrl)
    return () => URL.revokeObjectURL(objectUrl)
  }, [selectedFile])

  function selectFile(file) {
    if (isProcessing) return
    setError('')

    if (!file) {
      setSelectedFile(null)
      return
    }

    if (!acceptsAudioFile(file)) {
      setSelectedFile(null)
      setError('Please select a WAV, MP3, or FLAC audio file.')
      return
    }

    setSelectedFile(file)
  }

  function handleDrop(event) {
    event.preventDefault()
    setIsDragging(false)
    selectFile(event.dataTransfer.files[0])
  }

  function removeFile() {
    if (isProcessing) return
    setSelectedFile(null)
    setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  async function separateAudio() {
    if (!selectedFile) {
      setError('Select a stereo audio file before starting separation.')
      return
    }

    setIsProcessing(true)
    setError('')

    try {
      const result = await separateAudioRequest(selectedFile)
      onSeparated(result)
    } catch (separationError) {
      if (separationError.name === 'TypeError') {
        setError('The Flask server is unavailable. Start the backend and try again.')
      } else {
        setError(separationError.message || 'Audio separation failed.')
      }
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <section className="uploader music-page" aria-labelledby="separation-heading" aria-busy={isProcessing}>
      <div className="section-heading">
        <span className="page-kicker">Music workspace</span>
        <h2 id="separation-heading">Music Separation</h2>
        <p>Pull a track apart into clean vocals and instrumental audio.</p>
      </div>

      <div
        className={'drop-zone' + (isDragging ? ' is-dragging' : '')}
        onDragEnter={(event) => {
          event.preventDefault()
          setIsDragging(true)
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <span className="upload-mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14.5v3A2.5 2.5 0 0 0 7.5 20h9a2.5 2.5 0 0 0 2.5-2.5v-3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
        <div className="drop-zone-copy">
          <strong>Drop your audio file here</strong>
          <span>or browse from your device</span>
          <small>MP3 · WAV · FLAC · up to 200 MB</small>
        </div>
        <button
          type="button"
          className="secondary-button"
          onClick={() => inputRef.current?.click()}
          disabled={isProcessing}
        >
          Browse files
        </button>
        <input
          ref={inputRef}
          className="visually-hidden"
          type="file"
          accept=".wav,.mp3,.flac,audio/wav,audio/mpeg,audio/flac"
          onChange={(event) => {
            selectFile(event.target.files[0])
            event.target.value = ''
          }}
        />
      </div>

      {error && <ErrorMessage onDismiss={() => setError('')}>{error}</ErrorMessage>}

      {selectedFile && (
        <div className="selection">
          <div className="selected-file-heading">
            <div className="selected-file-name">
              <span className="selection-label">Ready to separate</span>
              <strong>{selectedFile.name}</strong>
            </div>
            <div className="selected-file-actions">
              <span className="file-size">{formatFileSize(selectedFile.size)}</span>
              <button type="button" className="text-button" onClick={() => inputRef.current?.click()} disabled={isProcessing}>
                Change
              </button>
              <button type="button" className="text-button is-danger" onClick={removeFile} disabled={isProcessing}>
                Remove
              </button>
            </div>
          </div>
          {previewUrl && (
            <audio controls preload="metadata" src={previewUrl}>
              Your browser does not support audio playback.
            </audio>
          )}
          <button type="button" className="primary-button" onClick={separateAudio} disabled={isProcessing}>
            {isProcessing ? 'Separating audio...' : 'Separate audio'}
          </button>
        </div>
      )}

      {isProcessing && (
        <ProcessingMessage>Separating audio...</ProcessingMessage>
      )}
    </section>
  )
}

export default MusicSeparationPage
