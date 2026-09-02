import { useEffect, useRef, useState } from 'react'
import { ErrorMessage, ProcessingMessage } from '../components/StatusMessage.jsx'
import WorkflowSteps from '../components/WorkflowSteps.jsx'
import { enhanceSpeech as enhanceSpeechRequest } from '../services/api.js'

const ACCEPTED_EXTENSIONS = ['wav']

function acceptsAudioFile(file) {
  const extension = file.name.split('.').pop()?.toLowerCase()
  return ACCEPTED_EXTENSIONS.includes(extension)
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB'
}

function SpeechEnhancementPage({ onEnhanced }) {
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
      setError('Please select a WAV speech recording.')
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

  async function enhanceSpeech() {
    if (!selectedFile) {
      setError('Select a noisy speech WAV before starting enhancement.')
      return
    }

    setIsProcessing(true)
    setError('')

    try {
      const result = await enhanceSpeechRequest(selectedFile)
      onEnhanced(result)
    } catch (enhancementError) {
      if (enhancementError.name === 'TypeError') {
        setError('The Flask server is unavailable. Start the backend and try again.')
      } else if (enhancementError.status === 400) {
        setError(enhancementError.message || 'The audio file is invalid or cannot be decoded.')
      } else {
        setError(enhancementError.message || 'Speech enhancement failed. Please try again.')
      }
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <section className="uploader speech-page" aria-labelledby="speech-heading" aria-busy={isProcessing}>
      <div className="section-heading">
        <span className="page-kicker">Speech workspace</span>
        <h2 id="speech-heading">Speech Enhancement</h2>
        <p>Reduce background noise and bring spoken words into focus.</p>
      </div>

      <WorkflowSteps currentStep={isProcessing ? 2 : 1} />

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
          <strong>Drop a speech recording here</strong>
          <span>or browse from your device</span>
          <small>WAV · up to 200 MB</small>
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
          accept=".wav,audio/wav,audio/x-wav"
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
              <span className="selection-label">Ready to enhance</span>
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
          <button type="button" className="primary-button" onClick={enhanceSpeech} disabled={isProcessing}>
            {isProcessing ? 'Enhancing speech...' : 'Enhance speech'}
          </button>
        </div>
      )}

      {isProcessing && (
        <ProcessingMessage>Enhancing speech...</ProcessingMessage>
      )}
    </section>
  )
}

export default SpeechEnhancementPage
