import { useEffect, useRef, useState } from 'react'
import { ErrorMessage, ProcessingMessage } from '../components/StatusMessage.jsx'
import { enhanceSpeech as enhanceSpeechRequest } from '../services/api.js'


const ACCEPTED_EXTENSIONS = ['wav']


function acceptsSpeechFile(file) {
  const extension = file.name.split('.').pop()?.toLowerCase()
  return ACCEPTED_EXTENSIONS.includes(extension)
}


function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
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

    if (!acceptsSpeechFile(file)) {
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
        <h2 id="speech-heading">Speech Enhancement</h2>
        <p>Reduce background noise and improve speech clarity.</p>
      </div>

      <div
        className={`drop-zone${isDragging ? ' is-dragging' : ''}`}
        onDragEnter={(event) => {
          event.preventDefault()
          setIsDragging(true)
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <span className="upload-mark" aria-hidden="true">↑</span>
        <div className="drop-zone-copy">
          <strong>Drop a WAV file here</strong>
          <span>or choose a file</span>
          <small>WAV</small>
        </div>
        <button
          type="button"
          className="secondary-button"
          onClick={() => inputRef.current?.click()}
          disabled={isProcessing}
        >
          Choose File
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
            <div>
              <span className="selection-label">Selected audio</span>
              <strong>{selectedFile.name}</strong>
            </div>
            <div className="selected-file-actions">
              <span className="file-size">{formatFileSize(selectedFile.size)}</span>
              <button
                type="button"
                className="text-button"
                onClick={() => inputRef.current?.click()}
                disabled={isProcessing}
              >
                Change File
              </button>
              <button
                type="button"
                className="text-button is-danger"
                onClick={removeFile}
                disabled={isProcessing}
              >
                Remove
              </button>
            </div>
          </div>
          {previewUrl && (
            <audio controls preload="metadata" src={previewUrl}>
              Your browser does not support audio playback.
            </audio>
          )}
          <button
            type="button"
            className="primary-button"
            onClick={enhanceSpeech}
            disabled={isProcessing}
          >
            {isProcessing ? 'Enhancing speech...' : 'Enhance Speech'}
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
