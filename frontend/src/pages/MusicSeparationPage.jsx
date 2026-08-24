import { useEffect, useRef, useState } from 'react'


const ACCEPTED_EXTENSIONS = ['wav', 'mp3', 'flac']


function acceptsAudioFile(file) {
  const extension = file.name.split('.').pop()?.toLowerCase()
  return ACCEPTED_EXTENSIONS.includes(extension)
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

  async function separateAudio() {
    if (!selectedFile) {
      setError('Select a stereo audio file before starting separation.')
      return
    }

    const formData = new FormData()
    formData.append('file', selectedFile)
    setIsProcessing(true)
    setError('')

    try {
      const response = await fetch('/api/separate', {
        method: 'POST',
        body: formData,
      })
      const result = await response.json().catch(() => ({}))

      if (!response.ok || !result.success) {
        throw new Error(result.error || 'Audio separation failed.')
      }

      onSeparated({
        ...result,
        originalFile: selectedFile,
      })
    } catch (separationError) {
      setError(separationError.message || 'Unable to connect to the backend.')
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <section className="uploader music-page" aria-labelledby="separation-heading" aria-busy={isProcessing}>
      <div className="section-heading">
        <h2 id="separation-heading">Music separation</h2>
        <p>Upload a stereo song to create vocal and instrumental stems.</p>
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
        <strong>Drag and drop a stereo song here</strong>
        <span>WAV, MP3, or FLAC</span>
        <button
          type="button"
          className="secondary-button"
          onClick={() => inputRef.current?.click()}
          disabled={isProcessing}
        >
          Browse file
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

      {error && <p className="message error-message" role="alert">{error}</p>}

      {selectedFile && (
        <div className="selection">
          <p><strong>Selected:</strong> {selectedFile.name}</p>
          {previewUrl && (
            <audio controls preload="metadata" src={previewUrl}>
              Your browser does not support audio playback.
            </audio>
          )}
          <button
            type="button"
            className="primary-button"
            onClick={separateAudio}
            disabled={isProcessing}
          >
            {isProcessing ? 'Separating audio...' : 'Separate Audio'}
          </button>
        </div>
      )}

      {isProcessing && (
        <p className="processing-message" role="status">
          Processing the full-resolution audio with mid-side spectral masking...
        </p>
      )}
    </section>
  )
}


export default MusicSeparationPage
