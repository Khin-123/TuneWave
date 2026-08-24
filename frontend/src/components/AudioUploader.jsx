import { useEffect, useRef, useState } from 'react'

const ACCEPTED_EXTENSIONS = ['wav', 'mp3', 'flac']

function isAcceptedAudio(file) {
  const extension = file.name.split('.').pop()?.toLowerCase()
  return ACCEPTED_EXTENSIONS.includes(extension)
}

function formatDuration(duration) {
  const roundedDuration = Math.round(duration * 100) / 100
  const minutes = Math.floor(roundedDuration / 60)
  const seconds = roundedDuration - minutes * 60
  return `${minutes}:${seconds.toFixed(2).padStart(5, '0')}`
}

function AudioUploader() {
  const inputRef = useRef(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [audioInfo, setAudioInfo] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
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
    setAudioInfo(null)
    setError('')

    if (!file) {
      setSelectedFile(null)
      return
    }

    if (!isAcceptedAudio(file)) {
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

  async function handleUpload() {
    if (!selectedFile) {
      setError('Select an audio file before uploading.')
      return
    }

    const formData = new FormData()
    formData.append('file', selectedFile)
    setIsUploading(true)
    setAudioInfo(null)
    setError('')

    try {
      const response = await fetch('/api/audio-info', {
        method: 'POST',
        body: formData,
      })
      const result = await response.json()

      if (!response.ok || !result.success) {
        throw new Error(result.error || 'The audio upload failed.')
      }

      setAudioInfo(result)
    } catch (uploadError) {
      setError(uploadError.message || 'Unable to connect to the backend.')
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <section className="uploader" aria-labelledby="upload-heading">
      <div className="section-heading">
        <h2 id="upload-heading">Upload audio</h2>
        <p>Select a WAV, MP3, or FLAC file to inspect its basic information.</p>
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
        <strong>Drag and drop an audio file here</strong>
        <span>or</span>
        <button type="button" className="secondary-button" onClick={() => inputRef.current?.click()}>
          Browse file
        </button>
        <input
          ref={inputRef}
          className="visually-hidden"
          type="file"
          accept=".wav,.mp3,.flac,audio/wav,audio/mpeg,audio/flac"
          onChange={(event) => selectFile(event.target.files[0])}
        />
      </div>

      {error && <p className="message error-message" role="alert">{error}</p>}

      {selectedFile && (
        <div className="selection">
          <p><strong>Selected:</strong> {selectedFile.name}</p>
          {previewUrl && <audio controls preload="metadata" src={previewUrl}>Your browser does not support audio playback.</audio>}
          <button type="button" className="primary-button" onClick={handleUpload} disabled={isUploading}>
            {isUploading ? 'Uploading…' : 'Upload and inspect'}
          </button>
        </div>
      )}

      {audioInfo && (
        <dl className="audio-info">
          <div><dt>File name</dt><dd>{audioInfo.filename}</dd></div>
          <div><dt>Duration</dt><dd>{formatDuration(audioInfo.duration)}</dd></div>
          <div><dt>Sample rate</dt><dd>{audioInfo.sample_rate.toLocaleString()} Hz</dd></div>
          <div><dt>Channels</dt><dd>{audioInfo.channels}</dd></div>
          <div><dt>Samples</dt><dd>{audioInfo.samples.toLocaleString()}</dd></div>
        </dl>
      )}
    </section>
  )
}

export default AudioUploader
