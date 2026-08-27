const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL || ''
const API_BASE_URL = configuredBaseUrl.replace(/\/$/, '')

function apiUrl(path) {
  return `${API_BASE_URL}${path}`
}

export function resolveApiUrl(url) {
  if (!url) return ''
  if (/^https?:\/\//i.test(url)) return url
  return `${API_BASE_URL}${url.startsWith('/') ? url : `/${url}`}`
}

async function parseResponse(response) {
  const payload = await response.json().catch(() => ({}))
  if (!response.ok || payload.success === false) {
    const error = new Error(payload.error || `Request failed with status ${response.status}.`)
    error.status = response.status
    throw error
  }
  return payload
}

export async function checkBackend(signal) {
  const response = await fetch(apiUrl('/api/health'), { signal })
  return parseResponse(response)
}

export async function separateAudio(file, signal) {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(apiUrl('/api/separate'), { method: 'POST', body: formData, signal })
  const result = await parseResponse(response)
  return {
    ...result,
    original: resolveSourceUrls(result.original),
    vocals: resolveSourceUrls(result.vocals),
    instrumental: resolveSourceUrls(result.instrumental),
  }
}

function resolveSourceUrls(source = {}) {
  return {
    ...source,
    audio_url: resolveApiUrl(source.audio_url),
    download_url: resolveApiUrl(source.download_url),
    waveform_url: resolveApiUrl(source.waveform_url),
    spectrogram_url: resolveApiUrl(source.spectrogram_url),
  }
}
