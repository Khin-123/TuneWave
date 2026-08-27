export function ErrorMessage({ children, onDismiss }) {
  return (
    <div className="message error-message" role="alert">
      <span className="message-icon" aria-hidden="true">!</span>
      <div className="message-copy">
        <strong>Something went wrong</strong>
        <p>{children}</p>
      </div>
      {onDismiss && (
        <button type="button" className="message-dismiss" onClick={onDismiss}>
          Dismiss
        </button>
      )}
    </div>
  )
}


export function ProcessingMessage({ children }) {
  return (
    <div className="processing-message" role="status">
      <span className="loading-indicator" aria-hidden="true" />
      <div>
        <strong>{children}</strong>
        <p>This may take a moment.</p>
      </div>
    </div>
  )
}
