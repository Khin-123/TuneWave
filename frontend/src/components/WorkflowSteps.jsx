function WorkflowSteps({ currentStep }) {
  const steps = ['Upload', 'Processing', 'Results']

  return (
    <ol className="workflow-steps" aria-label="Audio workflow progress">
      {steps.map((label, index) => {
        const step = index + 1
        const isComplete = step < currentStep
        const isCurrent = step === currentStep

        return (
          <li key={label} className={`${isComplete ? 'is-complete ' : ''}${isCurrent ? 'is-current' : ''}`}>
            <span className="step-index" aria-hidden="true">{isComplete ? '✓' : `0${step}`}</span>
            <span>{label}</span>
          </li>
        )
      })}
    </ol>
  )
}

export default WorkflowSteps
