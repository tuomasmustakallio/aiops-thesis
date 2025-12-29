import { useState, useEffect } from 'react'

interface HealthStatus {
  status: string
  service: string
}

function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/health')
      .then((res) => res.json())
      .then((data: HealthStatus) => setHealth(data))
      .catch((err: Error) => setError(err.message))
  }, [])

  return (
    <div style={{ fontFamily: 'system-ui', padding: '2rem', maxWidth: '600px', margin: '0 auto' }}>
      <h1>CI/CD Failure Prediction Experiment</h1>
      <p>Master's Thesis Technical Artifact</p>

      <h2>Backend Health</h2>
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}
      {health && (
        <div style={{ background: '#f0f0f0', padding: '1rem', borderRadius: '4px' }}>
          <p><strong>Status:</strong> {health.status}</p>
          <p><strong>Service:</strong> {health.service}</p>
        </div>
      )}
      {!health && !error && <p>Loading...</p>}
    </div>
  )
}

export default App
