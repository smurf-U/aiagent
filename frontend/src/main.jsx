// App bootstrap: mounts <App/> into #root. StrictMode double-invokes effects
// in development to surface side-effect bugs; it has no effect in production.
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
