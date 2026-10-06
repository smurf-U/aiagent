// Chat message rendering: the message bubble, the agent's activity steps
// (plan, tool calls, files), the live "thinking" bubble, and the
// human-in-the-loop approval card.
import { useState } from 'react'
import ReactMarkdown from 'react-markdown'

// ── Step event card ──────────────────────────────────────────────────────────
/** Icon, label and colour per step type (event types from the backend). */
const STEP_META = {
  plan:        { icon: '📋', label: 'Planning', color: '#a78bfa' },
  subagent:    { icon: '🤖', label: 'Subagent Delegation', color: '#22d3ee' },
  tool_started:   { icon: '🔧', label: 'Tool Call', color: '#f59e0b' },
  tool_completed: { icon: '↩️', label: 'Tool Result', color: '#94a3b8' },
  file_updated:   { icon: '📁', label: 'File Updated', color: '#10b981' },
  status:      { icon: '⚙️', label: 'Status', color: '#475569' },
}

/**
 * One collapsible activity step. Plans start expanded; status steps render as
 * a single line.
 * @param {{ step: { type: string, data: object } }} props
 */
function StepCard({ step }) {
  const [open, setOpen] = useState(step.type === 'plan')
  const meta = STEP_META[step.type] || { icon: '🛠️', label: step.type, color: '#6366f1' }
  const d = step.data

  /** One-line summary shown in the collapsed header. */
  const getTitle = () => {
    if (step.type === 'plan') {
      const t = d.todos || []
      const done = t.filter(x => x.status === 'completed').length
      return `Execution Plan (${done}/${t.length} done)`
    }
    if (step.type === 'subagent') return `Delegating → ${d.name}`
    if (step.type === 'tool_started') return `${d.name}${d.args?.query ? `: "${d.args.query}"` : ''}`
    if (step.type === 'tool_completed') return `${d.name} result`
    if (step.type === 'file_updated') return `file: ${d.path}`
    if (step.type === 'status') return d.status
    return meta.label
  }

  if (step.type === 'status') {
    return (
      <div className="status-bar">
        <span>⚙️</span>
        <span>{d.status}</span>
      </div>
    )
  }

  return (
    <div className="step-card">
      <div className="step-header" onClick={() => setOpen(o => !o)}>
        <span className="step-icon">{meta.icon}</span>
        <span className="step-title" style={{ color: meta.color }}>{getTitle()}</span>
        <span className={`step-chevron ${open ? 'open' : ''}`}>▶</span>
      </div>
      {open && (
        <div className="step-body">
          {step.type === 'plan' && (
            (d.todos || []).map((t, i) => (
              <div className="todo-item" key={i}>
                <span className={`todo-badge ${t.status || 'pending'}`}>
                  {t.status === 'completed' ? '✓' : t.status === 'in_progress' ? '…' : '○'}
                  &nbsp;{(t.status || 'pending').replace('_', ' ')}
                </span>
                <span>{t.content || String(t)}</span>
              </div>
            ))
          )}
          {step.type === 'subagent' && (
            <div style={{ color: 'var(--text-secondary)' }}>
              <strong>Task:</strong> {d.description}
            </div>
          )}
          {(step.type === 'tool_started' || step.type === 'tool_completed') && (
            <pre className="code-view" style={{ fontSize: 11 }}>
              {JSON.stringify(d.args || d.result, null, 2)}
            </pre>
          )}
          {step.type === 'file_updated' && (
            <pre className="code-view" style={{ fontSize: 11 }}>
              {(d.content || '').slice(0, 800)}{d.content?.length > 800 ? '\n…(truncated)' : ''}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}

// ── Thinking animation ────────────────────────────────────────────────────────
/**
 * The in-progress assistant bubble: live steps, the current status line, and
 * the answer streamed so far (typing dots until the first token).
 * @param {{ status: string, steps?: object[], answer?: string }} props
 */
export function TypingBubble({ status, steps = [], answer = '' }) {
  return (
    <div className="msg-row">
      <div className="avatar assistant">🧠</div>
      <div className="msg-content">
      {steps.length > 0 && (
        <div className="steps-wrapper" style={{ maxWidth: 600, marginBottom: 6 }}>
          {steps.map((s, i) => <StepCard key={i} step={s} />)}
        </div>
      )}
      <div className="msg-bubble assistant">
        <div className="status-bar" style={{ border: 'none', background: 'none', padding: '0', marginBottom: 4 }}>
          <div className="spinner-sm" />
          <span>{status || 'Thinking…'}</span>
        </div>
        {answer
          ? <ReactMarkdown>{answer}</ReactMarkdown>
          : <div className="typing-dots"><span /><span /><span /></div>}
      </div>
      </div>
    </div>
  )
}

// ── Human-in-the-loop approval card ──────────────────────────────────────────
/** How an answered decision is shown once the card collapses. */
const DECISION_LABEL = { approve: '✓ Approved', edit: '✎ Edited', reject: '✕ Rejected' }

/**
 * Review controls for one pending tool call: Approve, Edit (change the JSON
 * arguments) or Reject (with a reason the agent sees).
 * @param {{ action: { name: string, args: object }, allowed: string[],
 *           disabled: boolean, onDecision: (decision: object) => void }} props
 */
function ActionReview({ action, allowed, disabled, onDecision }) {
  const [mode, setMode] = useState(null) // null | 'edit' | 'reject'
  const [argsText, setArgsText] = useState(JSON.stringify(action.args, null, 2))
  const [reason, setReason] = useState('')
  const [error, setError] = useState('')

  const submitEdit = () => {
    try {
      const args = JSON.parse(argsText)
      onDecision({ type: 'edit', edited_action: { name: action.name, args } })
    } catch {
      setError('Arguments must be valid JSON.')
    }
  }

  return (
    <div className="approval-action">
      <div className="approval-action-name">🔧 {action.name}</div>
      {mode === 'edit' ? (
        <>
          <textarea
            className="approval-textarea"
            value={argsText}
            onChange={e => { setArgsText(e.target.value); setError('') }}
            rows={Math.min(14, argsText.split('\n').length + 1)}
          />
          {error && <div className="approval-error">{error}</div>}
        </>
      ) : (
        <pre className="code-view" style={{ fontSize: 11 }}>{JSON.stringify(action.args, null, 2)}</pre>
      )}
      {mode === 'reject' && (
        <input
          className="approval-input"
          placeholder="Why? (sent to the agent so it can adjust)"
          value={reason}
          onChange={e => setReason(e.target.value)}
          autoFocus
        />
      )}
      <div className="approval-buttons">
        {mode === null && (
          <>
            {allowed.includes('approve') && (
              <button className="approval-btn approve" disabled={disabled} onClick={() => onDecision({ type: 'approve' })}>Approve</button>
            )}
            {allowed.includes('edit') && (
              <button className="approval-btn" disabled={disabled} onClick={() => setMode('edit')}>Edit</button>
            )}
            {allowed.includes('reject') && (
              <button className="approval-btn reject" disabled={disabled} onClick={() => setMode('reject')}>Reject</button>
            )}
          </>
        )}
        {mode === 'edit' && (
          <>
            <button className="approval-btn approve" disabled={disabled} onClick={submitEdit}>Run with edits</button>
            <button className="approval-btn" onClick={() => setMode(null)}>Cancel</button>
          </>
        )}
        {mode === 'reject' && (
          <>
            <button
              className="approval-btn reject"
              disabled={disabled}
              onClick={() => onDecision({ type: 'reject', message: reason.trim() || 'Rejected by user.' })}
            >Confirm reject</button>
            <button className="approval-btn" onClick={() => setMode(null)}>Cancel</button>
          </>
        )}
      </div>
    </div>
  )
}

/**
 * Approval request for a paused turn. Collects one decision per pending tool
 * call and submits them together; after that (or when loaded from history)
 * it shows a one-line summary per call.
 * @param {{ interrupt: { action_requests: object[], review_configs?: object[],
 *           resolved?: string[] }, canDecide: boolean,
 *           onDecide: (decisions: object[]) => void }} props
 */
function ApprovalCard({ interrupt, canDecide, onDecide }) {
  const actions = interrupt.action_requests || []
  const [decisions, setDecisions] = useState({})

  if (interrupt.resolved) {
    return (
      <div className="approval-card resolved">
        {actions.map((a, i) => (
          <div key={i}>🔧 {a.name} — {DECISION_LABEL[interrupt.resolved[i]] || interrupt.resolved[i]}</div>
        ))}
      </div>
    )
  }

  /** Decisions the backend allows for a tool (defaults to all three). */
  const allowedFor = (name) =>
    interrupt.review_configs?.find(c => c.action_name === name)?.allowed_decisions
    || ['approve', 'edit', 'reject']

  /** Record the decision for action `i`; submit once every action has one. */
  const decide = (i, decision) => {
    const next = { ...decisions, [i]: decision }
    setDecisions(next)
    // Resume once every pending action has a decision.
    if (Object.keys(next).length === actions.length) {
      onDecide(actions.map((_, j) => next[j]))
    }
  }

  return (
    <div className="approval-card">
      <div className="approval-title">⏸ Approval needed</div>
      {actions.map((a, i) => (
        decisions[i]
          ? <div key={i} className="approval-action">🔧 {a.name} — {DECISION_LABEL[decisions[i].type]}</div>
          : <ActionReview key={i} action={a} allowed={allowedFor(a.name)} disabled={!canDecide} onDecision={d => decide(i, d)} />
      ))}
      {!canDecide && <div className="approval-error">This request is no longer active.</div>}
    </div>
  )
}

// ── Message bubble ────────────────────────────────────────────────────────────
/**
 * A finished chat message. Assistant messages also show their steps, a Retry
 * button (failed/stopped task, last message only), the approval card, and
 * the files the turn wrote.
 * @param {{ msg: object, canDecide?: boolean, onDecide?: Function,
 *           canRetry?: boolean, onRetry?: Function }} props
 */
export function ChatMessage({ msg, canDecide = false, onDecide, canRetry = false, onRetry }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`msg-row ${isUser ? 'user' : ''}`}>
      <div className={`avatar ${isUser ? 'user' : 'assistant'}`}>
        {isUser ? '👤' : '🧠'}
      </div>
      <div className="msg-content">
        {/* Steps first */}
        {!isUser && msg.steps?.length > 0 && (
          <div className="steps-wrapper" style={{ maxWidth: 600, marginBottom: 6 }}>
            {msg.steps.map((s, i) => <StepCard key={i} step={s} />)}
          </div>
        )}
        {/* Main content */}
        <div className={`msg-bubble ${isUser ? 'user' : 'assistant'}`}>
          {isUser
            ? <span>{msg.content}</span>
            : <ReactMarkdown>{msg.content || ''}</ReactMarkdown>
          }
        </div>
        {/* Failed or stopped task */}
        {canRetry && (
          <div className="retry-row">
            <span>{msg.status === 'cancelled' ? 'Task stopped.' : 'Task failed.'}</span>
            <button className="approval-btn" onClick={onRetry}>↻ Retry</button>
          </div>
        )}
        {/* Human-in-the-loop approval */}
        {!isUser && msg.interrupt && (
          <ApprovalCard interrupt={msg.interrupt} canDecide={canDecide} onDecide={onDecide} />
        )}
        {/* Virtual files if any */}
        {!isUser && msg.files && Object.keys(msg.files).length > 0 && (
          <div style={{ marginTop: 6, maxWidth: 600 }}>
            {Object.entries(msg.files).map(([path, content]) => (
              <div key={path} className="step-card" style={{ marginBottom: 4 }}>
                <div className="step-header" style={{ cursor: 'default' }}>
                  <span>📄</span>
                  <span className="step-title" style={{ color: 'var(--green)' }}>{path}</span>
                </div>
                <div className="step-body">
                  <pre className="code-view">{String(content).slice(0, 800)}</pre>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
