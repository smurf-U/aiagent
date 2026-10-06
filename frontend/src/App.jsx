// Main screen: thread list and agent settings (left), chat (centre), and
// skills / files / memory (right). A message becomes a server-side task; this
// component follows the task's events to show live progress, then reloads the
// thread from the server, which holds the final messages.
import { useEffect, useRef, useState } from 'react'
import { api } from './api'
import { useThreads, useSkills, useFiles, useMemory, useModels } from './hooks'
import { ChatMessage, TypingBubble } from './ChatMessage'
import './index.css'

/** Starter prompts shown in an empty chat, one per agent feature. */
const DEMO_PROMPTS = [
  { label: '📋 Planning', text: 'Plan a 4-step migration from Flask to FastAPI. Save it to /note/migration_plan.txt then read it back.' },
  { label: '🧠 Skills', text: 'Using your LangGraph skill, show a minimal StateGraph with conditional routing and MemorySaver. Add code.' },
  { label: '🤖 Subagent', text: 'Use your research subagent to find the top 3 improvements in LangChain deepagents. Cite sources.' },
  { label: '📊 Structured', text: 'Use your structured-researcher subagent: evaluate production readiness of SLMs. Return confidence score + sources.' },
]

/** Where the agent keeps its files (see backend/agent.py get_agent). */
const BACKEND_OPTIONS = [
  { value: 'StateBackend', label: 'StateBackend', desc: 'In-state, per thread' },
  { value: 'FilesystemBackend', label: 'FilesystemBackend', desc: 'Real disk files' },
  { value: 'StoreBackend', label: 'StoreBackend', desc: 'Cross-thread store' },
]

/** Human-readable file size ("512B", "3.4KB"); empty for 0/unknown. */
function formatSize(bytes) {
  if (!bytes) return ''
  if (bytes < 1024) return `${bytes}B`
  return `${(bytes / 1024).toFixed(1)}KB`
}

export default function App() {
  // ── Agent configuration state ──────────────────────────────────
  const models = useModels()
  const [model, setModel] = useState('ollama:qwen3:8b')
  const [backend, setBackend] = useState('StateBackend')
  const [useAgentsMd, setUseAgentsMd] = useState(true)
  const [useSkillsFlag, setUseSkillsFlag] = useState(true)
  const [useSubagents, setUseSubagents] = useState(true)
  const [requireApproval, setRequireApproval] = useState(false)

  // ── Thread management ──────────────────────────────────────────
  const { threads, createThread, refresh: refreshThreads } = useThreads()
  const [activeThread, setActiveThread] = useState(null)
  const [messages, setMessages] = useState([])

  // ── Chat state ─────────────────────────────────────────────────
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [streamStatus, setStreamStatus] = useState('')
  const [liveSteps, setLiveSteps] = useState([])
  const [liveAnswer, setLiveAnswer] = useState('')
  const chatEndRef = useRef(null)
  const textareaRef = useRef(null)
  const taskRef = useRef(null)          // { taskId, threadId, cancel, close } of the followed task
  const activeThreadRef = useRef(null)  // latest active thread, for async callbacks
  const [threadFiles, setThreadFiles] = useState([])  // agent's virtual files for the active thread

  // ── Right panel ────────────────────────────────────────────────
  const [rightTab, setRightTab] = useState('skills')
  const skills = useSkills()
  const { files, selected: selFile, content: fileContent, openFile, refresh: refreshFiles } = useFiles()
  const { items: memItems, deleteItem: deleteMemory, refresh: refreshMemory } = useMemory()

  // ── API key badges (from /api/health) ──────────────────────────
  const [keyStatus, setKeyStatus] = useState({})
  useEffect(() => {
    api.health().then(d => setKeyStatus(d)).catch(() => {})
  }, [])

  // Keep the newest message or live output in view.
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streaming, liveSteps, liveAnswer])

  // ── Input ──────────────────────────────────────────────────────
  /** Track the draft and grow the textarea with its content (max 160px). */
  const handleInput = (e) => {
    setInput(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
  }

  // ── Select thread ──────────────────────────────────────────────
  /** Reload a thread's messages and agent files, if it is still the active one. */
  const loadThread = async (id) => {
    const t = await api.threads.get(id).catch(() => ({ messages: [] }))
    if (activeThreadRef.current === id) setMessages(t.messages || [])
    api.threads.files(id).then(f => {
      if (activeThreadRef.current === id) setThreadFiles(Array.isArray(f) ? f : [])
    }).catch(() => {})
    return t
  }

  /** Make `id` the active thread and reset all live (in-progress) state. */
  const switchThread = (id) => {
    // Stop following the previous thread's task (it keeps running on the server).
    taskRef.current?.close()
    taskRef.current = null
    setStreaming(false)
    setStreamStatus('')
    setLiveSteps([])
    setLiveAnswer('')
    activeThreadRef.current = id
    setActiveThread(id)
  }

  /** Create and open an empty thread. */
  const handleNewThread = async () => {
    const t = await createThread()
    switchThread(t.id)
    setMessages([])
    setThreadFiles([])
  }

  // ── Tasks ──────────────────────────────────────────────────────
  /** The sidebar settings, in the shape /api/chat and /api/chat/resume expect. */
  const agentSettings = (tid) => ({
    thread_id: tid,
    model,
    backend,
    use_agents_md: useAgentsMd,
    use_skills: useSkillsFlag,
    use_subagents: useSubagents,
    require_approval: requireApproval,
  })

  /**
   * Show a task's live progress (status line, steps, streamed answer) until
   * it ends, then reload the thread for the stored final message.
   */
  const followTask = (taskId, threadId) => {
    taskRef.current?.close()
    setStreaming(true)
    setStreamStatus('Queued…')
    setLiveSteps([])
    setLiveAnswer('')

    const steps = []
    let answer = ''

    const finish = async (err) => {
      if (taskRef.current?.taskId !== taskId) return  // a newer task took over
      taskRef.current = null
      // The server is the source of truth: reload the thread for the final
      // assistant message (with its status, files and approval request).
      if (activeThreadRef.current === threadId) await loadThread(threadId)
      setStreaming(false)
      setStreamStatus('')
      setLiveSteps([])
      setLiveAnswer('')
      if (err) {
        setMessages(prev => [...prev, {
          id: `err-${Date.now()}`, role: 'assistant',
          content: `❌ Lost track of the task: ${err.message}`, steps: [], files: {},
        }])
      }
      refreshThreads()
      refreshFiles()
      refreshMemory()
    }

    const ctrl = api.followTask(taskId, {
      onEvent: ({ type, data }) => {
        if (type === 'task_queued') {
          setStreamStatus('Queued — waiting for a worker…')
        } else if (type === 'task_started') {
          setStreamStatus('Thinking…')
        } else if (type === 'status') {
          setStreamStatus(data.status)
        } else if (['plan', 'subagent', 'tool_started', 'tool_completed', 'file_updated'].includes(type)) {
          steps.push({ type, data })
          setLiveSteps([...steps])
        } else if (type === 'message_delta') {
          answer += data.delta
          setLiveAnswer(answer)
        } else if (type === 'message_reset') {
          answer = ''
          setLiveAnswer('')
        }
      },
      onDone: () => finish(),
      onError: (err) => finish(err),
    })
    taskRef.current = { taskId, threadId, ...ctrl }
  }

  /** Await a request that creates a task, then follow it; show errors inline. */
  const startTask = async (request, threadId) => {
    try {
      const { task_id } = await request
      followTask(task_id, threadId)
    } catch (err) {
      setMessages(prev => [...prev, {
        id: `err-${Date.now()}`, role: 'assistant',
        content: `❌ ${err.message}`, steps: [], files: {},
      }])
    }
  }

  /** Stop the followed task on the server (the Stop button). */
  const stopTask = () => taskRef.current?.cancel()

  /** Re-run the failed or cancelled task that produced `msg`. */
  const retryTask = (msg) => {
    if (streaming || !msg.task_id) return
    startTask(api.tasks.retry(msg.task_id), activeThread)
  }

  // ── Thread selection ───────────────────────────────────────────
  /** Open a thread from the sidebar, reattaching to its running task if any. */
  const selectThread = async (id) => {
    switchThread(id)
    const t = await loadThread(id)
    // A task keeps running on the server across reloads; pick it back up.
    if (t.active_task_id && activeThreadRef.current === id) followTask(t.active_task_id, id)
  }

  // ── Open the most recent thread on load ────────────────────────
  useEffect(() => {
    api.threads.list()
      .then(ts => { if (ts.length > 0) selectThread(ts[0].id) })
      .catch(() => {})
  }, [])  // eslint-disable-line react-hooks/exhaustive-deps -- run once on mount

  // ── Human-in-the-loop ──────────────────────────────────────────
  // The thread's last message, if it is an approval request nobody answered
  // yet. While set, the input is locked until the user decides.
  const pendingApproval = (() => {
    const last = messages[messages.length - 1]
    return last?.interrupt && !last.interrupt.resolved ? last : null
  })()

  /** Send the user's approve/edit/reject decisions; marks the card answered right away. */
  const submitDecisions = (msgId, decisions) => {
    if (streaming) return
    setMessages(prev => prev.map(m => m.id === msgId
      ? { ...m, interrupt: { ...m.interrupt, resolved: decisions.map(d => d.type) } }
      : m))
    startTask(api.chat.resume({ ...agentSettings(activeThread), decisions }), activeThread)
  }

  // ── Send message ───────────────────────────────────────────────
  /** Send the draft as a new task, creating a thread first if none is open. */
  const sendMessage = async () => {
    const text = input.trim()
    if (!text || streaming || pendingApproval) return

    let tid = activeThread
    if (!tid) {
      const t = await createThread()
      tid = t.id
      switchThread(t.id)
    }

    // Show the message immediately; the server copy replaces it on reload.
    const userMsg = { id: Date.now(), role: 'user', content: text, steps: [], files: {} }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'

    startTask(api.chat.start({ ...agentSettings(tid), message: text }), tid)
  }

  /** Enter sends; Shift+Enter inserts a newline. */
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div className="app-shell">
      {/* HEADER */}
      <header className="header">
        <div className="header-logo">
          <div className="header-logo-icon">🧠</div>
          <span>Deep Agents</span>
        </div>
        <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 8 }}>
          Production AI Platform
        </span>
        <div className="header-spacer" />
        <div className="header-badge">
          <div className="status-dot" />
          {models.find(m => m.id === model)?.name || model.split(':').slice(1).join(':')}
        </div>
        <div className="header-badge">
          <span>💾</span>
          {backend.replace('Backend', '')}
        </div>
        {activeThread && (
          <div className="header-badge">
            <span>🧵</span>
            {activeThread.slice(0, 8)}…
          </div>
        )}
      </header>

      {/* SIDEBAR */}
      <aside className="sidebar">
        <button className="new-thread-btn" onClick={handleNewThread}>
          <span>＋</span> New Thread
        </button>

        <div className="sidebar-section-label">Threads</div>
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {threads.map(t => (
            <button
              key={t.id}
              className={`thread-btn ${t.id === activeThread ? 'active' : ''}`}
              onClick={() => selectThread(t.id)}
            >
              <span className="thread-btn-icon">💬</span>
              <span className="truncate">{t.title}</span>
            </button>
          ))}
        </div>

        <div className="divider" />

        {/* Config */}
        <div className="config-panel">
          <div className="sidebar-section-label" style={{ paddingLeft: 4 }}>Configuration</div>

          <div className="config-group">
            <label className="config-label">Model</label>
            <select
              className="config-select"
              value={model}
              onChange={e => setModel(e.target.value)}
            >
              {models.map(m => (
                <option key={m.id} value={m.id}>{m.provider} · {m.name}</option>
              ))}
              {models.length === 0 && <option value={model}>{model}</option>}
            </select>
          </div>

          <div className="config-group">
            <label className="config-label">Backend</label>
            <div className="config-radios">
              {BACKEND_OPTIONS.map(opt => (
                <label
                  key={opt.value}
                  className={`config-radio-item ${backend === opt.value ? 'selected' : ''}`}
                >
                  <input
                    type="radio"
                    name="backend"
                    value={opt.value}
                    checked={backend === opt.value}
                    onChange={() => setBackend(opt.value)}
                  />
                  <div className={`config-radio-bullet ${backend === opt.value ? 'on' : ''}`} />
                  <div>
                    <div style={{ fontWeight: 500, fontSize: 12 }}>{opt.label}</div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{opt.desc}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          <div className="config-group">
            <label className="config-label">Features</label>
            {[
              ['AGENTS.md', useAgentsMd, setUseAgentsMd],
              ['Skills (/skills/)', useSkillsFlag, setUseSkillsFlag],
              ['Subagents', useSubagents, setUseSubagents],
              ['Require approval', requireApproval, setRequireApproval],
            ].map(([lbl, val, set]) => (
              <div className="toggle-row" key={lbl}>
                <span className="toggle-label">{lbl}</span>
                <div className={`toggle ${val ? 'on' : ''}`} onClick={() => set(!val)}>
                  <div className="toggle-knob" />
                </div>
              </div>
            ))}
          </div>

          <div className="config-group">
            <label className="config-label">API Keys</label>
            <div className="key-badges">
              {['OPENAI_API_KEY','GROQ_API_KEY','ANTHROPIC_API_KEY','GEMINI_API_KEY','TAVILY_API_KEY'].map(k => (
                <div className="key-badge" key={k}>
                  <div className={`dot ${keyStatus.api_keys?.[k] ? 'ok' : 'miss'}`} />
                  {k.replace('_API_KEY', '')}
                </div>
              ))}
            </div>
          </div>
        </div>
      </aside>

      {/* MAIN CHAT */}
      <div className="main-area">
        <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          {/* Chat Column */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div className="chat-area">
              {messages.length === 0 && !streaming ? (
                <div className="chat-empty">
                  <div className="chat-empty-icon">🧠</div>
                  <div className="chat-empty-title">Deep Agents Chat</div>
                  <p style={{ fontSize: 13, textAlign: 'center', maxWidth: 400 }}>
                    Planning · File System · Context Engineering · Skills · Subagents · Backends · Thread Memory
                  </p>
                  <div className="chips-grid">
                    {DEMO_PROMPTS.map((p, i) => (
                      <button key={i} className="chip" onClick={() => { setInput(p.text); textareaRef.current?.focus() }}>
                        <strong>{p.label}</strong>
                        {p.text.slice(0, 60)}…
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <>
                  {messages.map(m => (
                    <ChatMessage
                      key={m.id}
                      msg={m}
                      canDecide={!streaming && m === pendingApproval}
                      onDecide={(decisions) => submitDecisions(m.id, decisions)}
                      canRetry={!streaming && m === messages[messages.length - 1]
                        && ['failed', 'cancelled'].includes(m.status)}
                      onRetry={() => retryTask(m)}
                    />
                  ))}
                  {streaming && (
                    <TypingBubble
                      status={streamStatus}
                      steps={liveSteps}
                      answer={liveAnswer}
                    />
                  )}
                </>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Input Area */}
            <div className="input-area">
              <div className="input-row">
                <textarea
                  ref={textareaRef}
                  className="chat-textarea"
                  rows={1}
                  placeholder={pendingApproval
                    ? 'Approve, edit or reject the pending action above to continue…'
                    : 'Ask anything — research, code, AWS, LangGraph, plan a task…'}
                  value={input}
                  onChange={handleInput}
                  onKeyDown={handleKeyDown}
                  disabled={streaming || !!pendingApproval}
                />
                {streaming ? (
                  <button className="send-btn stop" onClick={stopTask} title="Stop the running task">
                    ■
                  </button>
                ) : (
                  <button
                    className="send-btn"
                    onClick={sendMessage}
                    disabled={!input.trim() || !!pendingApproval}
                    title="Send (Enter)"
                  >
                    ↑
                  </button>
                )}
              </div>
              <div style={{ display: 'flex', gap: 12, marginTop: 6, fontSize: 11, color: 'var(--text-muted)' }}>
                <span>Enter to send · Shift+Enter for newline</span>
                {activeThread && <span style={{ marginLeft: 'auto' }}>🧵 {activeThread.slice(0, 8)}…</span>}
              </div>
            </div>
          </div>

          {/* Right Panel */}
          <div className="right-panel">
            <div className="right-panel-tabs">
              {[
                ['skills', '🎯 Skills'],
                ['files', '📁 Files'],
                ['memory', '🧠 Memory'],
              ].map(([key, label]) => (
                <button
                  key={key}
                  className={`right-tab ${rightTab === key ? 'active' : ''}`}
                  onClick={() => setRightTab(key)}
                >
                  {label}
                </button>
              ))}
            </div>

            <div className="right-panel-content">
              {/* Skills Tab */}
              {rightTab === 'skills' && (
                <div>
                  {skills.map(s => (
                    <div className="skill-card" key={s.name}>
                      <h4>
                        {s.name === 'aws' ? '☁️' : s.name === 'langgraph' ? '🕸️' : s.name === 'python' ? '🐍' : '📝'}
                        {' '}{s.name}
                      </h4>
                      <p>{s.description.slice(0, 100)}…</p>
                      <div className="skill-tags">
                        {s.files.map(f => <span key={f} className="skill-tag">{f}</span>)}
                      </div>
                    </div>
                  ))}
                  {skills.length === 0 && (
                    <p style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: 20 }}>
                      No skills loaded. Skills are loaded from deep_agent/skills/
                    </p>
                  )}
                </div>
              )}

              {/* Files Tab */}
              {rightTab === 'files' && (
                <div>
                  <div className="files-section-label">Agent files · this thread</div>
                  {threadFiles.length === 0 ? (
                    <p className="files-empty">No files written in this thread yet.</p>
                  ) : threadFiles.map(f => (
                    <details key={f.path} className="thread-file">
                      <summary>📄 {f.path} <span className="file-size">{formatSize(f.size)}</span></summary>
                      <pre className="code-view">{f.content}</pre>
                    </details>
                  ))}
                  <div className="files-section-label">Workspace · deep_agent/</div>
                  {selFile && fileContent !== null && (
                    <div style={{ marginBottom: 12 }}>
                      <div style={{ fontSize: 11, color: 'var(--accent-hover)', marginBottom: 6, fontFamily: 'JetBrains Mono' }}>
                        📄 {selFile}
                      </div>
                      <pre className="code-view">{fileContent}</pre>
                    </div>
                  )}
                  <div className="file-tree">
                    {files.map(f => (
                      <div
                        key={f.relative_path}
                        className={`file-item ${f.is_dir ? 'dir' : ''} ${selFile === f.relative_path ? 'active' : ''}`}
                        style={{ paddingLeft: f.relative_path.split('/').length * 8 }}
                        onClick={() => !f.is_dir && openFile(f.relative_path)}
                      >
                        <span>{f.is_dir ? '📂' : '📄'}</span>
                        <span className="file-name">{f.name}</span>
                        {!f.is_dir && <span className="file-size">{formatSize(f.size)}</span>}
                      </div>
                    ))}
                    {files.length === 0 && (
                      <p style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: 20 }}>
                        No files found in deep_agent/
                      </p>
                    )}
                  </div>
                </div>
              )}

              {/* Memory Tab */}
              {rightTab === 'memory' && (
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Cross-thread InMemoryStore</span>
                    <button
                      onClick={refreshMemory}
                      style={{ background: 'none', border: 'none', color: 'var(--accent-hover)', cursor: 'pointer', fontSize: 11 }}
                    >
                      ↻ Refresh
                    </button>
                  </div>
                  {memItems.map(item => (
                    <div className="memory-item" key={item.key}>
                      <button className="mem-del-btn" onClick={() => deleteMemory(item.key)}>×</button>
                      <div className="memory-key">{item.key}</div>
                      <div className="memory-val">
                        {typeof item.value === 'object'
                          ? JSON.stringify(item.value).slice(0, 80) + (JSON.stringify(item.value).length > 80 ? '…' : '')
                          : String(item.value).slice(0, 80)}
                      </div>
                    </div>
                  ))}
                  {memItems.length === 0 && (
                    <p style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: 20 }}>
                      No memory items yet. Use StoreBackend to persist data across threads.
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
