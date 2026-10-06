// Client for the Deep Agents backend. All paths are relative (/api, /ws), so
// the same build works behind Vite's dev proxy and when FastAPI serves it.
const BASE = '/api'

/** Event types after which a task produces nothing more. */
export const TERMINAL_EVENTS = ['task_completed', 'task_failed', 'task_cancelled']

/** Resolve after `ms` milliseconds. */
const sleep = (ms) => new Promise(r => setTimeout(r, ms))

/**
 * POST JSON and return the parsed response.
 * @param {string} path  path under /api
 * @param {object} [body]  JSON body; omitted when undefined
 * @returns {Promise<object>}
 * @throws {Error} with the server's `detail` message on a non-2xx response
 */
async function postJson(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`)
  return data
}

/** GET a path under /api and parse the JSON response. */
const getJson = (path) => fetch(`${BASE}${path}`).then(r => r.json())

export const api = {
  // ── Service ──────────────────────────────────────────────────────
  /** Health, Redis reachability and which provider API keys are set. */
  health: () => getJson('/health'),
  /** Model catalogue for the model picker. */
  models: () => getJson('/models'),

  // ── Threads ──────────────────────────────────────────────────────
  threads: {
    /** All threads, most recent first (without messages). */
    list: () => getJson('/threads'),
    /** Create a thread; `title` defaults to "New Conversation" on the server. */
    create: (title) => postJson('/threads', { title }),
    /** One thread with its messages and `active_task_id`. */
    get: (id) => getJson(`/threads/${id}`),
    /** Delete a thread, its tasks and its agent state. */
    delete: (id) => fetch(`${BASE}/threads/${id}`, { method: 'DELETE' }).then(r => r.json()),
    /** Task history of a thread. */
    tasks: (id) => getJson(`/threads/${id}/tasks`),
    /** The agent's virtual files for a thread. */
    files: (id) => getJson(`/threads/${id}/files`),
  },

  // ── Chat & tasks ─────────────────────────────────────────────────
  // A chat turn is a task: POST returns a task_id at once, a worker runs the
  // agent, and the UI follows the task's events. The task keeps running if
  // the page closes.
  chat: {
    /** Queue a user message; resolves to { task_id, thread_id, status }. */
    start: (payload) => postJson('/chat', payload),
    /** Answer a pending approval; starts a new task on the thread. */
    resume: (payload) => postJson('/chat/resume', payload),
  },

  tasks: {
    /** Task status, progress and timestamps. */
    get: (id) => getJson(`/tasks/${id}`),
    /** Final answer and files of a finished task. */
    result: (id) => getJson(`/tasks/${id}/result`),
    /** Stop a queued or running task. */
    cancel: (id) => postJson(`/tasks/${id}/cancel`),
    /** Re-run a failed or cancelled task; resolves to the new task. */
    retry: (id) => postJson(`/tasks/${id}/retry`),
  },

  /**
   * Follow a task's events until it finishes.
   *
   * Uses the WebSocket and reconnects with backoff, resuming after the last
   * event_id it saw, so no event is lost or repeated. After 3 failed connection
   * attempts it falls back to long-polling.
   *
   * @param {string} taskId
   * @param {{ onEvent: (evt: object) => void, onDone?: () => void,
   *           onError?: (err: Error) => void }} handlers
   *   onEvent gets each event envelope { event_id, type, data, ... };
   *   onDone fires once after a terminal event; onError if the task is lost.
   * @returns {{ cancel: () => void, close: () => void }}
   *   cancel stops the task on the server; close only stops following it here.
   */
  followTask: (taskId, { onEvent, onDone, onError }) => {
    let last = 0          // highest event_id delivered to onEvent
    let finished = false  // set once; stops every transport
    let ws = null
    let wsFailures = 0    // consecutive attempts that never opened

    /** Finish following: notify exactly once, then drop the socket. */
    const end = (err) => {
      if (finished) return
      finished = true
      if (err) onError?.(err)
      else onDone?.()
      try { ws?.close() } catch { /* already closed */ }
    }

    /** Deliver one event, skipping pings and anything already delivered. */
    const handle = (evt) => {
      if (finished || evt.type === 'ping' || evt.event_id <= last) return
      last = evt.event_id
      onEvent(evt)
      if (TERMINAL_EVENTS.includes(evt.type)) end()
    }

    /** Fallback transport: long-poll; each request returns within ~20 s. */
    const poll = async () => {
      let failures = 0
      while (!finished) {
        try {
          const res = await fetch(`${BASE}/tasks/${taskId}/events?after=${last}`)
          if (res.status === 404) return end(new Error('Task not found.'))
          if (!res.ok) throw new Error(`HTTP ${res.status}`)
          const batch = await res.json()
          failures = 0
          batch.events.forEach(handle)
          if (batch.done) end()
        } catch (err) {
          if (++failures > 5) return end(err)
          await sleep(Math.min(1000 * 2 ** failures, 10000))  // 2 s, 4 s, ... max 10 s
        }
      }
    }

    /** Primary transport: WebSocket, resuming after `last`. */
    const connect = () => {
      if (finished) return
      let opened = false
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
      ws = new WebSocket(`${proto}//${location.host}/ws/tasks/${taskId}?last_event_id=${last}`)
      ws.onopen = () => { opened = true; wsFailures = 0 }
      ws.onmessage = (msg) => handle(JSON.parse(msg.data))
      ws.onclose = (e) => {
        if (finished) return
        if (e.code === 4404) return end(new Error('Task not found.'))
        // A clean close without a terminal event means the task ended in a way
        // this socket missed; one poll settles it. Repeated failures to connect
        // mean WebSockets are blocked (proxy, network): switch to polling.
        if (e.code === 1000) return poll()
        if (!opened) wsFailures += 1
        if (wsFailures >= 3) return poll()
        setTimeout(connect, Math.min(1000 * 2 ** wsFailures, 8000))  // 1 s, 2 s, 4 s ...
      }
    }

    connect()
    return {
      cancel: () => {
        // Prefer the open socket; fall back to REST if it is reconnecting.
        if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'cancel' }))
        else api.tasks.cancel(taskId).catch(() => {})
      },
      close: () => { finished = true; ws?.close() },
    }
  },

  // ── Workspace files (deep_agent/ on disk) ────────────────────────
  files: {
    /** Every file and folder under deep_agent/. */
    list: () => getJson('/files'),
    /** Text content of one file (path relative to deep_agent/). */
    content: (path) => getJson(`/files/content?path=${encodeURIComponent(path)}`),
    /** Create or overwrite a file. */
    write: (path, content) => postJson('/files', { path, content }),
    /** Delete a file or folder. */
    delete: (path) => fetch(`${BASE}/files/${encodeURIComponent(path)}`, { method: 'DELETE' }).then(r => r.json()),
  },

  // ── Skills ───────────────────────────────────────────────────────
  skills: {
    /** Catalogue of skills in deep_agent/skills/. */
    list: () => getJson('/skills'),
    /** All Markdown files of one skill. */
    get: (name) => getJson(`/skills/${name}`),
  },

  // ── Cross-thread memory store ────────────────────────────────────
  memory: {
    /** Entries in a namespace (default "memories"). */
    list: (ns = 'memories') => getJson(`/memory?namespace=${ns}`),
    /** Add or replace an entry. */
    create: (key, value, namespace = ['memories']) => postJson('/memory', { key, value, namespace }),
    /** Delete an entry (keys may contain "/"). */
    delete: (key, ns = 'memories') => fetch(`${BASE}/memory/${encodeURIComponent(key)}?namespace=${ns}`, {
      method: 'DELETE',
    }).then(r => r.json()),
  },
}
