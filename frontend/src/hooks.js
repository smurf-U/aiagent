// Data hooks for the sidebar and right panel. Each loads once on mount and
// exposes a refresh() so the app can reload after a task changes something.
// Load errors are swallowed on purpose: these panels are secondary, and an
// empty list is a better failure mode than breaking the chat.
import { useState, useEffect } from 'react'
import { api } from './api'

/**
 * Conversation threads for the sidebar.
 * @returns {{ threads: object[], loading: boolean, refresh: () => Promise<void>,
 *             createThread: (title?: string) => Promise<object> }}
 */
export function useThreads() {
  const [threads, setThreads] = useState([])
  const [loading, setLoading] = useState(true)

  const refresh = async () => {
    try {
      setThreads(await api.threads.list())
    } catch { /* keep the previous list */ }
    setLoading(false)
  }

  useEffect(() => { refresh() }, [])

  /** Create a thread on the server and put it at the top of the list. */
  const createThread = async (title) => {
    const thread = await api.threads.create(title)
    setThreads(prev => [thread, ...prev])
    return thread
  }

  return { threads, loading, refresh, createThread }
}

/**
 * Skill catalogue from deep_agent/skills/ (read-only).
 * @returns {object[]} skills with name, description and files
 */
export function useSkills() {
  const [skills, setSkills] = useState([])
  useEffect(() => {
    api.skills.list().then(setSkills).catch(() => {})
  }, [])
  return skills
}

/**
 * Files under deep_agent/ on disk, plus the content of the selected one.
 * @returns {{ files: object[], selected: string|null, content: string|null,
 *             loading: boolean, refresh: () => Promise<void>,
 *             openFile: (path: string) => Promise<void> }}
 */
export function useFiles() {
  const [files, setFiles] = useState([])
  const [selected, setSelected] = useState(null)
  const [content, setContent] = useState(null)
  const [loading, setLoading] = useState(false)

  const refresh = async () => {
    try {
      setFiles(await api.files.list())
    } catch { /* keep the previous list */ }
  }

  /** Load a file's text and mark it as selected. */
  const openFile = async (path) => {
    setLoading(true)
    setSelected(path)
    try {
      const data = await api.files.content(path)
      setContent(data.content)
    } catch {
      setContent('Error loading file.')
    }
    setLoading(false)
  }

  useEffect(() => { refresh() }, [])

  return { files, selected, content, loading, refresh, openFile }
}

/**
 * Cross-thread memory store entries (the "memories" namespace).
 * @returns {{ items: object[], refresh: () => Promise<void>,
 *             deleteItem: (key: string) => Promise<void> }}
 */
export function useMemory() {
  const [items, setItems] = useState([])

  const refresh = async () => {
    try {
      setItems(await api.memory.list())
    } catch { /* keep the previous list */ }
  }

  /** Delete an entry on the server, then drop it from the list. */
  const deleteItem = async (key) => {
    await api.memory.delete(key)
    setItems(prev => prev.filter(i => i.key !== key))
  }

  useEffect(() => { refresh() }, [])

  return { items, refresh, deleteItem }
}

/**
 * Model catalogue for the model picker.
 * @returns {object[]} models with id, name, provider and badge
 */
export function useModels() {
  const [models, setModels] = useState([])
  useEffect(() => {
    api.models().then(setModels).catch(() => {})
  }, [])
  return models
}
