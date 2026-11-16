// SPDX-License-Identifier: MIT
/**
 * Session sidebar helpers (Phase 7) — pure functions + DOM rendering.
 *
 * Conversations are grouped by date and titled by their first question when it is known.
 * The API does not persist questions, so titles exist only for conversations started in this
 * run (kept in memory, never in browser storage: questions can identify patients).
 */
import { icon } from './icons.js';

export const UNTITLED_SESSION = 'Earlier conversation';

/**
 * Make sure the conversation exists on the server, creating it at most once. The in-flight
 * creation is kept on `state.sessionCreating`, so a second question asked while the first
 * creation is still running (for example after Stop) waits for that same creation instead of
 * starting another session. Setting `sessionCreating` to null (New chat, another cohort)
 * abandons it: a creation that returns after that changes nothing.
 * @param {{ currentDatasetId?: string | null, sessionId?: string, sessionOnServer: boolean,
 *   sessionCreating: Promise<void> | null }} state
 * @param {{ createSession?: (datasetId: string) => Promise<{ session_id?: string }> }} api
 * @param {{ warn?: (message: string, detail?: string) => void }} [opts]
 * @returns {Promise<void>}
 */
export function ensureServerSession(state, api, opts = {}) {
  if (state.sessionOnServer || typeof api?.createSession !== 'function') return Promise.resolve();
  if (state.sessionCreating) return state.sessionCreating;
  const warn = opts.warn ?? ((message, detail) => console.warn(message, detail));
  /** @type {Promise<void>} */
  const pending = Promise.resolve()
    .then(() => api.createSession?.(/** @type {string} */ (state.currentDatasetId)))
    .then(
      (res) => {
        if (state.sessionCreating !== pending) return;
        if (res?.session_id) {
          state.sessionId = res.session_id;
          state.sessionOnServer = true;
        }
      },
      (error) => warn('Could not create a session; continuing without one:', error?.message)
    )
    .finally(() => {
      if (state.sessionCreating === pending) state.sessionCreating = null;
    });
  state.sessionCreating = pending;
  return pending;
}
const TITLE_MAX = 80;

/**
 * @param {Record<string, unknown>} raw
 * @returns {{ sessions: object[], total: number }}
 */
export function normalizeSessionListResponse(raw) {
  const sessions = Array.isArray(raw?.sessions) ? raw.sessions : [];
  const total = typeof raw?.total === 'number' ? raw.total : sessions.length;
  return { sessions, total };
}

/**
 * Most-recently-updated first.
 * @param {object[]} sessions
 * @returns {object[]}
 */
export function sortSessionsNewestFirst(sessions) {
  return [...sessions].sort((a, b) => {
    const ta = String(a?.updated_at ?? a?.created_at ?? '');
    const tb = String(b?.updated_at ?? b?.created_at ?? '');
    return tb.localeCompare(ta);
  });
}

/**
 * Human-readable relative timestamp.
 * @param {string} iso
 * @param {number} [now]
 * @returns {string}
 */
export function formatRelativeTime(iso, now = Date.now()) {
  if (!iso) return '';
  const date = new Date(iso);
  if (isNaN(date.getTime())) return '';
  const diffMin = Math.floor((now - date.getTime()) / 60000);
  if (diffMin < 1) return 'just now';
  if (diffMin < 60) return `${diffMin} min ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr} h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 30) return `${diffDay} ${diffDay === 1 ? 'day' : 'days'} ago`;
  return date.toLocaleDateString();
}

/**
 * Short title from a question (first line, at most 80 characters).
 * @param {string} question
 * @returns {string}
 */
export function titleFromQuestion(question) {
  const line = String(question || '')
    .split('\n')[0]
    .replace(/\s+/g, ' ')
    .trim();
  return line.length > TITLE_MAX ? `${line.slice(0, TITLE_MAX - 1).trimEnd()}…` : line;
}

/**
 * Title for a session row: API title, then the first question asked in this run.
 * @param {Record<string, unknown>} session
 * @param {Map<string, string>} [titles]
 * @returns {string}
 */
export function sessionTitle(session, titles) {
  if (typeof session?.title === 'string' && session.title.trim()) {
    return titleFromQuestion(session.title);
  }
  const known = titles?.get(String(session?.session_id ?? ''));
  return known || UNTITLED_SESSION;
}

/**
 * Group sessions (newest first) into Today / Yesterday / Previous 7 days / Older.
 * @param {object[]} sessions
 * @param {number} [now]
 * @returns {{ label: string, sessions: object[] }[]}
 */
export function groupSessionsByDate(sessions, now = Date.now()) {
  const startOfToday = new Date(now);
  startOfToday.setHours(0, 0, 0, 0);
  const today = startOfToday.getTime();
  const day = 24 * 60 * 60 * 1000;
  const order = ['Today', 'Yesterday', 'Previous 7 days', 'Older'];
  const buckets = new Map(order.map((label) => [label, []]));
  for (const s of sortSessionsNewestFirst(sessions)) {
    const t = new Date(String(s?.updated_at ?? s?.created_at ?? '')).getTime();
    let label = 'Older';
    if (!isNaN(t)) {
      if (t >= today) label = 'Today';
      else if (t >= today - day) label = 'Yesterday';
      else if (t >= today - 7 * day) label = 'Previous 7 days';
    }
    buckets.get(label).push(s);
  }
  return order
    .map((label) => ({ label, sessions: buckets.get(label) }))
    .filter((g) => g.sessions.length);
}

/**
 * Ask in the row itself before deleting (replaces window.confirm).
 * @param {HTMLLIElement} li
 * @param {() => void} onDelete
 * @param {() => void} restore
 */
function showDeleteConfirm(li, onDelete, restore) {
  const saved = [...li.childNodes];
  li.classList.add('session-item--confirming');
  li.replaceChildren();

  const text = document.createElement('p');
  text.className = 'session-confirm-text';
  text.textContent = "Delete this conversation? It can't be recovered.";

  const actions = document.createElement('div');
  actions.className = 'session-confirm-actions';
  const del = document.createElement('button');
  del.type = 'button';
  del.className = 'btn btn-danger btn-sm';
  del.dataset.testid = 'session-delete-confirm';
  del.textContent = 'Delete';
  const keep = document.createElement('button');
  keep.type = 'button';
  keep.className = 'btn btn-secondary btn-sm';
  keep.dataset.testid = 'session-delete-cancel';
  keep.textContent = 'Keep';

  const onKey = (e) => {
    if (e.key === 'Escape') back();
  };
  const back = () => {
    li.removeEventListener('keydown', onKey);
    li.classList.remove('session-item--confirming');
    li.replaceChildren(...saved);
    restore();
  };
  del.addEventListener('click', (e) => {
    e.stopPropagation();
    del.disabled = true;
    keep.disabled = true;
    onDelete();
  });
  keep.addEventListener('click', (e) => {
    e.stopPropagation();
    back();
  });
  li.addEventListener('keydown', onKey);

  actions.append(keep, del);
  li.append(text, actions);
  keep.focus();
}

/**
 * @param {HTMLElement} container - <ul> element
 * @param {object[]} sessions
 * @param {string|null} activeSessionId
 * @param {{ onSelect: (id: string) => void, onDelete: (id: string) => void }} handlers
 * @param {{ titles?: Map<string, string>, datasetNames?: Map<string, string>, now?: number }} [opts]
 */
export function renderSessionList(container, sessions, activeSessionId, handlers, opts = {}) {
  container.replaceChildren();
  const now = opts.now ?? Date.now();
  const groups = groupSessionsByDate(sessions, now);

  if (!groups.length) {
    const li = document.createElement('li');
    li.className = 'session-list-empty';
    li.textContent = 'Your conversations will appear here.';
    container.appendChild(li);
    return;
  }

  for (const group of groups) {
    const groupLi = document.createElement('li');
    groupLi.className = 'session-group';
    // h2: the sidebar's h1 (the app name) is the only heading above it.
    const heading = document.createElement('h2');
    heading.className = 'session-group-label';
    heading.textContent = group.label;
    const list = document.createElement('ul');
    list.className = 'session-group-list';
    groupLi.append(heading, list);

    for (const s of group.sessions) {
      const id = String(s.session_id ?? '');
      const li = document.createElement('li');
      li.className = 'session-item';
      li.dataset.sessionId = id;
      const active = s.session_id === activeSessionId;
      if (active) li.classList.add('session-item--active');

      const main = document.createElement('button');
      main.type = 'button';
      main.className = 'session-item-main';
      main.dataset.testid = 'session-open';
      if (active) main.setAttribute('aria-current', 'true');

      const label = document.createElement('span');
      label.className = 'session-item-label';
      label.textContent = sessionTitle(s, opts.titles);

      const meta = document.createElement('span');
      meta.className = 'session-item-meta';
      const datasetName = s.dataset_id
        ? opts.datasetNames?.get(String(s.dataset_id)) || String(s.dataset_id)
        : '';
      const ts = formatRelativeTime(s.updated_at || s.created_at, now);
      meta.textContent = [datasetName, ts].filter(Boolean).join(' · ');

      main.append(label, meta);
      main.addEventListener('click', () => handlers.onSelect(id));

      const delBtn = document.createElement('button');
      delBtn.type = 'button';
      delBtn.className = 'btn-session-delete';
      delBtn.innerHTML = icon('trash');
      delBtn.title = 'Delete conversation';
      delBtn.setAttribute('aria-label', `Delete conversation: ${label.textContent}`);
      delBtn.dataset.testid = 'session-delete';
      delBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        showDeleteConfirm(
          li,
          () => handlers.onDelete(id),
          () => delBtn.focus()
        );
      });

      li.append(main, delBtn);
      list.appendChild(li);
    }
    container.appendChild(groupLi);
  }
}
