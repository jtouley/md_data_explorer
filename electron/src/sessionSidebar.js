/**
 * Session sidebar helpers (Phase 7) — pure functions + DOM rendering.
 */

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
 * @returns {string}
 */
export function formatRelativeTime(iso) {
  if (!iso) return '';
  const date = new Date(iso);
  if (isNaN(date.getTime())) return '';
  const now = Date.now();
  const diffMs = now - date.getTime();
  const diffMin = Math.floor(diffMs / 60000);
  if (diffMin < 1) return 'just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 30) return `${diffDay}d ago`;
  return date.toLocaleDateString();
}

/**
 * @param {HTMLElement} container - <ul> element
 * @param {object[]} sessions
 * @param {string|null} activeSessionId
 * @param {{ onSelect: (id: string) => void, onDelete: (id: string) => void }} handlers
 */
export function renderSessionList(container, sessions, activeSessionId, handlers) {
  container.replaceChildren();
  const ordered = sortSessionsNewestFirst(sessions);

  if (!ordered.length) {
    const li = document.createElement('li');
    li.className = 'session-list-empty';
    li.textContent = 'No sessions yet.';
    container.appendChild(li);
    return;
  }

  for (const s of ordered) {
    const li = document.createElement('li');
    li.className = 'session-item';
    li.dataset.sessionId = String(s.session_id ?? '');
    if (s.session_id === activeSessionId) {
      li.classList.add('session-item--active');
    }

    const info = document.createElement('div');
    info.className = 'session-item-info';

    const label = document.createElement('div');
    label.className = 'session-item-label';
    label.textContent = s.dataset_id ? `Dataset: ${s.dataset_id}` : 'Session';
    info.appendChild(label);

    const meta = document.createElement('div');
    meta.className = 'session-item-meta';
    const msgs = typeof s.message_count === 'number' ? `${s.message_count} msgs` : '';
    const ts = formatRelativeTime(s.updated_at || s.created_at);
    meta.textContent = [msgs, ts].filter(Boolean).join(' · ');
    info.appendChild(meta);

    li.appendChild(info);

    const delBtn = document.createElement('button');
    delBtn.type = 'button';
    delBtn.className = 'btn-session-delete';
    delBtn.textContent = '✕';
    delBtn.title = 'Delete session';
    delBtn.dataset.testid = 'session-delete';
    delBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      handlers.onDelete(String(s.session_id));
    });
    li.appendChild(delBtn);

    li.addEventListener('click', () => {
      handlers.onSelect(String(s.session_id));
    });

    container.appendChild(li);
  }
}
