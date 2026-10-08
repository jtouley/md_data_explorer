/**
 * Enrichment pending suggestions — pure helpers and DOM rendering (ADR011 / Phase 6b).
 */

/**
 * @param {Record<string, unknown>} raw - Parsed JSON from GET .../enrichments/pending
 * @returns {{ suggestions: object[], total: number }}
 */
export function normalizePendingResponse(raw) {
  const suggestions = Array.isArray(raw?.suggestions) ? raw.suggestions : [];
  const total = typeof raw?.total === 'number' ? raw.total : suggestions.length;
  return { suggestions, total };
}

/**
 * Stable row data for display / testing.
 * @param {object} s - PendingSuggestion-shaped object from API
 * @returns {[string, string][]}
 */
export function suggestionToRows(s) {
  return [
    ['Operation', String(s?.operation ?? '')],
    ['Column', String(s?.column ?? '')],
    ['Suggested', String(s?.suggested_value ?? '')],
    ['Current', s?.current_value != null && s.current_value !== '' ? String(s.current_value) : '—'],
    ['Confidence', String(s?.confidence ?? '')],
    ['Model', String(s?.model_id ?? '')],
  ];
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}

/**
 * Render pending suggestions into a container (createElement only, no raw HTML injection).
 * @param {HTMLElement} container
 * @param {object[]} suggestions
 * @param {{ onAccept: (patchId: string) => void, onReject: (patchId: string) => void }} handlers
 */
export function renderEnrichmentCards(container, suggestions, handlers) {
  container.replaceChildren();

  if (!suggestions.length) {
    const empty = document.createElement('p');
    empty.className = 'enrichment-empty';
    empty.textContent = 'No pending metadata suggestions for this dataset.';
    container.appendChild(empty);
    return;
  }

  for (const s of suggestions) {
    const patchId = s?.patch_id;
    if (!patchId) continue;

    const card = document.createElement('article');
    card.className = 'enrichment-card';
    card.dataset.patchId = patchId;

    const title = document.createElement('h3');
    title.className = 'enrichment-card-title';
    title.textContent = `${s.operation ?? '?'} · ${s.column ?? '?'}`;
    card.appendChild(title);

    const dl = document.createElement('dl');
    dl.className = 'enrichment-dl';
    for (const [dt, dd] of suggestionToRows(s)) {
      if (dt === 'Operation' || dt === 'Column') continue;
      const dti = document.createElement('dt');
      dti.textContent = dt;
      const ddi = document.createElement('dd');
      ddi.innerHTML = escapeHtml(dd);
      dl.appendChild(dti);
      dl.appendChild(ddi);
    }
    card.appendChild(dl);

    const actions = document.createElement('div');
    actions.className = 'enrichment-card-actions';

    const accept = document.createElement('button');
    accept.type = 'button';
    accept.className = 'btn-enrichment btn-enrichment-accept';
    accept.textContent = 'Accept';
    accept.dataset.testid = 'enrichment-accept';
    accept.addEventListener('click', () => handlers.onAccept(patchId));

    const reject = document.createElement('button');
    reject.type = 'button';
    reject.className = 'btn-enrichment btn-enrichment-reject';
    reject.textContent = 'Reject';
    reject.dataset.testid = 'enrichment-reject';
    reject.addEventListener('click', () => handlers.onReject(patchId));

    actions.append(accept, reject);
    card.appendChild(actions);
    container.appendChild(card);
  }
}
