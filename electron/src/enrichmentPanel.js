// SPDX-License-Identifier: MIT
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
    ['Variable', String(s?.column ?? '')],
    ['Suggested', String(s?.suggested_value ?? '')],
    ['Current', s?.current_value != null && s.current_value !== '' ? String(s.current_value) : '—'],
    ['Confidence', String(s?.confidence ?? '')],
    ['Model', String(s?.model_id ?? '')],
  ];
}

/**
 * Run one accept/reject/revert call and refresh the panel. A stale patch (resolved or removed
 * elsewhere) also refreshes, so it disappears, and then shows why.
 */
async function runOneEnrichmentDecision(call, { reload, showMessage, fallback, isCurrent }) {
  // The user moved to another dataset after this click: the decision is still sent (for the
  // dataset it was made on), but its reload and message would land on the wrong panel.
  const stillCurrent = () => !isCurrent || isCurrent();
  try {
    const res = await call();
    if (!stillCurrent()) return;
    if (res?.stale) {
      await reload();
      showMessage(res.message || fallback);
      return;
    }
    if (res && res.success === false) {
      throw new Error(res.message || fallback);
    }
    await reload();
  } catch (error) {
    console.error(`${fallback}:`, error);
    if (stillCurrent()) showMessage(error?.message || fallback);
  }
}

let decisionQueue = Promise.resolve();

/**
 * Run an accept/reject/revert decision after any earlier one has finished its reload and
 * message. Without the queue, a double-click's slower success reload overwrote the 409 message
 * of the second click with the idle text. The second click is queued, not dropped, so its
 * message is shown.
 * @param {() => Promise<{ success?: boolean, stale?: boolean, message?: string } | undefined>} call
 * Callers must bind the dataset id at click time inside `call`; `isCurrent` says whether that
 * dataset is still the one on screen when the queued work runs.
 * @param {{ reload: () => Promise<void>, showMessage: (text: string) => void, fallback: string,
 *   isCurrent?: () => boolean }} opts
 * @returns {Promise<void>}
 */
export function runEnrichmentDecision(call, opts) {
  const next = decisionQueue.then(() => runOneEnrichmentDecision(call, opts));
  // The caller still sees a rejection (say a throwing showMessage), but the queue itself must
  // never hold a rejected promise, or every later decision would be skipped.
  decisionQueue = next.catch(() => {});
  return next;
}

/**
 * Orders the writers of the enrichment panel. A load's content is dropped when a newer load has
 * started (it is out of date), and a load does not write a hint (the suggestions hint or the
 * history hint) after a decision message was shown there, so a refresh that started earlier
 * cannot replace, say, the 409 explanation.
 * @returns {{ startLoad: () => number, isLatestLoad: (token: number) => boolean,
 *   mayWriteHint: (token: number, hint: string) => boolean, noteMessage: (hint: string) => void }}
 */
export function createEnrichmentPanelGate() {
  let latestLoad = 0;
  /** @type {Map<string, number>} hint -> load token current when a message was last shown */
  const messageAfterLoad = new Map();
  return {
    startLoad() {
      latestLoad += 1;
      return latestLoad;
    },
    isLatestLoad(token) {
      return token === latestLoad;
    },
    mayWriteHint(token, hint) {
      return token === latestLoad && (messageAfterLoad.get(hint) ?? 0) < token;
    },
    noteMessage(hint) {
      messageAfterLoad.set(hint, latestLoad);
    },
  };
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
    empty.textContent = 'No suggestions for this cohort.';
    container.appendChild(empty);
    return;
  }

  for (const s of suggestions) {
    const patchId = s?.patch_id;
    if (!patchId) continue;

    const card = document.createElement('article');
    card.className = 'enrichment-card';
    card.dataset.patchId = patchId;

    // h4: each card sits under the section's h3 ("Variable suggestions").
    const title = document.createElement('h4');
    title.className = 'enrichment-card-title';
    title.textContent = `${s.operation ?? '?'} · ${s.column ?? '?'}`;
    card.appendChild(title);

    const dl = document.createElement('dl');
    dl.className = 'enrichment-dl';
    for (const [dt, dd] of suggestionToRows(s)) {
      if (dt === 'Operation' || dt === 'Variable') continue;
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
