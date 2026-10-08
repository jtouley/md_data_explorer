/**
 * Enrichment patch history (ADR011 / Phase 6c) — helpers + table rendering.
 */

/**
 * @param {Record<string, unknown>} raw - GET .../enrichments/history JSON
 * @returns {{ patches: object[], total: number }}
 */
export function normalizeHistoryResponse(raw) {
  const patches = Array.isArray(raw?.patches) ? raw.patches : [];
  const total = typeof raw?.total === 'number' ? raw.total : patches.length;
  return { patches, total };
}

/**
 * Newest first by ISO created_at.
 * @param {object[]} patches
 * @returns {object[]}
 */
export function sortPatchesNewestFirst(patches) {
  return [...patches].sort((a, b) => {
    const ta = String(a?.created_at ?? '');
    const tb = String(b?.created_at ?? '');
    return tb.localeCompare(ta);
  });
}

/**
 * Revert is only for accepted patches not already reverted.
 * @param {object} p
 * @returns {boolean}
 */
export function canRevertPatch(p) {
  const st = String(p?.status ?? '').toUpperCase();
  if (st !== 'ACCEPTED') return false;
  if (p?.reverted_at) return false;
  return true;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}

/**
 * @param {HTMLElement} container
 * @param {object[]} patches
 * @param {{ onRevert: (patchId: string) => void }} handlers
 */
export function renderPatchHistoryTable(container, patches, handlers) {
  container.replaceChildren();

  const ordered = sortPatchesNewestFirst(patches);

  if (!ordered.length) {
    const empty = document.createElement('p');
    empty.className = 'patch-history-empty';
    empty.textContent = 'No patch history for this dataset.';
    container.appendChild(empty);
    return;
  }

  const wrap = document.createElement('div');
  wrap.className = 'patch-history-scroll';

  const table = document.createElement('table');
  table.className = 'patch-history-table';

  const thead = document.createElement('thead');
  const hr = document.createElement('tr');
  for (const label of ['Column', 'Operation', 'Value', 'Status', 'Created', '']) {
    const th = document.createElement('th');
    th.textContent = label;
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');

  for (const p of ordered) {
    const tr = document.createElement('tr');
    tr.dataset.patchId = String(p.patch_id ?? '');

    const cells = [p.column, p.operation, p.value, p.status, p.created_at].map((val) => {
      const td = document.createElement('td');
      td.innerHTML = escapeHtml(val == null ? '' : String(val));
      return td;
    });

    cells.forEach((td) => tr.appendChild(td));

    const actionTd = document.createElement('td');
    if (canRevertPatch(p) && p.patch_id) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'btn-patch-revert';
      btn.textContent = 'Revert';
      btn.dataset.testid = 'patch-revert';
      btn.addEventListener('click', () => handlers.onRevert(String(p.patch_id)));
      actionTd.appendChild(btn);
    }
    tr.appendChild(actionTd);

    tbody.appendChild(tr);
  }

  table.appendChild(tbody);
  wrap.appendChild(table);
  container.appendChild(wrap);
}
