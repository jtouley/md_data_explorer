/**
 * In-page confirmation for deleting a dataset (no window.confirm: the panel names exactly
 * what will be removed and keeps keyboard focus inside the page).
 */

/**
 * Render the confirmation panel into a container.
 * @param {HTMLElement} container
 * @param {{ datasetName: string, onConfirm: () => (void | Promise<void>), onCancel: () => void }} opts
 * @returns {{ setBusy: (busy: boolean) => void, setError: (message: string) => void }}
 */
export function renderDatasetDeleteConfirm(container, { datasetName, onConfirm, onCancel }) {
  container.replaceChildren();
  container.classList.remove('hidden');

  const panel = document.createElement('div');
  panel.className = 'dataset-delete-confirm';
  panel.setAttribute('role', 'alertdialog');
  panel.setAttribute('aria-labelledby', 'dataset-delete-title');
  panel.setAttribute('aria-describedby', 'dataset-delete-body');
  panel.dataset.testid = 'dataset-delete-confirm';

  const title = document.createElement('p');
  title.id = 'dataset-delete-title';
  title.className = 'dataset-delete-title';
  title.textContent = `Delete "${datasetName}" from this computer?`;

  const body = document.createElement('p');
  body.id = 'dataset-delete-body';
  body.className = 'dataset-delete-body';
  body.textContent =
    'This removes the uploaded file, its tables and exports, accepted enrichments, and every ' +
    'session and answer for this dataset. It cannot be undone.';

  const error = document.createElement('p');
  error.className = 'dataset-delete-error hidden';
  error.setAttribute('role', 'alert');

  const actions = document.createElement('div');
  actions.className = 'result-confirm-actions';

  const confirmBtn = document.createElement('button');
  confirmBtn.type = 'button';
  confirmBtn.className = 'btn-danger';
  confirmBtn.dataset.testid = 'dataset-delete-confirm-btn';
  confirmBtn.textContent = 'Delete permanently';

  const cancelBtn = document.createElement('button');
  cancelBtn.type = 'button';
  cancelBtn.className = 'btn-reject';
  cancelBtn.dataset.testid = 'dataset-delete-cancel-btn';
  cancelBtn.textContent = 'Cancel';

  const setBusy = (busy) => {
    confirmBtn.disabled = busy;
    cancelBtn.disabled = busy;
    confirmBtn.textContent = busy ? 'Deleting…' : 'Delete permanently';
  };
  const setError = (message) => {
    error.textContent = message;
    error.classList.toggle('hidden', !message);
  };

  confirmBtn.addEventListener('click', () => {
    setError('');
    void onConfirm();
  });
  cancelBtn.addEventListener('click', () => onCancel());
  panel.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !cancelBtn.disabled) onCancel();
  });

  actions.append(cancelBtn, confirmBtn);
  panel.append(title, body, error, actions);
  container.appendChild(panel);
  // Safe default: focus lands on Cancel, so a stray Enter does not delete.
  cancelBtn.focus();

  return { setBusy, setError };
}

/**
 * Hide and empty the confirmation container.
 * @param {HTMLElement} container
 */
export function closeDatasetDeleteConfirm(container) {
  container.replaceChildren();
  container.classList.add('hidden');
}
