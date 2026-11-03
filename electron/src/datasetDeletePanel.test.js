/** @vitest-environment jsdom */
import { describe, test, expect, vi } from 'vitest';
import { closeDatasetDeleteConfirm, renderDatasetDeleteConfirm } from './datasetDeletePanel.js';

function setup() {
  const container = document.createElement('div');
  container.className = 'hidden';
  document.body.appendChild(container);
  const onConfirm = vi.fn();
  const onCancel = vi.fn();
  const api = renderDatasetDeleteConfirm(container, {
    datasetName: 'Ward 7 cohort',
    onConfirm,
    onCancel,
  });
  const q = (id) => container.querySelector(`[data-testid="${id}"]`);
  return { container, onConfirm, onCancel, api, q };
}

describe('renderDatasetDeleteConfirm', () => {
  test('shows the dataset name and focuses Cancel, not Delete', () => {
    // Arrange / Act
    const { container, q } = setup();

    // Assert
    expect(container.classList.contains('hidden')).toBe(false);
    expect(container.textContent).toContain('Ward 7 cohort');
    expect(document.activeElement).toBe(q('dataset-delete-cancel-btn'));
  });

  test('deletes only when the danger button is clicked', () => {
    // Arrange
    const { onConfirm, onCancel, q } = setup();

    // Act
    q('dataset-delete-confirm-btn').click();

    // Assert
    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(onCancel).not.toHaveBeenCalled();
  });

  test.each([
    ['cancel button', (q) => q('dataset-delete-cancel-btn').click()],
    [
      'Escape key',
      (q) =>
        q('dataset-delete-confirm').dispatchEvent(
          new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })
        ),
    ],
  ])('%s cancels without deleting', (_name, act) => {
    // Arrange
    const { onConfirm, onCancel, q } = setup();

    // Act
    act(q);

    // Assert
    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onConfirm).not.toHaveBeenCalled();
  });

  test('busy state disables both buttons and errors are announced', () => {
    // Arrange
    const { api, q, container } = setup();

    // Act
    api.setBusy(true);
    api.setError('Delete failed: locked');

    // Assert
    expect(q('dataset-delete-confirm-btn').disabled).toBe(true);
    expect(q('dataset-delete-cancel-btn').disabled).toBe(true);
    const alert = container.querySelector('[role="alert"]');
    expect(alert.textContent).toBe('Delete failed: locked');
    expect(alert.classList.contains('hidden')).toBe(false);
  });
});

describe('closeDatasetDeleteConfirm', () => {
  test('empties and hides the container', () => {
    // Arrange
    const { container } = setup();

    // Act
    closeDatasetDeleteConfirm(container);

    // Assert
    expect(container.children.length).toBe(0);
    expect(container.classList.contains('hidden')).toBe(true);
  });
});
