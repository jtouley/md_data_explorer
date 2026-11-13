// SPDX-License-Identifier: MIT
/**
 * Native dialogs (main process) say one plain sentence. Paths, exit signals, log lines and
 * build commands are for whoever supports the app, so they sit behind "Copy details" instead of
 * on screen. Kept free of Electron imports so it can be unit-tested.
 */

export const COPY_DETAILS = 'Copy details';
/** A person pressing Copy details again and again still gets the dialog back, a few times. */
const MAX_COPIES = 3;

/**
 * @param {{ buttons?: string[] }} options
 * @param {string | undefined} details
 */
function withCopyButton(options, details) {
  const buttons = [...(options.buttons || ['OK'])];
  if (!details) return { options: { ...options, buttons }, copyIndex: -1 };
  return {
    options: { ...options, buttons: [...buttons, COPY_DETAILS] },
    copyIndex: buttons.length,
  };
}

/**
 * Show a synchronous dialog until a button other than Copy details is chosen.
 * @param {(options: object) => number} show e.g. dialog.showMessageBoxSync
 * @param {(text: string) => void} copy e.g. clipboard.writeText
 * @param {{ options: object, details?: string }} spec
 * @returns {number} index of the chosen button in `spec.options.buttons`
 */
export function showWithCopyDetailsSync(show, copy, { options, details }) {
  const { options: shown, copyIndex } = withCopyButton(options, details);
  for (let round = 0; round <= MAX_COPIES; round += 1) {
    const response = show(shown);
    if (response !== copyIndex) return response;
    copy(/** @type {string} */ (details));
  }
  return shown.cancelId ?? 0;
}

/**
 * Show an asynchronous dialog until a button other than Copy details is chosen.
 * @param {(options: object) => Promise<{ response: number }>} show e.g. dialog.showMessageBox
 * @param {(text: string) => void} copy
 * @param {{ options: object, details?: string }} spec
 * @returns {Promise<number>}
 */
export async function showWithCopyDetails(show, copy, { options, details }) {
  const { options: shown, copyIndex } = withCopyButton(options, details);
  for (let round = 0; round <= MAX_COPIES; round += 1) {
    const { response } = await show(shown);
    if (response !== copyIndex) return response;
    copy(/** @type {string} */ (details));
  }
  return shown.cancelId ?? 0;
}
