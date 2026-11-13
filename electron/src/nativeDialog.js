// SPDX-License-Identifier: MIT
/**
 * Native dialogs (main process) say one plain sentence. Paths, exit signals, log lines and
 * build commands are for whoever supports the app, so they sit behind "Copy details" instead of
 * on screen. Kept free of Electron imports so it can be unit-tested.
 */

export const COPY_DETAILS = 'Copy details';
export const COPIED_NOTE = 'Details copied. Paste them to whoever supports the app.';
export const COPY_FAILED_NOTE = 'Couldn’t copy the details. Restart the app and try again.';
/**
 * After this many copies the dialog comes back without Copy details, so it can only end on a
 * button the user pressed (never a default the code picks for them).
 */
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
 * Copy, then the dialog to show next: the same one with a note saying whether it worked, and
 * without the Copy details button once it has been used MAX_COPIES times.
 * @param {(text: string) => void} copy
 * @param {string} details
 * @param {{ detail?: string, buttons?: string[] }} base the caller's options
 * @param {object} shown the dialog just shown
 * @param {number} copies how many times Copy details has been pressed, this one included
 */
function afterCopy(copy, details, base, shown, copies) {
  let note = COPIED_NOTE;
  try {
    copy(details);
  } catch {
    note = COPY_FAILED_NOTE;
  }
  const detail = base.detail ? `${base.detail}\n\n${note}` : note;
  const buttons = copies >= MAX_COPIES ? [...(base.buttons || ['OK'])] : shown.buttons;
  return { ...shown, detail, buttons };
}

/**
 * Show a synchronous dialog until a button other than Copy details is chosen.
 * @param {(options: object) => number} show e.g. dialog.showMessageBoxSync
 * @param {(text: string) => void} copy e.g. clipboard.writeText
 * @param {{ options: object, details?: string }} spec
 * @returns {number} index of the chosen button in `spec.options.buttons`
 */
export function showWithCopyDetailsSync(show, copy, { options, details }) {
  let { options: shown, copyIndex } = withCopyButton(options, details);
  for (let copies = 1; ; copies += 1) {
    const response = show(shown);
    if (response !== copyIndex || !shown.buttons.includes(COPY_DETAILS)) return response;
    shown = afterCopy(copy, /** @type {string} */ (details), options, shown, copies);
  }
}

/**
 * Show an asynchronous dialog until a button other than Copy details is chosen.
 * @param {(options: object) => Promise<{ response: number }>} show e.g. dialog.showMessageBox
 * @param {(text: string) => void} copy
 * @param {{ options: object, details?: string }} spec
 * @returns {Promise<number>}
 */
export async function showWithCopyDetails(show, copy, { options, details }) {
  let { options: shown, copyIndex } = withCopyButton(options, details);
  for (let copies = 1; ; copies += 1) {
    const { response } = await show(shown);
    if (response !== copyIndex || !shown.buttons.includes(COPY_DETAILS)) return response;
    shown = afterCopy(copy, /** @type {string} */ (details), options, shown, copies);
  }
}
