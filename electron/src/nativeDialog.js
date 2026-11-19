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
 * The dialog to show after a copy: the same one with a note saying whether it worked, and
 * without the Copy details button once it has been used MAX_COPIES times.
 * @param {boolean} copied
 * @param {{ detail?: string, buttons?: string[] }} base the caller's options
 * @param {object} shown the dialog just shown
 * @param {number} copies how many times Copy details has been pressed, this one included
 */
function afterCopy(copied, base, shown, copies) {
  const note = copied ? COPIED_NOTE : COPY_FAILED_NOTE;
  const detail = base.detail ? `${base.detail}\n\n${note}` : note;
  const buttons = copies >= MAX_COPIES ? [...(base.buttons || ['OK'])] : shown.buttons;
  return { ...shown, detail, buttons };
}

/**
 * Copy without waiting, for the synchronous dialog. Electron 44+ clipboard.writeText returns a
 * Promise; a rejection cannot be reported in a dialog that is already showing again, so it is
 * logged instead of left unhandled.
 * @param {(text: string) => void | Promise<void>} copy
 * @param {string} details
 * @returns {boolean} false if the copy failed synchronously
 */
function copyNow(copy, details) {
  try {
    const pending = copy(details);
    if (pending && typeof pending.then === 'function') {
      pending.then(undefined, (error) =>
        console.error('Could not copy the details:', error?.message || error)
      );
    }
    return true;
  } catch {
    return false;
  }
}

/**
 * Copy and wait for the result, for the asynchronous dialog (clipboard.writeText is async in
 * Electron 44+, so a failure arrives as a rejection, not a throw).
 * @param {(text: string) => void | Promise<void>} copy
 * @param {string} details
 * @returns {Promise<boolean>} whether the copy worked
 */
async function copyAndWait(copy, details) {
  try {
    await copy(details);
    return true;
  } catch {
    return false;
  }
}

/**
 * Show a synchronous dialog until a button other than Copy details is chosen.
 * @param {(options: object) => number} show e.g. dialog.showMessageBoxSync
 * @param {(text: string) => void | Promise<void>} copy e.g. clipboard.writeText
 * @param {{ options: object, details?: string }} spec
 * @returns {number} index of the chosen button in `spec.options.buttons`
 */
export function showWithCopyDetailsSync(show, copy, { options, details }) {
  let { options: shown, copyIndex } = withCopyButton(options, details);
  for (let copies = 1; ; copies += 1) {
    const response = show(shown);
    if (response !== copyIndex || !shown.buttons.includes(COPY_DETAILS)) return response;
    const copied = copyNow(copy, /** @type {string} */ (details));
    shown = afterCopy(copied, options, shown, copies);
  }
}

/**
 * Show an asynchronous dialog until a button other than Copy details is chosen.
 * @param {(options: object) => Promise<{ response: number }>} show e.g. dialog.showMessageBox
 * @param {(text: string) => void | Promise<void>} copy e.g. clipboard.writeText
 * @param {{ options: object, details?: string }} spec
 * @returns {Promise<number>}
 */
export async function showWithCopyDetails(show, copy, { options, details }) {
  let { options: shown, copyIndex } = withCopyButton(options, details);
  for (let copies = 1; ; copies += 1) {
    const { response } = await show(shown);
    if (response !== copyIndex || !shown.buttons.includes(COPY_DETAILS)) return response;
    const copied = await copyAndWait(copy, /** @type {string} */ (details));
    shown = afterCopy(copied, options, shown, copies);
  }
}
