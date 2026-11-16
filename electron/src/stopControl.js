// SPDX-License-Identifier: MIT
/**
 * Stop, acknowledged at once (docs/app/DESIGN.md, "Stopped"). A question can be stopped before
 * its request has returned: the renderer shows "Stopped" immediately and stops waiting for the
 * request, whatever it later answers.
 */

/** What `race` resolves to when Stop came first. */
export const STOPPED = Symbol('stopped');

/**
 * @returns {{ readonly stopped: boolean, stop: () => boolean, onStop: (fn: () => void) => void,
 *   race: <T>(promise: Promise<T>) => Promise<T | typeof STOPPED> }}
 */
export function createStopController() {
  let stopped = false;
  /** @type {(() => void)[]} */
  const listeners = [];
  const controller = {
    get stopped() {
      return stopped;
    },
    /** @returns {boolean} true the first time only */
    stop() {
      if (stopped) return false;
      stopped = true;
      for (const fn of listeners.splice(0)) fn();
      return true;
    },
    /** @param {() => void} fn runs once, on Stop (at once if already stopped) */
    onStop(fn) {
      if (stopped) fn();
      else listeners.push(fn);
    },
    /**
     * The request's result, or STOPPED as soon as Stop is pressed (a later failure is ignored).
     * @template T
     * @param {Promise<T>} promise
     * @returns {Promise<T | typeof STOPPED>}
     */
    race(promise) {
      return new Promise((resolve, reject) => {
        controller.onStop(() => resolve(STOPPED));
        promise.then(resolve, (error) => {
          if (stopped) resolve(STOPPED);
          else reject(error);
        });
      });
    },
  };
  return controller;
}
