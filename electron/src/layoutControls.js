// SPDX-License-Identifier: MIT
/**
 * Narrow windows (under 700 CSS px, e.g. the minimum window at 200% zoom): the sidebar folds
 * behind a toggle and the banners can be dismissed, so the composer always stays on screen.
 * The CSS decides when the toggle is visible; this module only keeps state, focus and ARIA right.
 */

const FOCUSABLE = 'button:not([disabled]), select:not([disabled]), [href], input:not([disabled])';

/**
 * @param {{ container: HTMLElement, toggle: HTMLButtonElement, sidebar: HTMLElement }} parts
 * @returns {{ open: () => void, close: (opts?: { focus?: boolean }) => void, isOpen: () => boolean }}
 */
export function bindSidebarToggle({ container, toggle, sidebar }) {
  const isOpen = () => container.classList.contains('sidebar-open');
  const open = () => {
    container.classList.add('sidebar-open');
    toggle.setAttribute('aria-expanded', 'true');
    /** @type {HTMLElement | null} */ (sidebar.querySelector(FOCUSABLE))?.focus();
  };
  const close = ({ focus = true } = {}) => {
    if (!isOpen()) return;
    container.classList.remove('sidebar-open');
    toggle.setAttribute('aria-expanded', 'false');
    if (focus) toggle.focus();
  };
  toggle.addEventListener('click', () => (isOpen() ? close() : open()));
  sidebar.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && isOpen()) {
      event.stopPropagation();
      close();
    }
  });
  return { open, close, isOpen };
}

/**
 * A banner's dismiss button hides it until the condition it reports changes.
 * @param {HTMLElement} banner
 * @param {{ onDismiss?: () => void, focusAfter?: HTMLElement | null }} [opts]
 */
export function bindBannerDismiss(banner, opts = {}) {
  const button = banner.querySelector('.banner-dismiss');
  if (!button) return;
  button.addEventListener('click', () => {
    banner.classList.add('hidden');
    opts.onDismiss?.();
    // The button has gone; focus moves on rather than falling to the page.
    if (opts.focusAfter && !(/** @type {HTMLTextAreaElement} */ (opts.focusAfter).disabled)) {
      opts.focusAfter.focus();
    }
  });
}
