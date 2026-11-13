// SPDX-License-Identifier: MIT
/**
 * Narrow windows (under 700 CSS px, e.g. the minimum window at 200% zoom): the sidebar folds
 * behind a toggle and the banners can be dismissed, so the composer always stays on screen.
 * The CSS decides when the toggle is visible; this module only keeps state, focus and ARIA right.
 */

const FOCUSABLE = 'button:not([disabled]), select:not([disabled]), [href], input:not([disabled])';

/**
 * The open drawer is not modal, so it never traps focus. It closes, and never strands anyone,
 * when its own close button is pressed (focus returns to the toggle), on Escape inside it or on
 * the toggle, on a pointer press outside it, and when focus moves out of it (focus stays where
 * it went).
 * @param {{ container: HTMLElement, toggle: HTMLButtonElement, sidebar: HTMLElement,
 *   closeButton?: HTMLElement | null }} parts
 * @returns {{ open: () => void, close: (opts?: { focus?: boolean }) => void, isOpen: () => boolean }}
 */
export function bindSidebarToggle({ container, toggle, sidebar, closeButton = null }) {
  const isOpen = () => container.classList.contains('sidebar-open');
  const open = () => {
    container.classList.add('sidebar-open');
    toggle.setAttribute('aria-expanded', 'true');
    // The first control the user came for, not the close button.
    const first = [...sidebar.querySelectorAll(FOCUSABLE)].find((node) => node !== closeButton);
    /** @type {HTMLElement | undefined} */ (first)?.focus();
  };
  const close = ({ focus = true } = {}) => {
    if (!isOpen()) return;
    container.classList.remove('sidebar-open');
    toggle.setAttribute('aria-expanded', 'false');
    if (focus) toggle.focus();
  };
  /** @param {EventTarget | null} target */
  const outside = (target) =>
    target instanceof Node && !sidebar.contains(target) && !toggle.contains(target);

  toggle.addEventListener('click', () => (isOpen() ? close() : open()));
  closeButton?.addEventListener('click', () => close());
  const onEscape = (/** @type {KeyboardEvent} */ event) => {
    if (event.key === 'Escape' && isOpen()) {
      event.stopPropagation();
      close();
    }
  };
  sidebar.addEventListener('keydown', onEscape);
  toggle.addEventListener('keydown', onEscape);
  const doc = container.ownerDocument;
  doc.addEventListener('pointerdown', (event) => {
    if (isOpen() && outside(event.target)) close({ focus: false });
  });
  doc.addEventListener('focusin', (event) => {
    if (isOpen() && outside(event.target)) close({ focus: false });
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
