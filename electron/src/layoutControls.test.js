// SPDX-License-Identifier: MIT
/** @vitest-environment jsdom */
import { describe, expect, it, vi } from 'vitest';
import { bindBannerDismiss, bindSidebarToggle } from './layoutControls.js';

function sidebarSetup() {
  document.body.innerHTML = `
    <div class="app-container">
      <aside id="session-sidebar">
        <button id="sidebar-close" class="sidebar-close" aria-label="Close cohort and conversations"></button>
        <button id="inside">Inside</button>
      </aside>
      <main><button id="sidebar-toggle" aria-controls="session-sidebar" aria-expanded="false">Cohort and conversations</button>
        <textarea id="composer"></textarea></main>
    </div>`;
  const container = document.querySelector('.app-container');
  const toggle = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('session-sidebar');
  const closeButton = document.getElementById('sidebar-close');
  const api = bindSidebarToggle({ container, toggle, sidebar, closeButton });
  return { container, toggle, sidebar, closeButton, api };
}

describe('bindSidebarToggle (H6)', () => {
  it('toggle_opensTheSidebar_andMovesFocusIntoIt', () => {
    // Arrange
    const { container, toggle } = sidebarSetup();

    // Act
    toggle.click();

    // Assert
    expect(container.classList.contains('sidebar-open')).toBe(true);
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    expect(document.activeElement.id).toBe('inside');
  });

  it('escape_closesTheSidebar_andReturnsFocusToTheToggle', () => {
    // Arrange
    const { container, toggle, sidebar } = sidebarSetup();
    toggle.click();

    // Act
    sidebar.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

    // Assert
    expect(container.classList.contains('sidebar-open')).toBe(false);
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(toggle);
  });

  it('close_afterAChoiceInTheSidebar_closesWithoutStealingFocus', () => {
    // Arrange
    const { container, toggle, api } = sidebarSetup();
    toggle.click();

    // Act
    api.close({ focus: false });

    // Assert
    expect(container.classList.contains('sidebar-open')).toBe(false);
    expect(document.activeElement).not.toBe(toggle);
  });
});

describe('bindSidebarToggle: the drawer can always be closed (round 2)', () => {
  it('closeButton_closesTheDrawer_andReturnsFocusToTheToggle', () => {
    // Arrange
    const { container, toggle, closeButton } = sidebarSetup();
    toggle.click();

    // Act
    closeButton.click();

    // Assert
    expect(container.classList.contains('sidebar-open')).toBe(false);
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(toggle);
  });

  it('open_movesFocusToTheFirstControl_notTheCloseButton', () => {
    // Arrange
    const { toggle } = sidebarSetup();

    // Act
    toggle.click();

    // Assert
    expect(document.activeElement.id).toBe('inside');
  });

  it('pointerDownOutside_closesTheDrawer_withoutMovingFocusBack', () => {
    // Arrange
    const { container, toggle } = sidebarSetup();
    toggle.click();
    const composer = document.getElementById('composer');

    // Act
    composer.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));

    // Assert
    expect(container.classList.contains('sidebar-open')).toBe(false);
    expect(document.activeElement).not.toBe(toggle);
  });

  it.each([['inside'], ['sidebar-toggle']])(
    'pointerDownOn_%s_leavesTheDrawerToItsOwnHandler',
    (id) => {
      // Arrange
      const { container, toggle } = sidebarSetup();
      toggle.click();

      // Act
      document.getElementById(id).dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));

      // Assert
      expect(container.classList.contains('sidebar-open')).toBe(true);
    }
  );

  it('focusMovingOutOfTheDrawer_closesIt_andLeavesFocusWhereItWent', () => {
    // Arrange
    const { container, toggle } = sidebarSetup();
    toggle.click();
    const composer = document.getElementById('composer');

    // Act: Tab past the last control in the drawer
    composer.focus();

    // Assert
    expect(container.classList.contains('sidebar-open')).toBe(false);
    expect(document.activeElement).toBe(composer);
  });

  it('escapeOnTheToggle_closesTheOpenDrawer', () => {
    // Arrange
    const { container, toggle } = sidebarSetup();
    toggle.click();
    toggle.focus();

    // Act
    toggle.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

    // Assert
    expect(container.classList.contains('sidebar-open')).toBe(false);
    expect(document.activeElement).toBe(toggle);
  });
});

describe('bindBannerDismiss (H6)', () => {
  it('dismiss_hidesTheBanner_andReportsIt', () => {
    // Arrange
    document.body.innerHTML = `<div id="b" class="banner"><span>Text</span><button class="banner-dismiss" aria-label="Dismiss this message">x</button></div><textarea id="q"></textarea>`;
    const banner = document.getElementById('b');
    const onDismiss = vi.fn();
    bindBannerDismiss(banner, { onDismiss, focusAfter: document.getElementById('q') });

    // Act
    banner.querySelector('.banner-dismiss').click();

    // Assert
    expect(banner.classList.contains('hidden')).toBe(true);
    expect(onDismiss).toHaveBeenCalledTimes(1);
    expect(document.activeElement.id).toBe('q');
  });
});
