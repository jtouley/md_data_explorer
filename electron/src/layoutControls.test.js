// SPDX-License-Identifier: MIT
/** @vitest-environment jsdom */
import { describe, expect, it, vi } from 'vitest';
import { bindBannerDismiss, bindSidebarToggle } from './layoutControls.js';

function sidebarSetup() {
  document.body.innerHTML = `
    <div class="app-container">
      <aside id="session-sidebar"><button id="inside">Inside</button></aside>
      <main><button id="sidebar-toggle" aria-controls="session-sidebar" aria-expanded="false">Cohort and conversations</button></main>
    </div>`;
  const container = document.querySelector('.app-container');
  const toggle = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('session-sidebar');
  const api = bindSidebarToggle({ container, toggle, sidebar });
  return { container, toggle, sidebar, api };
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
