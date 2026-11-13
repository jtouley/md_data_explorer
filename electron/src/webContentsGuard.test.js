// SPDX-License-Identifier: MIT
import { describe, expect, it, vi } from 'vitest';
import { lockDownWebContents } from './webContentsGuard.js';

/** A stand-in for WebContents that records listeners so tests can fire Electron's events. */
function fakeContents() {
  const listeners = new Map();
  return {
    on: (name, listener) => listeners.set(name, listener),
    setWindowOpenHandler: vi.fn(),
    emit: (name, ...args) => listeners.get(name)?.(...args),
    has: (name) => listeners.has(name),
  };
}

/** Electron 25+ passes one details object (an Event with url, isMainFrame, preventDefault). */
const frameNavigation = (url, isMainFrame = false) => ({ url, isMainFrame, preventDefault: vi.fn() });

describe('lockDownWebContents', () => {
  it.each([
    ['https://attacker.example/phish.html', true],
    ['http://127.0.0.1:8000/api/datasets', true],
    ['file:///etc/passwd', true],
    ['', true],
    ['app://clinical-analytics/index.html', false],
    ['about:blank', false],
    ['about:srcdoc', false],
  ])('lockDownWebContents_subframeNavigatesTo_%s_blocked_%s', (url, blocked) => {
    // Arrange
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const contents = fakeContents();
    lockDownWebContents(contents, undefined);
    const details = frameNavigation(url);
    // Act
    contents.emit('will-frame-navigate', details);
    // Assert
    expect(details.preventDefault).toHaveBeenCalledTimes(blocked ? 1 : 0);
  });

  it('lockDownWebContents_devServerSubframe_allowedOnlyForDevOrigin', () => {
    // Arrange
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const contents = fakeContents();
    lockDownWebContents(contents, 'http://localhost:5173');
    const same = frameNavigation('http://localhost:5173/index.html');
    const other = frameNavigation('http://localhost:3000/');
    // Act
    contents.emit('will-frame-navigate', same);
    contents.emit('will-frame-navigate', other);
    // Assert
    expect(same.preventDefault).not.toHaveBeenCalled();
    expect(other.preventDefault).toHaveBeenCalledTimes(1);
  });

  it('lockDownWebContents_mainFrameRemoteNavigation_blockedByWillNavigate', () => {
    // Arrange
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const contents = fakeContents();
    lockDownWebContents(contents, undefined);
    const event = { preventDefault: vi.fn() };
    // Act
    contents.emit('will-navigate', event, 'https://attacker.example/');
    // Assert
    expect(event.preventDefault).toHaveBeenCalledTimes(1);
    expect(contents.setWindowOpenHandler.mock.calls[0][0]()).toEqual({ action: 'deny' });
  });
});
