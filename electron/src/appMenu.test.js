// SPDX-License-Identifier: MIT
import { describe, expect, it } from 'vitest';
import { appMenuTemplate } from './appMenu.js';

/** Every role and label in a template, flattened. */
function entries(template) {
  return template.flatMap((menu) => [
    menu.label || menu.role,
    ...(menu.submenu || []).map((item) => item.role || item.label || item.type),
  ]);
}

describe('appMenuTemplate (L2)', () => {
  it.each([[{ isMac: false, isDev: false }], [{ isMac: true, isDev: false }]])(
    'appMenuTemplate_%o_hasNoDeveloperOrReloadItems',
    (opts) => {
      // Arrange / Act
      const shown = entries(appMenuTemplate(opts));

      // Assert
      expect(shown).not.toContain('toggleDevTools');
      expect(shown).not.toContain('reload');
      expect(shown).not.toContain('forceReload');
      expect(shown).toEqual(
        expect.arrayContaining(['copy', 'paste', 'zoomIn', 'zoomOut', 'resetZoom'])
      );
    }
  );

  it('appMenuTemplate_windows_topLevelMenusArePlain', () => {
    // Arrange / Act
    const labels = appMenuTemplate({ isMac: false, isDev: false }).map((m) => m.label);

    // Assert
    expect(labels).toEqual(['File', 'Edit', 'View', 'Window']);
  });

  it('appMenuTemplate_mac_startsWithTheAppMenu', () => {
    // Arrange / Act
    const [first] = appMenuTemplate({ isMac: true, isDev: false });

    // Assert
    expect(first.role).toBe('appMenu');
  });

  it('appMenuTemplate_dev_keepsDeveloperTools', () => {
    // Arrange / Act
    const shown = entries(appMenuTemplate({ isMac: false, isDev: true }));

    // Assert
    expect(shown).toContain('toggleDevTools');
    expect(shown).not.toContain('reload');
  });
});
