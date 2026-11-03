import { describe, expect, it } from 'vitest';
import {
  THEME_SOURCES,
  WINDOW_BACKGROUND,
  isThemeSource,
  parseThemeSetting,
  serializeThemeSetting,
  themeFromArgv,
} from './theme.js';

describe('isThemeSource', () => {
  it.each([
    ['system', true],
    ['light', true],
    ['dark', true],
    ['Dark', false],
    ['', false],
    [null, false],
    [undefined, false],
    [1, false],
  ])('isThemeSource_%s_is_%s', (value, expected) => {
    // Arrange / Act / Assert
    expect(isThemeSource(value)).toBe(expected);
  });

  it('themeSources_exactlySystemLightDark', () => {
    expect(THEME_SOURCES).toEqual(['system', 'light', 'dark']);
  });
});

describe('parseThemeSetting', () => {
  it.each([
    ['{"theme":"dark"}', 'dark'],
    ['{"theme":"light","other":1}', 'light'],
    ['{"theme":"neon"}', 'system'],
    ['{}', 'system'],
    ['not json', 'system'],
    ['', 'system'],
    [null, 'system'],
    ['[]', 'system'],
  ])('parseThemeSetting_%s_returns_%s', (text, expected) => {
    // Arrange / Act
    const result = parseThemeSetting(text);
    // Assert
    expect(result).toBe(expected);
  });
});

describe('serializeThemeSetting', () => {
  it('serializeThemeSetting_existingKeys_keptAndThemeReplaced', () => {
    // Arrange
    const previous = '{"theme":"light","windowWidth":1200}';
    // Act
    const text = serializeThemeSetting(previous, 'dark');
    // Assert
    expect(JSON.parse(text)).toEqual({ theme: 'dark', windowWidth: 1200 });
  });

  it.each([[null], [''], ['broken {'], ['[1,2]']])(
    'serializeThemeSetting_unusablePrevious_%s_startsFresh',
    (previous) => {
      // Arrange / Act
      const text = serializeThemeSetting(previous, 'system');
      // Assert
      expect(JSON.parse(text)).toEqual({ theme: 'system' });
    }
  );

  it('serializeThemeSetting_unknownTheme_throws', () => {
    expect(() => serializeThemeSetting(null, 'neon')).toThrow('Unknown theme');
  });

  it('windowBackground_matchesCanvasTokens', () => {
    expect(WINDOW_BACKGROUND).toEqual({ light: '#F7F6F3', dark: '#161513' });
  });
});

describe('themeFromArgv', () => {
  it.each([
    [['electron', '--ca-theme=dark'], 'dark'],
    [['--ca-theme=light'], 'light'],
    [['--ca-theme=system'], 'system'],
    [['--ca-theme=purple'], null],
    [['--ca-theme='], null],
    [['electron'], null],
    [[], null],
    [undefined, null],
  ])('themeFromArgv_case%#_returns_%s', (argv, expected) => {
    // Arrange / Act
    const result = themeFromArgv(argv);
    // Assert
    expect(result).toBe(expected);
  });
});
