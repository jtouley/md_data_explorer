// SPDX-License-Identifier: MIT
/**
 * Theme choice shared by main, preload and renderer (no Node or DOM APIs here).
 *
 * Electron: main sets nativeTheme.themeSource, which drives prefers-color-scheme in the page, and
 * persists the choice; the page also mirrors it to data-theme so a plain browser behaves the same.
 */

export const THEME_SOURCES = Object.freeze(['system', 'light', 'dark']);
export const THEME_GET_CHANNEL = 'app:theme-get';
export const THEME_SET_CHANNEL = 'app:theme-set';
export const THEME_STORAGE_KEY = 'ca-theme';
/** main passes the saved choice to preload as this argument, so first paint follows it. */
export const THEME_ARG_PREFIX = '--ca-theme=';
export const WINDOW_BACKGROUND = Object.freeze({ light: '#F7F6F3', dark: '#161513' });

/**
 * @param {unknown} value
 * @returns {value is 'system' | 'light' | 'dark'}
 */
export function isThemeSource(value) {
  return typeof value === 'string' && THEME_SOURCES.includes(value);
}

/**
 * Read the theme from the settings file text. Anything unreadable means "system".
 * @param {string | null | undefined} text
 * @returns {'system' | 'light' | 'dark'}
 */
export function parseThemeSetting(text) {
  if (!text) return 'system';
  try {
    const data = JSON.parse(text);
    return isThemeSource(data?.theme) ? data.theme : 'system';
  } catch (error) {
    if (error instanceof SyntaxError) return 'system';
    throw error;
  }
}

/**
 * Settings file text with the theme replaced, keeping any other keys.
 * @param {string | null | undefined} previousText
 * @param {'system' | 'light' | 'dark'} theme
 * @returns {string}
 */
export function serializeThemeSetting(previousText, theme) {
  if (!isThemeSource(theme)) throw new Error(`Unknown theme: ${String(theme)}`);
  let data = {};
  if (previousText) {
    try {
      const parsed = JSON.parse(previousText);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) data = parsed;
    } catch (error) {
      if (!(error instanceof SyntaxError)) throw error;
    }
  }
  return `${JSON.stringify({ ...data, theme }, null, 2)}\n`;
}

/**
 * The saved theme main passed on the command line, or null when absent or invalid.
 * @param {readonly string[] | undefined} argv
 * @returns {'system' | 'light' | 'dark' | null}
 */
export function themeFromArgv(argv) {
  const arg = (argv || []).find((a) => typeof a === 'string' && a.startsWith(THEME_ARG_PREFIX));
  if (!arg) return null;
  const value = arg.slice(THEME_ARG_PREFIX.length);
  return isThemeSource(value) ? value : null;
}
