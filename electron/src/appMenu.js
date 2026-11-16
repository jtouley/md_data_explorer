// SPDX-License-Identifier: MIT
/**
 * The application menu: what a clinician needs (copy and paste, zoom, full screen, close), and
 * nothing that loses their work or exposes developer tools. Reload is left out on purpose: it
 * would drop the conversation on screen. Kept free of Electron imports so it can be unit-tested.
 */

/**
 * @param {{ isMac: boolean, isDev: boolean }} opts
 * @returns {object[]} a template for Menu.buildFromTemplate
 */
export function appMenuTemplate({ isMac, isDev }) {
  const view = [
    { role: 'resetZoom', label: 'Actual size' },
    { role: 'zoomIn' },
    { role: 'zoomOut' },
    { type: 'separator' },
    { role: 'togglefullscreen' },
    ...(isDev ? [{ type: 'separator' }, { role: 'toggleDevTools' }] : []),
  ];
  return [
    ...(isMac ? [{ role: 'appMenu' }] : []),
    { label: 'File', submenu: [isMac ? { role: 'close' } : { role: 'quit', label: 'Quit' }] },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'selectAll' },
      ],
    },
    { label: 'View', submenu: view },
    {
      label: 'Window',
      submenu: [{ role: 'minimize' }, ...(isMac ? [{ role: 'zoom' }] : [{ role: 'close' }])],
    },
  ];
}
