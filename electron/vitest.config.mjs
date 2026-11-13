import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.js'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      reportsDirectory: './coverage',
      include: [
        'src/answerCard.js',
        'src/appMenu.js',
        'src/backendSidecar.js',
        'src/startupWindow.js',
        'src/confirmHold.js',
        'src/enrichmentPanel.js',
        'src/icons.js',
        'src/layoutControls.js',
        'src/nativeDialog.js',
        'src/patchHistoryPanel.js',
        'src/resultPresentation.js',
        'src/sessionSidebar.js',
        'src/stopControl.js',
        'src/theme.js',
        'src/uploadPanel.js',
        'src/webContentsGuard.js',
      ],
      exclude: ['src/**/*.test.js'],
      thresholds: {
        lines: 70,
        functions: 70,
        statements: 70,
        branches: 65,
      },
    },
  },
});
