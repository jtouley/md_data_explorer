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
        'src/confirmHold.js',
        'src/enrichmentPanel.js',
        'src/icons.js',
        'src/patchHistoryPanel.js',
        'src/resultPresentation.js',
        'src/sessionSidebar.js',
        'src/theme.js',
        'src/uploadPanel.js',
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
