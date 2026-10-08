import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.js'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      reportsDirectory: './coverage',
      include: ['src/enrichmentPanel.js', 'src/patchHistoryPanel.js', 'src/resultPresentation.js', 'src/sessionSidebar.js', 'src/uploadPanel.js'],
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
