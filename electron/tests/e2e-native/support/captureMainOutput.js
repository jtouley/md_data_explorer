// SPDX-License-Identifier: MIT
/**
 * Test-only preload for the Electron main process (`electron -r <this file> ...`): copies every
 * write to process.stdout and process.stderr into the file named by CA_TEST_MAIN_OUTPUT, from the
 * first line on.
 *
 * Why: Playwright reads the app's stdout itself (readline in launchProcess), so the stream is
 * already flowing when electron.launch() resolves. Lines the app prints before the test attaches
 * its own 'data' listener reach only Playwright and are lost to the test. The app logs
 * "Backend starting at" during that window on Electron 45, so a listener-based check never saw it.
 * A file written synchronously by the main process itself has no such window.
 */
const fs = require('node:fs');

const file = process.env.CA_TEST_MAIN_OUTPUT;
if (file) {
  for (const stream of [process.stdout, process.stderr]) {
    const write = stream.write.bind(stream);
    stream.write = (chunk, ...rest) => {
      fs.appendFileSync(file, chunk);
      return write(chunk, ...rest);
    };
  }
}
