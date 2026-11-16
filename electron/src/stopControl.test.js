// SPDX-License-Identifier: MIT
import { describe, expect, it, vi } from 'vitest';
import { STOPPED, createStopController } from './stopControl.js';

describe('createStopController (M2)', () => {
  it('race_stopBeforeTheRequestReturns_resolvesStoppedAtOnce', async () => {
    // Arrange: a submit that never answers
    const stop = createStopController();
    const pending = stop.race(new Promise(() => {}));

    // Act
    stop.stop();

    // Assert
    await expect(pending).resolves.toBe(STOPPED);
  });

  it('race_requestReturnsFirst_resolvesItsValue', async () => {
    // Arrange
    const stop = createStopController();

    // Act
    const value = await stop.race(Promise.resolve({ query_id: 'q1' }));

    // Assert
    expect(value).toEqual({ query_id: 'q1' });
    expect(stop.stopped).toBe(false);
  });

  it('race_requestFailsAfterStop_failureIsIgnored', async () => {
    // Arrange
    const stop = createStopController();
    let fail;
    const pending = stop.race(new Promise((_, reject) => (fail = reject)));

    // Act
    stop.stop();
    fail(new Error('AbortError'));

    // Assert
    await expect(pending).resolves.toBe(STOPPED);
  });

  it('onStop_runsListenersOnce_andImmediatelyWhenAlreadyStopped', () => {
    // Arrange
    const stop = createStopController();
    const first = vi.fn();
    const late = vi.fn();
    stop.onStop(first);

    // Act
    const did = stop.stop();
    const again = stop.stop();
    stop.onStop(late);

    // Assert
    expect(did).toBe(true);
    expect(again).toBe(false);
    expect(first).toHaveBeenCalledTimes(1);
    expect(late).toHaveBeenCalledTimes(1);
  });
});
