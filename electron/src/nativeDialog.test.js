// SPDX-License-Identifier: MIT
import { describe, expect, it, vi } from 'vitest';
import { COPY_DETAILS, showWithCopyDetails, showWithCopyDetailsSync } from './nativeDialog.js';

describe('native dialogs keep developer output behind Copy details (M1)', () => {
  it('showWithCopyDetailsSync_copyThenQuit_copiesTheDetailsAndShowsAgain', () => {
    // Arrange: first click Copy details, then Quit
    const show = vi.fn().mockReturnValueOnce(1).mockReturnValueOnce(0);
    const copy = vi.fn();

    // Act
    const response = showWithCopyDetailsSync(show, copy, {
      options: { message: 'Plain sentence.', buttons: ['Quit'] },
      details: '/opt/app/models missing; run make fetch-embedding-model',
    });

    // Assert
    expect(response).toBe(0);
    expect(copy).toHaveBeenCalledWith('/opt/app/models missing; run make fetch-embedding-model');
    expect(show).toHaveBeenCalledTimes(2);
    expect(show.mock.calls[0][0].buttons).toEqual(['Quit', COPY_DETAILS]);
    expect(show.mock.calls[0][0].message).toBe('Plain sentence.');
    expect(JSON.stringify(show.mock.calls[0][0])).not.toMatch(/fetch-embedding-model|\/opt\//);
  });

  it('showWithCopyDetails_async_returnsTheFirstNonCopyAnswer', async () => {
    // Arrange: Copy details, then Restart engine
    const show = vi
      .fn()
      .mockResolvedValueOnce({ response: 2 })
      .mockResolvedValueOnce({ response: 0 });
    const copy = vi.fn();

    // Act
    const response = await showWithCopyDetails(show, copy, {
      options: { message: 'The analysis engine stopped.', buttons: ['Restart engine', 'Quit'] },
      details: 'signal SIGKILL',
    });

    // Assert
    expect(response).toBe(0);
    expect(copy).toHaveBeenCalledWith('signal SIGKILL');
    expect(show.mock.calls[0][0].buttons).toEqual(['Restart engine', 'Quit', COPY_DETAILS]);
  });

  it('showWithCopyDetailsSync_noDetails_noCopyButton', () => {
    // Arrange
    const show = vi.fn().mockReturnValue(0);

    // Act
    showWithCopyDetailsSync(show, vi.fn(), { options: { message: 'x', buttons: ['Quit'] } });

    // Assert
    expect(show.mock.calls[0][0].buttons).toEqual(['Quit']);
  });

  it('showWithCopyDetailsSync_copyKeepsBeingPressed_stopsAfterAFewRounds', () => {
    // Arrange
    const show = vi.fn().mockReturnValue(1);

    // Act
    const response = showWithCopyDetailsSync(show, vi.fn(), {
      options: { message: 'x', buttons: ['Quit'] },
      details: 'd',
    });

    // Assert
    expect(response).toBe(0);
    expect(show.mock.calls.length).toBeLessThanOrEqual(5);
  });
});
