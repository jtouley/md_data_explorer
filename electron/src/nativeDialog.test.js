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
});

describe('native Copy details confirms and never answers for the user (round 2)', () => {
  it('showWithCopyDetailsSync_afterCopy_reShowsWithConfirmation', () => {
    // Arrange: Copy details, then Quit
    const show = vi.fn().mockReturnValueOnce(1).mockReturnValueOnce(0);

    // Act
    showWithCopyDetailsSync(show, vi.fn(), {
      options: { message: 'Plain sentence.', detail: 'Check the folder.', buttons: ['Quit'] },
      details: 'd',
    });

    // Assert
    expect(show.mock.calls[0][0].detail).toBe('Check the folder.');
    expect(show.mock.calls[1][0].detail).toBe(
      'Check the folder.\n\nDetails copied. Paste them to whoever supports the app.'
    );
    expect(show.mock.calls[1][0].message).toBe('Plain sentence.');
  });

  it('showWithCopyDetails_copyFails_saysSo', async () => {
    // Arrange
    const show = vi
      .fn()
      .mockResolvedValueOnce({ response: 1 })
      .mockResolvedValueOnce({ response: 0 });
    const copy = vi.fn(() => {
      throw new Error('clipboard unavailable');
    });

    // Act
    await showWithCopyDetails(show, copy, {
      options: { message: 'x', buttons: ['Quit'] },
      details: 'd',
    });

    // Assert
    expect(show.mock.calls[1][0].detail).toBe(
      'Couldn’t copy the details. Restart the app and try again.'
    );
  });

  it('showWithCopyDetails_crashDialogCopyPressedManyTimes_neverReturnsQuit', async () => {
    // Arrange: the crash dialog (Restart = 0, Quit = 1, cancelId 1). The user presses Copy
    // details over and over, then Restart engine once the button is gone.
    let calls = 0;
    const show = vi.fn(async (options) => {
      calls += 1;
      if (options.buttons.includes(COPY_DETAILS)) return { response: 2 };
      return { response: 0 };
    });

    // Act
    const response = await showWithCopyDetails(show, vi.fn(), {
      options: { message: 'Stopped.', buttons: ['Restart engine', 'Quit'], cancelId: 1 },
      details: 'signal SIGKILL',
    });

    // Assert: the answer is the button the user pressed, never the cancel button
    expect(response).toBe(0);
    expect(calls).toBe(4);
    const last = show.mock.calls[show.mock.calls.length - 1][0];
    expect(last.buttons).toEqual(['Restart engine', 'Quit']);
  });

  it('showWithCopyDetailsSync_copyKeepsBeingPressed_dropsTheButtonInsteadOfAnswering', () => {
    // Arrange
    const show = vi.fn((options) => (options.buttons.includes(COPY_DETAILS) ? 1 : 0));

    // Act
    const response = showWithCopyDetailsSync(show, vi.fn(), {
      options: { message: 'x', buttons: ['Quit'] },
      details: 'd',
    });

    // Assert
    expect(response).toBe(0);
    expect(show.mock.calls[show.mock.calls.length - 1][0].buttons).toEqual(['Quit']);
  });
});
