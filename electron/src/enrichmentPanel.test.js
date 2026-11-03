import { describe, expect, it, vi } from 'vitest';
import {
  createEnrichmentPanelGate,
  normalizePendingResponse,
  runEnrichmentDecision,
  suggestionToRows,
} from './enrichmentPanel.js';

describe('normalizePendingResponse', () => {
  it('returns empty suggestions for invalid payload', () => {
    expect(normalizePendingResponse(null)).toEqual({ suggestions: [], total: 0 });
    expect(normalizePendingResponse({})).toEqual({ suggestions: [], total: 0 });
  });

  it('uses API total when present', () => {
    const raw = {
      suggestions: [{ patch_id: 'a' }],
      total: 5,
    };
    expect(normalizePendingResponse(raw)).toEqual({
      suggestions: [{ patch_id: 'a' }],
      total: 5,
    });
  });

  it('falls back total to suggestions length', () => {
    const raw = {
      suggestions: [{ patch_id: 'x' }, { patch_id: 'y' }],
    };
    expect(normalizePendingResponse(raw).total).toBe(2);
  });
});

describe('suggestionToRows', () => {
  it('maps API fields to display rows', () => {
    const rows = suggestionToRows({
      operation: 'SET',
      column: 'age',
      suggested_value: 'numeric',
      current_value: null,
      confidence: 0.9,
      model_id: 'm1',
    });
    expect(rows).toContainEqual(['Operation', 'SET']);
    expect(rows).toContainEqual(['Column', 'age']);
    expect(rows).toContainEqual(['Suggested', 'numeric']);
    expect(rows).toContainEqual(['Current', '—']);
    expect(rows).toContainEqual(['Confidence', '0.9']);
    expect(rows).toContainEqual(['Model', 'm1']);
  });

  it('shows current value when set', () => {
    const rows = suggestionToRows({
      operation: 'X',
      column: 'c',
      suggested_value: 's',
      current_value: 'old',
      confidence: 0,
      model_id: 'z',
    });
    expect(rows).toContainEqual(['Current', 'old']);
  });
});

describe('runEnrichmentDecision', () => {
  function harness() {
    const events = [];
    return {
      events,
      opts: {
        reload: vi.fn(async () => {
          events.push('reload');
        }),
        showMessage: vi.fn((text) => {
          events.push(`message:${text}`);
        }),
        fallback: 'Accept failed',
      },
    };
  }

  it('reloads the panel after a successful decision', async () => {
    const { events, opts } = harness();
    await runEnrichmentDecision(async () => ({ success: true, message: 'ok' }), opts);
    expect(events).toEqual(['reload']);
  });

  it('reloads and then explains when the patch was resolved elsewhere (404/409)', async () => {
    const { events, opts } = harness();
    await runEnrichmentDecision(
      async () => ({
        success: false,
        stale: true,
        message: 'Only a pending patch can be accepted',
      }),
      opts
    );
    expect(events).toEqual(['reload', 'message:Only a pending patch can be accepted']);
  });

  it('shows the error without reloading when the call throws', async () => {
    const { events, opts } = harness();
    vi.spyOn(console, 'error').mockImplementation(() => {});
    await runEnrichmentDecision(async () => {
      throw new Error('Could not update the enrichment overlay');
    }, opts);
    expect(events).toEqual(['message:Could not update the enrichment overlay']);
  });

  it('treats success false as a failure without reloading', async () => {
    const { events, opts } = harness();
    vi.spyOn(console, 'error').mockImplementation(() => {});
    await runEnrichmentDecision(async () => ({ success: false }), opts);
    expect(events).toEqual(['message:Accept failed']);
  });
});

describe('runEnrichmentDecision overlapping calls', () => {
  it('runEnrichmentDecision_doubleClickSlowSuccessThenStale_keepsTheStaleMessage', async () => {
    // Arrange: one shared hint, like loadEnrichmentPanel writing "No pending suggestions." when a
    // reload finishes. The first click's reload is slow; the server rejects the second click (409).
    let hint = '';
    let releaseFirstReload;
    const firstReloadGate = new Promise((resolve) => {
      releaseFirstReload = resolve;
    });
    const reloadSlow = async () => {
      await firstReloadGate;
      hint = 'No pending suggestions.';
    };
    const reloadFast = async () => {
      hint = 'No pending suggestions.';
    };
    const show = (text) => {
      hint = text;
    };

    // Act: second click starts before the first click's reload has finished
    const first = runEnrichmentDecision(async () => ({ success: true }), {
      reload: reloadSlow,
      showMessage: show,
      fallback: 'Accept failed',
    });
    const second = runEnrichmentDecision(
      async () => ({
        success: false,
        stale: true,
        message: 'Only a pending patch can be accepted',
      }),
      { reload: reloadFast, showMessage: show, fallback: 'Accept failed' }
    );
    // Let the second click run as far as it can (unqueued, it reaches its 409 message here)
    // before the first click's slow reload lands, as in the reviewer's race.
    await new Promise((resolve) => setTimeout(resolve, 20));
    releaseFirstReload();
    await Promise.all([first, second]);

    // Assert: the 409 explanation is the last thing the user sees, not the idle text
    expect(hint).toBe('Only a pending patch can be accepted');
  });

  it('runEnrichmentDecision_queuedCall_startsOnlyAfterEarlierReload', async () => {
    // Arrange
    const events = [];
    let releaseFirstReload;
    const gate = new Promise((resolve) => {
      releaseFirstReload = resolve;
    });

    // Act
    const first = runEnrichmentDecision(
      async () => {
        events.push('call-1');
        return { success: true };
      },
      {
        reload: async () => {
          await gate;
          events.push('reload-1');
        },
        showMessage: () => events.push('message-1'),
        fallback: 'x',
      }
    );
    const second = runEnrichmentDecision(
      async () => {
        events.push('call-2');
        return { success: true };
      },
      { reload: async () => events.push('reload-2'), showMessage: () => {}, fallback: 'x' }
    );
    await Promise.resolve();
    const beforeRelease = [...events];
    releaseFirstReload();
    await Promise.all([first, second]);

    // Assert: the second click is queued, not dropped, and not interleaved
    expect(beforeRelease).toEqual(['call-1']);
    expect(events).toEqual(['call-1', 'reload-1', 'call-2', 'reload-2']);
  });
});

describe('runEnrichmentDecision queue robustness', () => {
  it('runEnrichmentDecision_showMessageThrows_laterDecisionStillRuns', async () => {
    // Arrange
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const events = [];
    const broken = runEnrichmentDecision(
      async () => {
        throw new Error('server down');
      },
      {
        reload: async () => {},
        showMessage: () => {
          throw new Error('hint dead');
        },
        fallback: 'x',
      }
    );

    // Act
    const later = runEnrichmentDecision(
      async () => {
        events.push('call-2');
        return { success: true };
      },
      { reload: async () => events.push('reload-2'), showMessage: () => {}, fallback: 'x' }
    );

    // Assert: the caller still sees the failure, but the queue keeps working
    await expect(broken).rejects.toThrow('hint dead');
    await later;
    expect(events).toEqual(['call-2', 'reload-2']);
  });

  it.each([
    [{ stale: true, message: 'Only a pending patch can be accepted' }],
    [{ success: false, message: 'Accept failed on server' }],
    [{ success: true }],
  ])('runEnrichmentDecision_datasetChangedBeforeRun_noReloadOrMessage_case%#', async (res) => {
    // Arrange: the user switches dataset while the decision is queued
    vi.spyOn(console, 'error').mockImplementation(() => {});
    let current = true;
    const events = [];
    let release;
    const gate = new Promise((resolve) => {
      release = resolve;
    });
    const first = runEnrichmentDecision(
      async () => {
        await gate;
        return { success: true };
      },
      { reload: async () => {}, showMessage: () => {}, fallback: 'x' }
    );
    const queued = runEnrichmentDecision(
      async () => {
        events.push('call');
        return res;
      },
      {
        reload: async () => events.push('reload'),
        showMessage: (t) => events.push(`message:${t}`),
        isCurrent: () => current,
        fallback: 'x',
      }
    );

    // Act
    current = false;
    release();
    await Promise.all([first, queued]);

    // Assert: the decision is still sent, but nothing lands on the other dataset's panel
    expect(events).toEqual(['call']);
  });
});

describe('createEnrichmentPanelGate', () => {
  it('gate_refreshStartedBeforeMessage_doesNotOverwriteThatHint', () => {
    // Arrange: decision reload, then a refresh starts, then the decision shows its 409
    const gate = createEnrichmentPanelGate();
    const decisionReload = gate.startLoad();
    const refresh = gate.startLoad();

    // Act
    gate.noteMessage('suggestions');

    // Assert
    expect(gate.isLatestLoad(decisionReload)).toBe(false);
    expect(gate.isLatestLoad(refresh)).toBe(true);
    expect(gate.mayWriteHint(refresh, 'suggestions')).toBe(false);
    expect(gate.mayWriteHint(refresh, 'history')).toBe(true);
  });

  it('gate_loadStartedAfterMessage_writesHint', () => {
    // Arrange
    const gate = createEnrichmentPanelGate();
    gate.startLoad();
    gate.noteMessage('suggestions');

    // Act
    const later = gate.startLoad();

    // Assert: a refresh the user starts after reading the message shows fresh text
    expect(gate.mayWriteHint(later, 'suggestions')).toBe(true);
  });

  it('gate_noMessages_latestLoadWritesBothHints', () => {
    // Arrange / Act
    const gate = createEnrichmentPanelGate();
    const load = gate.startLoad();

    // Assert
    expect(gate.mayWriteHint(load, 'suggestions')).toBe(true);
    expect(gate.mayWriteHint(load, 'history')).toBe(true);
  });
});
