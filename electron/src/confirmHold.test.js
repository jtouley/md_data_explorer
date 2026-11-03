import { describe, expect, it, vi } from 'vitest';
import { buildConfirmRequest, submitConfirmation } from './confirmHold.js';

const HELD = {
  pendingQuery: 'Compare LDL by treatment in older patients',
  planLine:
    'COMPARE_GROUPS · outcome ldl · grouping treatment · filters age > 65 · test Welch t-test',
  datasetId: 'ds_a',
  sessionId: 'sess_held',
};

describe('buildConfirmRequest', () => {
  it('buildConfirmRequest_validHold_bindsToHeldDatasetSessionAndExactPlan', () => {
    // Arrange
    const context = { currentDatasetId: 'ds_a', fallbackSessionId: 'sess_now' };
    // Act
    const result = buildConfirmRequest(HELD, context);
    // Assert
    expect(result).toEqual({
      ok: true,
      request: {
        sessionId: 'sess_held',
        datasetId: 'ds_a',
        queryText: HELD.pendingQuery,
        planLine: HELD.planLine,
      },
    });
  });

  it('buildConfirmRequest_holdWithoutBinding_usesCurrentDatasetAndSession', () => {
    // Arrange
    const held = { pendingQuery: 'q', planLine: 'DESCRIBE · outcome age' };
    // Act
    const result = buildConfirmRequest(held, { currentDatasetId: 'ds_b', fallbackSessionId: 's1' });
    // Assert
    expect(result.ok).toBe(true);
    expect(result.request).toMatchObject({ datasetId: 'ds_b', sessionId: 's1' });
  });

  it.each([
    [null, 'This plan can no longer be run. Ask the question again.'],
    [{ ...HELD, pendingQuery: '' }, 'This plan can no longer be run. Ask the question again.'],
    [{ ...HELD, planLine: '' }, 'This plan can no longer be run. Ask the question again.'],
    [
      { ...HELD, datasetId: 'ds_other' },
      'This plan was for a different cohort. Ask the question again on this one.',
    ],
    [{ pendingQuery: 'q', planLine: 'p' }, 'Choose a cohort, then ask the question again.'],
  ])('buildConfirmRequest_invalid_case%#_refusesWithMessage', (held, message) => {
    // Arrange
    const context = {
      currentDatasetId: held?.datasetId === 'ds_other' ? 'ds_a' : (held?.datasetId ?? null),
      fallbackSessionId: 's',
    };
    // Act
    const result = buildConfirmRequest(held, context);
    // Assert
    expect(result).toEqual({ ok: false, message });
  });
});

describe('submitConfirmation', () => {
  it('submitConfirmation_request_callsSubmitQueryConfirmedWithPlanLine', async () => {
    // Arrange
    const api = { submitQuery: vi.fn().mockResolvedValue({ query_id: 'q1' }) };
    const { request } = buildConfirmRequest(HELD, {
      currentDatasetId: 'ds_a',
      fallbackSessionId: 's',
    });
    // Act
    const response = await submitConfirmation(api, request);
    // Assert
    expect(api.submitQuery).toHaveBeenCalledWith(
      'sess_held',
      'ds_a',
      HELD.pendingQuery,
      true,
      HELD.planLine
    );
    expect(response).toEqual({ query_id: 'q1' });
  });
});
