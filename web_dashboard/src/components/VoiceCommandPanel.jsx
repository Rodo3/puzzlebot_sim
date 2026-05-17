import React from 'react';

const MAX_HISTORY = 8;

export default function VoiceCommandPanel({ voiceData, history }) {
  if (!voiceData) {
    return (
      <div className="panel">
        <h3>Voice Commands</h3>
        <p className="muted">Waiting for data…</p>
      </div>
    );
  }

  const confidencePct = voiceData.confidence != null
    ? `${(voiceData.confidence * 100).toFixed(1)}%`
    : '—';

  return (
    <div className="panel">
      <h3>Voice Commands</h3>
      <div className="voice-main">
        <span className="voice-command">{voiceData.command ?? '—'}</span>
        <span className="muted">confidence: {confidencePct}</span>
        {voiceData.status && (
          <span className="badge badge-info">{voiceData.status}</span>
        )}
        {voiceData.inference_time_ms != null && (
          <span className="muted">inference: {voiceData.inference_time_ms.toFixed(2)} ms</span>
        )}
      </div>

      {voiceData.ranked_predictions?.length > 0 && (
        <div className="ranked-list">
          {voiceData.ranked_predictions.slice(0, 3).map((p, i) => (
            <div key={i} className="ranked-row">
              <span>{p.command}</span>
              <span className="muted">{(p.score * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}

      {history.length > 0 && (
        <>
          <hr />
          <div className="voice-history">
            <span className="muted small">Recent:</span>
            {history.slice(-MAX_HISTORY).reverse().map((h, i) => (
              <span key={i} className="history-item">{h}</span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
