import React, { useState, useRef, useCallback } from 'react';

const MAX_HISTORY = 8;
const RECORD_DURATION_S = 1.5;

function getAudioUrl() {
  const wsUrl = import.meta.env.VITE_WS_URL ?? `ws://${window.location.hostname}:8000/ws`;
  return wsUrl.replace(/^ws(s?):/, 'http$1:').replace(/\/ws$/, '/audio');
}

function encodeWAV(samples, sampleRate) {
  const dataLen = samples.length * 2;
  const buf = new ArrayBuffer(44 + dataLen);
  const v = new DataView(buf);
  const str = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };

  str(0,  'RIFF'); v.setUint32(4,  36 + dataLen, true);
  str(8,  'WAVE'); str(12, 'fmt '); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, dataLen, true);

  let off = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(off, s < 0 ? s * 32768 : s * 32767, true);
    off += 2;
  }
  return buf;
}

function recLabel(state) {
  return { idle: 'Escuchar', recording: 'Grabando…', uploading: 'Procesando…', error: 'Reintentar' }[state];
}

// Renders a top-3 list for one model.
// For KMeans: lower dist = better. For HMM: higher log_lik = better.
function ModelPredictions({ label, rows, winner, scoreLabel }) {
  if (!rows || rows.length === 0) return null;
  return (
    <div className="model-block">
      <div className="model-header">
        <span className="model-name">{label}</span>
        <span className="muted small">{scoreLabel}</span>
      </div>
      {rows.slice(0, 3).map(([cmd, score], i) => (
        <div key={i} className={`ranked-row ${i === 0 && cmd === winner ? 'ranked-winner' : ''}`}>
          <span>{i === 0 ? '▶ ' : '  '}{cmd}</span>
          <span className="muted">
            {typeof score === 'number' ? score.toFixed(4) : score}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function VoiceCommandPanel({ voiceData, history }) {
  const [recState, setRecState] = useState('idle');
  const [recError, setRecError] = useState(null);
  const samplesRef = useRef([]);

  const startRecording = useCallback(async () => {
    setRecError(null);
    setRecState('recording');
    samplesRef.current = [];

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          channelCount: 1,
        },
      });
    } catch (e) {
      setRecError(`Mic: ${e.message}`);
      setRecState('error');
      return;
    }

    const ctx = new AudioContext();
    const source = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) => {
      samplesRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };
    source.connect(processor);
    processor.connect(ctx.destination);

    setTimeout(async () => {
      processor.disconnect();
      source.disconnect();
      stream.getTracks().forEach(t => t.stop());

      const chunks = samplesRef.current;
      const totalLen = chunks.reduce((acc, c) => acc + c.length, 0);
      const merged = new Float32Array(totalLen);
      let off = 0;
      for (const c of chunks) { merged.set(c, off); off += c.length; }

      const sampleRate = ctx.sampleRate;
      ctx.close();
      console.log(`[Voice] sampleRate=${sampleRate} samples=${totalLen} duration=${(totalLen/sampleRate).toFixed(2)}s`);
      setRecState('uploading');

      const wav = encodeWAV(merged, sampleRate);
      try {
        const res = await fetch(getAudioUrl(), {
          method: 'POST',
          headers: { 'Content-Type': 'audio/wav' },
          body: wav,
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          setRecError(err.error ?? `HTTP ${res.status}`);
          setRecState('error');
        } else {
          setRecState('idle');
        }
      } catch (e) {
        setRecError(`Conexión: ${e.message}`);
        setRecState('error');
      }
    }, RECORD_DURATION_S * 1000);
  }, []);

  const busy = recState === 'recording' || recState === 'uploading';

  // ranked_predictions arrives as { kmeans: [[label, dist], ...], hmm: [[label, ll], ...] }
  const ranked = voiceData?.ranked_predictions;
  const kmeansRows = Array.isArray(ranked?.kmeans) ? ranked.kmeans : null;
  const hmmRows   = Array.isArray(ranked?.hmm)    ? ranked.hmm    : null;

  return (
    <div className="panel">
      <h3>Voice Commands</h3>

      {/* Recording button */}
      <div className="voice-trigger">
        <button
          className={`btn-listen ${recState}`}
          onClick={startRecording}
          disabled={busy}
        >
          {recLabel(recState)}
        </button>
        {recState === 'recording' && (
          <span className="muted small">{RECORD_DURATION_S}s…</span>
        )}
        {recError && (
          <span className="badge badge-error small">{recError}</span>
        )}
      </div>

      {voiceData ? (
        <>
          {/* Final decision */}
          <div className="voice-main">
            <div className="voice-decision-row">
              <span className="voice-command">{voiceData.command ?? '—'}</span>
              <span className="badge badge-info">HMM</span>
            </div>
            <div className="voice-meta">
              <span className="muted">margen: <b>{voiceData.confidence != null ? voiceData.confidence.toFixed(4) : '—'}</b></span>
              {voiceData.inference_time_ms != null && (
                <span className="muted">{voiceData.inference_time_ms.toFixed(0)} ms</span>
              )}
            </div>
          </div>

          {/* HMM top-3 */}
          {hmmRows && (
            <div className="models-grid">
              <ModelPredictions
                label="HMM"
                rows={hmmRows}
                winner={voiceData.command}
                scoreLabel="log-lik ↑"
              />
            </div>
          )}
        </>
      ) : (
        <p className="muted">Sin resultados aún — pulsa Escuchar</p>
      )}

      {history.length > 0 && (
        <>
          <hr />
          <div className="voice-history">
            <span className="muted small">Historial:</span>
            {history.slice(-MAX_HISTORY).reverse().map((h, i) => (
              <span key={i} className="history-item">{h}</span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
