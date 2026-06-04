import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createWebSocketClient } from './services/websocketClient.js';
import SlamMap           from './components/SlamMap.jsx';
import LidarView         from './components/LidarView.jsx';
import CameraPanel       from './components/CameraPanel.jsx';
import VelocityPanel     from './components/VelocityPanel.jsx';
import VoiceCommandPanel from './components/VoiceCommandPanel.jsx';
import ModePanel         from './components/ModePanel.jsx';
import TeleopPanel       from './components/TeleopPanel.jsx';
import WaypointPanel     from './components/WaypointPanel.jsx';
import LogsPanel         from './components/LogsPanel.jsx';
import ElevatorPanel     from './components/ElevatorPanel.jsx';
import MetricsPanel      from './components/MetricsPanel.jsx';

const WS_URL            = import.meta.env.VITE_WS_URL ?? `ws://${window.location.hostname}:8000/ws`;
const ROBOT_ENV         = import.meta.env.VITE_ROBOT_ENV ?? 'sim';
const MAX_TRAJECTORY    = 500;
const MAX_LOGS          = 50;
const MAX_VOICE_HISTORY = 20;
const MAX_METRICS       = 400;   // max points per time-series
const METRICS_MIN_DT    = 0.15;  // min seconds between stored metric points (throttle)

function nowStr() { return new Date().toLocaleTimeString(); }

function navStateClass(state) {
  if (!state || state === 'NORMAL')         return 'nav-state-normal';
  if (state === 'FOLLOW_NEW_PATH')          return 'nav-state-follow';
  if (state === 'BRAKE_FOR_REPLAN' || state === 'REPLAN') return 'nav-state-replan';
  return 'nav-state-recovery';
}

const TABS = [
  { id: 'mode',      label: 'Modo' },
  { id: 'waypoints', label: 'Waypoints' },
  { id: 'voice',     label: 'Voz' },
  { id: 'elevator',  label: 'Elevador' },
];

function makeSessionStats() {
  return {
    startTime:        Date.now() / 1000,
    distanceTraveled: 0,
    obstacleStops:    0,
    replanCount:      0,
    maxLinearVel:     0,
    goalsSent:        0,
  };
}

export default function App() {
  // ── Core robot data ─────────────────────────────────────────────────────────
  const [connected,      setConnected]      = useState(false);
  const [lastUpdate,     setLastUpdate]     = useState(null);
  const [robotState,     setRobotState]     = useState(null);
  const [scanData,       setScanData]       = useState(null);
  const [mapData,        setMapData]        = useState(null);
  const [augMapData,     setAugMapData]     = useState(null);
  const [cmdVel,         setCmdVel]         = useState(null);
  const [cmdVelIn,       setCmdVelIn]       = useState(null);
  const [cmdVelSteering, setCmdVelSteering] = useState(null);
  const [navState,       setNavState]       = useState('NORMAL');
  const [voiceData,      setVoiceData]      = useState(null);
  const [cameraData,     setCameraData]     = useState(null);
  const [trajectory,     setTrajectory]     = useState([]);
  const [voiceHistory,   setVoiceHistory]   = useState([]);
  const [logs,           setLogs]           = useState([]);
  const [mode,           setMode]           = useState('mapping');
  const [goalMarker,     setGoalMarker]     = useState(null);
  const [activeTab,      setActiveTab]      = useState('mode');
  const [availableMaps,  setAvailableMaps]  = useState([]);
  const [mapSource,      setMapSource]      = useState('live');

  // ── Metrics state ───────────────────────────────────────────────────────────
  const [velHistory,       setVelHistory]       = useState([]);
  const [lidarHist,        setLidarHist]        = useState([]);
  const [domStateLog,      setDomStateLog]      = useState([]);
  const [sessionStats,     setSessionStats]     = useState(makeSessionStats);
  const [metricsOpen,      setMetricsOpen]      = useState(false);
  const [metricsFullscreen, setMetricsFullscreen] = useState(false);

  // ── Refs ────────────────────────────────────────────────────────────────────
  const clientRef      = useRef(null);
  const navStateRef    = useRef('NORMAL');
  const prevPoseRef    = useRef(null);
  const lastVelTimeRef = useRef(0);
  const lastLidTimeRef = useRef(0);

  // ── Helpers ─────────────────────────────────────────────────────────────────
  const addLog = useCallback((msg) => {
    setLogs(prev => [...prev.slice(-(MAX_LOGS - 1)), { time: nowStr(), msg }]);
  }, []);

  const sendCommand = useCallback((data) => {
    clientRef.current?.send(data);
  }, []);

  const handleCommand = useCallback((data) => {
    if (data.type === 'load_map')    setMapSource('static');
    if (data.type === 'use_slam_map') setMapSource('live');
    sendCommand(data);
  }, [sendCommand]);

  // ── WebSocket message handler ────────────────────────────────────────────────
  const handleMessage = useCallback((msg) => {
    setLastUpdate(msg.timestamp ?? Date.now() / 1000);

    switch (msg.type) {

      case 'robot_state': {
        setRobotState(msg);
        setTrajectory(prev => {
          const next = [...prev, msg.pose];
          return next.length > MAX_TRAJECTORY ? next.slice(-MAX_TRAJECTORY) : next;
        });
        // Accumulate distance traveled
        if (prevPoseRef.current && msg.pose) {
          const dx   = msg.pose.x - prevPoseRef.current.x;
          const dy   = msg.pose.y - prevPoseRef.current.y;
          const dist = Math.hypot(dx, dy);
          // Guard against teleport artifacts and pure noise
          if (dist > 0.001 && dist < 0.5) {
            setSessionStats(prev => ({
              ...prev,
              distanceTraveled: prev.distanceTraveled + dist,
            }));
          }
        }
        prevPoseRef.current = msg.pose ?? null;
        break;
      }

      case 'scan': {
        setScanData(msg);
        const t = msg.timestamp ?? Date.now() / 1000;
        if (t - lastLidTimeRef.current >= METRICS_MIN_DT) {
          const ranges = Array.isArray(msg.ranges)
            ? msg.ranges.filter(r => isFinite(r) && r > 0.05)
            : [];
          if (ranges.length > 0) {
            const minDist = Math.min(...ranges);
            setLidarHist(prev => {
              const next = [...prev, { time: t, min: minDist }];
              return next.length > MAX_METRICS ? next.slice(-MAX_METRICS) : next;
            });
            lastLidTimeRef.current = t;
          }
        }
        break;
      }

      case 'map':
        setMapData(msg);
        break;

      case 'augmented_map':
        setAugMapData(msg);
        break;

      case 'nav_state': {
        const s = msg.state ?? 'NORMAL';
        setNavState(s);
        if (s !== navStateRef.current) {
          addLog(`DOM: ${navStateRef.current} → ${s}`);
          const t = msg.timestamp ?? Date.now() / 1000;
          setDomStateLog(prev => {
            const next = [...prev, { time: t, state: s }];
            return next.length > 500 ? next.slice(-500) : next;
          });
          if (s === 'REPLAN' || s === 'BRAKE_FOR_REPLAN') {
            setSessionStats(prev => ({ ...prev, replanCount: prev.replanCount + 1 }));
          }
          if (s === 'SAFE_STOP') {
            setSessionStats(prev => ({ ...prev, obstacleStops: prev.obstacleStops + 1 }));
          }
          navStateRef.current = s;
        }
        break;
      }

      case 'velocity_command': {
        if (msg.source === 'cmd_vel') {
          setCmdVel(msg);
          const t = msg.timestamp ?? Date.now() / 1000;
          if (t - lastVelTimeRef.current >= METRICS_MIN_DT) {
            setVelHistory(prev => {
              const next = [...prev, {
                time:    t,
                linear:  msg.linear_x  ?? 0,
                angular: msg.angular_z ?? 0,
              }];
              return next.length > MAX_METRICS ? next.slice(-MAX_METRICS) : next;
            });
            setSessionStats(prev => ({
              ...prev,
              maxLinearVel: Math.max(prev.maxLinearVel, Math.abs(msg.linear_x ?? 0)),
            }));
            lastVelTimeRef.current = t;
          }
        }
        if (msg.source === 'cmd_vel_in')       setCmdVelIn(msg);
        if (msg.source === 'cmd_vel_steering') setCmdVelSteering(msg);
        break;
      }

      case 'voice_command':
        setVoiceData(msg);
        if (msg.command) {
          setVoiceHistory(prev => [...prev.slice(-(MAX_VOICE_HISTORY - 1)), msg.command]);
          addLog(`Voice: ${msg.command}`);
        }
        break;

      case 'camera_frame':
        setCameraData(msg);
        break;

      case 'available_maps':
        setAvailableMaps(msg.maps ?? []);
        break;

      default:
        break;
    }
  }, [addLog]);

  useEffect(() => {
    const client = createWebSocketClient(WS_URL, {
      onConnect:    () => { setConnected(true);  addLog('WebSocket connected'); },
      onDisconnect: () => { setConnected(false); addLog('WebSocket disconnected'); },
      onMessage:    handleMessage,
    });
    clientRef.current = client;
    return () => client.close();
  }, [handleMessage, addLog]);

  // ── Action callbacks ─────────────────────────────────────────────────────────
  const handleGoalPose = useCallback(({ x, y, theta = 0 }) => {
    sendCommand({ type: 'goal_pose', x, y, theta });
    setGoalMarker({ x, y });
    setSessionStats(prev => ({ ...prev, goalsSent: prev.goalsSent + 1 }));
    addLog(`Goal → (${x.toFixed(2)}, ${y.toFixed(2)})`);
  }, [sendCommand, addLog]);

  const handleSlamReset = useCallback(() => {
    sendCommand({ type: 'slam_reset' });
    setGoalMarker(null);
    setTrajectory([]);
    setAugMapData(null);
    addLog('SLAM map reset');
  }, [sendCommand, addLog]);

  const handleModeChange = useCallback((newMode) => {
    setMode(newMode);
    addLog(`Mode → ${newMode}`);
  }, [addLog]);

  const handleResetSession = useCallback(() => {
    setSessionStats(makeSessionStats());
    setVelHistory([]);
    setLidarHist([]);
    setDomStateLog([]);
    prevPoseRef.current   = null;
    lastVelTimeRef.current = 0;
    lastLidTimeRef.current = 0;
    addLog('Metrics session reset');
  }, [addLog]);

  // ── Header topic dots ────────────────────────────────────────────────────────
  const topicDots = [
    { key: 'odom',  label: 'odom',  active: !!robotState },
    { key: 'scan',  label: 'scan',  active: !!scanData },
    { key: 'map',   label: 'map',   active: !!mapData },
    { key: 'aug',   label: 'aug',   active: !!augMapData },
    { key: 'vel',   label: 'vel',   active: !!cmdVel },
    { key: 'cam',   label: 'cam',   active: !!cameraData },
    { key: 'voice', label: 'voice', active: !!voiceData },
  ];

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <span className="header-title">PUZZLEBOT LIVE DASHBOARD</span>
        <span className="header-sep">|</span>
        <span className={`conn-badge ${connected ? 'conn-ok' : 'conn-err'}`}>
          {connected ? 'Connected' : 'Disconnected'}
        </span>
        {lastUpdate && (
          <span className="header-timestamp">
            {new Date(lastUpdate * 1000).toLocaleTimeString()}
          </span>
        )}
        <div className="topic-dots">
          {topicDots.map(({ key, label, active }) => (
            <div key={key} className="topic-dot">
              <div className={`dot ${active ? 'dot-ok' : 'dot-err'}`} />
              {label}
            </div>
          ))}
        </div>
        <span className={`env-badge env-badge-${ROBOT_ENV}`}>
          {ROBOT_ENV.toUpperCase()}
        </span>
        <span className={`mode-pill mode-pill-${mode}`}>
          {mode === 'mapping' ? 'MAPPING' : 'NAV'}
        </span>
        {mode === 'navigation' && (
          <span className={`nav-state-pill ${navStateClass(navState)}`}>
            {navState.replace(/_/g, ' ')}
          </span>
        )}
      </header>

      {/* ── Main area ── */}
      <div className="main-area">
        {/* SLAM map column */}
        <div className="col-slam">
          <SlamMap
            mapData={mapData}
            augMapData={augMapData}
            robotPose={robotState?.pose}
            trajectory={trajectory}
            mode={mode}
            goalMarker={goalMarker}
            onGoalPose={handleGoalPose}
          />
        </div>

        {/* Right column */}
        <div className="col-right">
          {/* Pinned — always visible regardless of scroll */}
          <div className="sensors-row">
            <LidarView scanData={scanData} />
            <CameraPanel cameraData={cameraData} />
          </div>

          {/* Scrollable: Teleop + Tabs */}
          <div className="col-right-scroll">
          <TeleopPanel connected={connected} onCommand={sendCommand} />

          <div className="tabs-card">
            <div className="tabs-header">
              {TABS.map(t => (
                <button
                  key={t.id}
                  className={`tab-btn ${activeTab === t.id ? 'tab-btn-active' : ''}`}
                  onClick={() => setActiveTab(t.id)}
                >{t.label}</button>
              ))}
            </div>
            <div className="tab-content">
              {activeTab === 'mode' && (
                <ModePanel
                  mode={mode}
                  connected={connected}
                  onModeChange={handleModeChange}
                  onSlamReset={handleSlamReset}
                  availableMaps={availableMaps}
                  mapSource={mapSource}
                  onCommand={handleCommand}
                />
              )}
              {activeTab === 'waypoints' && (
                <WaypointPanel
                  connected={connected}
                  mode={mode}
                  onGoalPose={handleGoalPose}
                  onCommand={sendCommand}
                  addLog={addLog}
                />
              )}
              {activeTab === 'voice' && (
                <VoiceCommandPanel voiceData={voiceData} history={voiceHistory} />
              )}
              {activeTab === 'elevator' && (
                <ElevatorPanel connected={connected} onCommand={handleCommand} />
              )}
            </div>
          </div>
          </div>{/* end col-right-scroll */}
        </div>
      </div>

      {/* ── Metrics bar (Option A) ── */}
      <div className={`metrics-bar ${metricsOpen ? 'metrics-bar-open' : ''}`}>
        <div
          className="metrics-bar-header"
          onClick={() => setMetricsOpen(v => !v)}
          title={metricsOpen ? 'Colapsar métricas' : 'Expandir métricas'}
        >
          <span className="metrics-bar-chevron">{metricsOpen ? '▼' : '▶'}</span>
          <span className="metrics-bar-title">MÉTRICAS</span>

          {/* Mini stats always visible — at a glance even when collapsed */}
          <div className="metrics-bar-mini">
            <span className="metrics-mini-stat">
              <span className="metrics-mini-label">dist</span>
              <b>{sessionStats.distanceTraveled.toFixed(1)}m</b>
            </span>
            <span className="metrics-mini-stat">
              <span className="metrics-mini-label">vel max</span>
              <b>{sessionStats.maxLinearVel.toFixed(2)}m/s</b>
            </span>
            <span className="metrics-mini-stat" style={{ color: 'var(--warn)' }}>
              <span className="metrics-mini-label">replans</span>
              <b>{sessionStats.replanCount}</b>
            </span>
            <span className="metrics-mini-stat" style={{ color: 'var(--err)' }}>
              <span className="metrics-mini-label">stops</span>
              <b>{sessionStats.obstacleStops}</b>
            </span>
          </div>

          {/* Controls — stopPropagation so they don't toggle the bar */}
          <div className="metrics-bar-actions" onClick={e => e.stopPropagation()}>
            <button className="btn-sm" onClick={handleResetSession} title="Reiniciar sesión">↺</button>
            <button className="btn-sm btn-sm-accent" onClick={() => setMetricsFullscreen(true)} title="Ver en pantalla completa (Opción B)">⛶</button>
          </div>
        </div>

        {metricsOpen && (
          <div className="metrics-bar-body">
            <MetricsPanel
              velHistory={velHistory}
              lidarHist={lidarHist}
              domStateLog={domStateLog}
              sessionStats={sessionStats}
              onReset={handleResetSession}
            />
          </div>
        )}
      </div>

      {/* ── Footer ── */}
      <footer className="footer">
        <VelocityPanel
          cmdVel={cmdVel}
          cmdVelIn={cmdVelIn}
          cmdVelSteering={cmdVelSteering}
          navState={navState}
          mode={mode}
        />
        <LogsPanel logs={logs} />
      </footer>

      {/* ── Fullscreen metrics overlay (Option B) ── */}
      {metricsFullscreen && (
        <div className="metrics-overlay">
          <div className="metrics-overlay-header">
            <span className="metrics-overlay-title">📊 MÉTRICAS — VISTA COMPLETA</span>
            <div style={{ display: 'flex', gap: 6 }}>
              <button
                className="btn-sm btn-sm-accent"
                onClick={() => { setMetricsOpen(true); setMetricsFullscreen(false); }}
                title="Volver a vista de barra"
              >⊡ Barra</button>
              <button
                className="btn-sm btn-sm-green"
                onClick={() => setMetricsFullscreen(false)}
                title="Cerrar"
              >✕ Cerrar</button>
            </div>
          </div>
          <div className="metrics-overlay-body">
            <MetricsPanel
              velHistory={velHistory}
              lidarHist={lidarHist}
              domStateLog={domStateLog}
              sessionStats={sessionStats}
              onReset={handleResetSession}
            />
          </div>
        </div>
      )}
    </div>
  );
}
