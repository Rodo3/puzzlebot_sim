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
import MissionPanel      from './components/MissionPanel.jsx';
import MetricsPanel      from './components/MetricsPanel.jsx';

const WS_URL            = import.meta.env.VITE_WS_URL ?? `ws://${window.location.hostname}:8000/ws`;
const ROBOT_ENV         = import.meta.env.VITE_ROBOT_ENV ?? 'sim';
const MAX_TRAJECTORY    = 500;
const MAX_LOGS          = 50;
const MAX_VOICE_HISTORY = 20;
const MAX_METRICS       = 400;
const METRICS_MIN_DT    = 0.15;

function nowStr() { return new Date().toLocaleTimeString(); }

function navStateClass(state) {
  if (!state || state === 'NORMAL')         return 'nav-state-normal';
  if (state === 'FOLLOW_NEW_PATH')          return 'nav-state-follow';
  if (state === 'BRAKE_FOR_REPLAN' || state === 'REPLAN') return 'nav-state-replan';
  return 'nav-state-recovery';
}

function makeSessionStats() {
  return {
    startTime:        Date.now() / 1000,
    distanceTraveled: 0,
    obstacleStops:    0,
    replanCount:      0,
    maxLinearVel:     0,
    goalsSent:        0,
    arucoCount:       0,
    scanMatchCount:   0,
  };
}

// ── ANTI MASS logo mark (inline SVG) ─────────────────────
function LogoMark() {
  return (
    <svg viewBox="0 0 48 36" width="48" height="36" aria-hidden="true" style={{ flexShrink: 0 }}>
      {/* White A — mountain peak */}
      <path d="M3 33 L16 5 L22 17 L16 33Z" fill="#f0f0f8" />
      {/* Purple M */}
      <path d="M16 5 L22 17 L27 9 L32 17 L38 5 L45 33 L37 33 L32 20 L27 26 L22 20 L18 33 L10 33Z"
            fill="url(#mGrad)" />
      {/* Cyan arrow inside A */}
      <path d="M9 29 L13 14" stroke="#22d3ee" strokeWidth="2" strokeLinecap="round" fill="none"/>
      <polygon points="11 12 16 10 14 17" fill="#22d3ee" />
      {/* Route start circle */}
      <circle cx="3" cy="33" r="2.5" fill="none" stroke="#7c3aed" strokeWidth="1.5"/>
      {/* Route end pin */}
      <circle cx="45" cy="33" r="2" fill="#22d3ee" />
      {/* Gradient def */}
      <defs>
        <linearGradient id="mGrad" x1="16" y1="5" x2="45" y2="33" gradientUnits="userSpaceOnUse">
          <stop offset="0%"   stopColor="#7c3aed" />
          <stop offset="100%" stopColor="#a855f7" />
        </linearGradient>
      </defs>
    </svg>
  );
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
  const [missionState,   setMissionState]   = useState('IDLE');
  const [cameraData,     setCameraData]     = useState(null);
  const [qrDetections,   setQrDetections]   = useState([]);
  const [logoDetections, setLogoDetections] = useState([]);
  const [trajectory,     setTrajectory]     = useState([]);
  const [voiceHistory,   setVoiceHistory]   = useState([]);
  const [logs,           setLogs]           = useState([]);
  const [mode,           setMode]           = useState('mapping');
  const [goalMarker,     setGoalMarker]     = useState(null);
  const [availableMaps,  setAvailableMaps]  = useState([]);
  const [mapSource,      setMapSource]      = useState('live');
  const [mainView,       setMainView]       = useState('robot');

  // ── Metrics state ───────────────────────────────────────────────────────────
  const [velHistory,     setVelHistory]     = useState([]);
  const [lidarHist,      setLidarHist]      = useState([]);
  const [domStateLog,    setDomStateLog]    = useState([]);
  const [sessionStats,   setSessionStats]   = useState(makeSessionStats);
  const [poseHist,       setPoseHist]       = useState([]);
  const [kalmanCovHist,  setKalmanCovHist]  = useState([]);
  const [locErrorHist,   setLocErrorHist]   = useState([]);
  const [arucoEvents,    setArucoEvents]    = useState([]);
  const [scanMatchHist,  setScanMatchHist]  = useState([]);

  // ── Refs ────────────────────────────────────────────────────────────────────
  const clientRef         = useRef(null);
  const navStateRef       = useRef('NORMAL');
  const prevPoseRef       = useRef(null);
  const lastVelTimeRef    = useRef(0);
  const lastLidTimeRef    = useRef(0);
  const lastKalTimeRef    = useRef(0);
  const latestKalmanPose  = useRef(null);

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
        if (prevPoseRef.current && msg.pose) {
          const dx   = msg.pose.x - prevPoseRef.current.x;
          const dy   = msg.pose.y - prevPoseRef.current.y;
          const dist = Math.hypot(dx, dy);
          if (dist > 0.001 && dist < 0.5) {
            setSessionStats(prev => ({
              ...prev,
              distanceTraveled: prev.distanceTraveled + dist,
            }));
          }
        }
        prevPoseRef.current = msg.pose ?? null;
        if (msg.kalman_pose) latestKalmanPose.current = msg.kalman_pose;
        const t = msg.timestamp ?? Date.now() / 1000;
        if (t - lastKalTimeRef.current >= METRICS_MIN_DT) {
          const kp = msg.kalman_pose ?? msg.pose;
          if (kp) {
            setPoseHist(prev => {
              const next = [...prev, { time: t, x: kp.x, y: kp.y, theta: kp.theta }];
              return next.length > MAX_METRICS ? next.slice(-MAX_METRICS) : next;
            });
          }
          if (msg.cov_xx != null && msg.cov_yy != null) {
            const traceP = (msg.cov_xx ?? 0) + (msg.cov_yy ?? 0);
            setKalmanCovHist(prev => {
              const next = [...prev, { time: t, traceP, pTheta: msg.cov_tt ?? 0 }];
              return next.length > MAX_METRICS ? next.slice(-MAX_METRICS) : next;
            });
          }
          if (msg.kalman_pose && msg.slam_pose) {
            const dx  = msg.slam_pose.x - msg.kalman_pose.x;
            const dy  = msg.slam_pose.y - msg.kalman_pose.y;
            const dth = Math.abs(msg.slam_pose.theta - msg.kalman_pose.theta);
            setLocErrorHist(prev => {
              const next = [...prev, { time: t, error_xy: Math.hypot(dx, dy), error_theta: dth }];
              return next.length > MAX_METRICS ? next.slice(-MAX_METRICS) : next;
            });
          }
          lastKalTimeRef.current = t;
        }
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

      case 'mission_state': {
        const s = msg.state ?? 'IDLE';
        setMissionState(prev => {
          if (prev !== s) addLog(`Misión: ${prev} → ${s}`);
          return s;
        });
        break;
      }

      case 'camera_frame':
        setCameraData(msg);
        break;

      case 'qr_detections':
        setQrDetections(msg.detections ?? []);
        break;

      case 'logo_detection':
        setLogoDetections(msg.detections ?? []);
        break;

      case 'available_maps':
        setAvailableMaps(msg.maps ?? []);
        break;

      case 'aruco_pose': {
        const t = msg.timestamp ?? Date.now() / 1000;
        let innovation = null;
        if (latestKalmanPose.current) {
          const dx = (msg.x ?? 0) - latestKalmanPose.current.x;
          const dy = (msg.y ?? 0) - latestKalmanPose.current.y;
          innovation = Math.hypot(dx, dy);
        }
        setArucoEvents(prev => {
          const next = [...prev, {
            time:       t,
            x:          msg.x ?? 0,
            y:          msg.y ?? 0,
            theta:      msg.theta ?? 0,
            cov_xy:     (msg.cov_xx ?? 0) + (msg.cov_yy ?? 0),
            innovation: innovation ?? 0,
          }];
          return next.length > MAX_METRICS ? next.slice(-MAX_METRICS) : next;
        });
        setSessionStats(prev => ({ ...prev, arucoCount: prev.arucoCount + 1 }));
        break;
      }

      case 'scan_match_pose': {
        const t = msg.timestamp ?? Date.now() / 1000;
        setScanMatchHist(prev => {
          const next = [...prev, {
            time:   t,
            x:      msg.x ?? 0,
            y:      msg.y ?? 0,
            cov_xy: ((msg.cov_xx ?? 0) + (msg.cov_yy ?? 0)),
          }];
          return next.length > MAX_METRICS ? next.slice(-MAX_METRICS) : next;
        });
        setSessionStats(prev => ({ ...prev, scanMatchCount: prev.scanMatchCount + 1 }));
        break;
      }

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
    setPoseHist([]);
    setKalmanCovHist([]);
    setLocErrorHist([]);
    setArucoEvents([]);
    setScanMatchHist([]);
    prevPoseRef.current    = null;
    lastVelTimeRef.current = 0;
    lastLidTimeRef.current = 0;
    lastKalTimeRef.current = 0;
    addLog('Metrics session reset');
  }, [addLog]);

  // ── Header topic chips ───────────────────────────────────────────────────────
  const topicDots = [
    { key: 'odom',  label: 'odom',  active: !!robotState },
    { key: 'scan',  label: 'scan',  active: !!scanData },
    { key: 'map',   label: 'map',   active: !!mapData },
    { key: 'aug',   label: 'aug',   active: !!augMapData },
    { key: 'vel',   label: 'vel',   active: !!cmdVel },
    { key: 'aruco', label: 'aruco', active: arucoEvents.length > 0 },
    { key: 'cam',   label: 'cam',   active: !!cameraData },
    { key: 'voice', label: 'voice', active: !!voiceData },
  ];

  return (
    <div className="app">

      {/* ── Header ── */}
      <header className="header">
        {/* Brand */}
        <div className="header-brand">
          <LogoMark />
          <div className="header-brand-text">
            <span className="header-title-main">ANTI MASS</span>
            <span className="header-title-sub">AUTONOMOUS LOGISTICS</span>
          </div>
        </div>

        <div className="header-divider" />

        {/* View tabs */}
        <div className="view-tabs">
          <button
            className={`view-tab ${mainView === 'robot'   ? 'view-tab-active' : ''}`}
            onClick={() => setMainView('robot')}
          >Robot</button>
          <button
            className={`view-tab ${mainView === 'metrics' ? 'view-tab-active' : ''}`}
            onClick={() => setMainView('metrics')}
          >Métricas</button>
        </div>

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

      {/* ── Metrics full-screen view ── */}
      {mainView === 'metrics' && (
        <div className="metrics-view">
          <MetricsPanel
            velHistory={velHistory}
            lidarHist={lidarHist}
            domStateLog={domStateLog}
            poseHist={poseHist}
            kalmanCovHist={kalmanCovHist}
            locErrorHist={locErrorHist}
            arucoEvents={arucoEvents}
            scanMatchHist={scanMatchHist}
            sessionStats={sessionStats}
            onReset={handleResetSession}
          />
        </div>
      )}

      {/* ── Robot view — 3 columns ── */}
      {mainView === 'robot' && (
        <div className="main-area">

          {/* ── Column 1: Control ── */}
          <div className="col-left">
            {/* Mission — pinned at top */}
            <div className="section-mission">
              <MissionPanel
                missionState={missionState}
                connected={connected}
                onCommand={sendCommand}
              />
            </div>

            {/* Mode + Waypoints — scrollable */}
            <div className="col-left-scroll">
              <div className="section-mode">
                <ModePanel
                  mode={mode}
                  connected={connected}
                  onModeChange={handleModeChange}
                  onSlamReset={handleSlamReset}
                  availableMaps={availableMaps}
                  mapSource={mapSource}
                  onCommand={handleCommand}
                />
              </div>
              <div className="section-waypoints">
                <WaypointPanel
                  connected={connected}
                  mode={mode}
                  onGoalPose={handleGoalPose}
                  onCommand={sendCommand}
                  addLog={addLog}
                />
              </div>
            </div>
          </div>

          {/* ── Column 2: SLAM Map ── */}
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

          {/* ── Column 3: Sensors & I/O ── */}
          <div className="col-right">
            {/* Camera — pinned at top */}
            <div className="section-camera">
              <CameraPanel
                cameraData={cameraData}
                qrDetections={qrDetections}
                logoDetections={logoDetections}
              />
            </div>

            {/* LiDAR + Teleop + Voice — scrollable */}
            <div className="col-right-scroll">
              <div className="section-lidar">
                <LidarView scanData={scanData} />
              </div>
              <div className="section-teleop">
                <TeleopPanel connected={connected} onCommand={sendCommand} />
              </div>
              <div className="section-voice">
                <VoiceCommandPanel voiceData={voiceData} history={voiceHistory} />
              </div>
            </div>
          </div>

        </div>
      )}

      {/* ── Footer (always visible) ── */}
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

    </div>
  );
}
