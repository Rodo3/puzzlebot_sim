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

const WS_URL            = import.meta.env.VITE_WS_URL ?? `ws://${window.location.hostname}:8000/ws`;
const MAX_TRAJECTORY    = 500;
const MAX_LOGS          = 50;
const MAX_VOICE_HISTORY = 20;

function nowStr() { return new Date().toLocaleTimeString(); }

const TABS = [
  { id: 'mode',      label: 'Modo' },
  { id: 'waypoints', label: 'Waypoints' },
  { id: 'voice',     label: 'Voz' },
  { id: 'elevator',  label: 'Elevador' },
];

export default function App() {
  const [connected,     setConnected]     = useState(false);
  const [lastUpdate,    setLastUpdate]    = useState(null);
  const [robotState,    setRobotState]    = useState(null);
  const [scanData,      setScanData]      = useState(null);
  const [mapData,       setMapData]       = useState(null);
  const [cmdVel,        setCmdVel]        = useState(null);
  const [cmdVelIn,      setCmdVelIn]      = useState(null);
  const [voiceData,     setVoiceData]     = useState(null);
  const [cameraData,    setCameraData]    = useState(null);
  const [trajectory,    setTrajectory]    = useState([]);
  const [voiceHistory,  setVoiceHistory]  = useState([]);
  const [logs,          setLogs]          = useState([]);
  const [mode,          setMode]          = useState('mapping');
  const [goalMarker,    setGoalMarker]    = useState(null);
  const [activeTab,     setActiveTab]     = useState('mode');
  const [availableMaps, setAvailableMaps] = useState([]);
  const [mapSource,     setMapSource]     = useState('live');

  const clientRef = useRef(null);

  const addLog = useCallback((msg) => {
    setLogs(prev => [...prev.slice(-(MAX_LOGS - 1)), { time: nowStr(), msg }]);
  }, []);

  const sendCommand = useCallback((data) => {
    clientRef.current?.send(data);
  }, []);

  // Wraps sendCommand to also update mapSource locally
  const handleCommand = useCallback((data) => {
    if (data.type === 'load_map')    setMapSource('static');
    if (data.type === 'use_slam_map') setMapSource('live');
    sendCommand(data);
  }, [sendCommand]);

  const handleMessage = useCallback((msg) => {
    setLastUpdate(msg.timestamp ?? Date.now() / 1000);

    switch (msg.type) {
      case 'robot_state':
        setRobotState(msg);
        setTrajectory(prev => {
          const next = [...prev, msg.pose];
          return next.length > MAX_TRAJECTORY ? next.slice(-MAX_TRAJECTORY) : next;
        });
        break;
      case 'scan':
        setScanData(msg);
        break;
      case 'map':
        setMapData(msg);
        break;
      case 'velocity_command':
        if (msg.source === 'cmd_vel')    setCmdVel(msg);
        if (msg.source === 'cmd_vel_in') setCmdVelIn(msg);
        break;
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

  const handleGoalPose = useCallback(({ x, y, theta = 0 }) => {
    sendCommand({ type: 'goal_pose', x, y, theta });
    setGoalMarker({ x, y });
    addLog(`Goal → (${x.toFixed(2)}, ${y.toFixed(2)})`);
  }, [sendCommand, addLog]);

  const handleSlamReset = useCallback(() => {
    sendCommand({ type: 'slam_reset' });
    setGoalMarker(null);
    setTrajectory([]);
    addLog('SLAM map reset');
  }, [sendCommand, addLog]);

  const handleModeChange = useCallback((newMode) => {
    setMode(newMode);
    addLog(`Mode → ${newMode}`);
  }, [addLog]);

  const topicDots = [
    { key: 'odom',  label: 'odom',  active: !!robotState },
    { key: 'scan',  label: 'scan',  active: !!scanData },
    { key: 'map',   label: 'map',   active: !!mapData },
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
        <span className={`mode-pill mode-pill-${mode}`}>
          {mode === 'mapping' ? 'MAPPING' : 'NAVIGATION'}
        </span>
      </header>

      {/* ── Main area ── */}
      <div className="main-area">
        {/* SLAM map column */}
        <div className="col-slam">
          <SlamMap
            mapData={mapData}
            robotPose={robotState?.pose}
            trajectory={trajectory}
            mode={mode}
            goalMarker={goalMarker}
            onGoalPose={handleGoalPose}
          />
        </div>

        {/* Right column */}
        <div className="col-right">
          {/* Sensors row: LiDAR + Camera */}
          <div className="sensors-row">
            <LidarView scanData={scanData} />
            <CameraPanel cameraData={cameraData} />
          </div>

          {/* Teleop */}
          <TeleopPanel connected={connected} onCommand={sendCommand} />

          {/* Tabs card */}
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
        </div>
      </div>

      {/* ── Footer ── */}
      <footer className="footer">
        <VelocityPanel cmdVel={cmdVel} cmdVelIn={cmdVelIn} />
        <LogsPanel logs={logs} />
      </footer>
    </div>
  );
}
