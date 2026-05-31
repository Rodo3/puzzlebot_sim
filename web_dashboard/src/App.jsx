import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createWebSocketClient } from './services/websocketClient.js';
import StatusPanel       from './components/StatusPanel.jsx';
import SlamMap           from './components/SlamMap.jsx';
import LidarView         from './components/LidarView.jsx';
import CameraPanel       from './components/CameraPanel.jsx';
import VelocityPanel     from './components/VelocityPanel.jsx';
import VoiceCommandPanel from './components/VoiceCommandPanel.jsx';
import ModePanel         from './components/ModePanel.jsx';
import TeleopPanel       from './components/TeleopPanel.jsx';
import WaypointPanel     from './components/WaypointPanel.jsx';
import LogsPanel         from './components/LogsPanel.jsx';

const WS_URL = import.meta.env.VITE_WS_URL ?? `ws://${window.location.hostname}:8000/ws`;
const MAX_TRAJECTORY = 500;
const MAX_LOGS = 50;
const MAX_VOICE_HISTORY = 20;

function nowStr() {
  return new Date().toLocaleTimeString();
}

export default function App() {
  const [connected,    setConnected]    = useState(false);
  const [lastUpdate,   setLastUpdate]   = useState(null);
  const [robotState,   setRobotState]   = useState(null);
  const [scanData,     setScanData]     = useState(null);
  const [mapData,      setMapData]      = useState(null);
  const [cmdVel,       setCmdVel]       = useState(null);
  const [cmdVelIn,     setCmdVelIn]     = useState(null);
  const [voiceData,    setVoiceData]    = useState(null);
  const [cameraData,   setCameraData]   = useState(null);
  const [trajectory,   setTrajectory]   = useState([]);
  const [voiceHistory, setVoiceHistory] = useState([]);
  const [logs,         setLogs]         = useState([]);

  // Control state
  const [mode,       setMode]       = useState('mapping');   // 'mapping' | 'navigation'
  const [goalMarker, setGoalMarker] = useState(null);        // {x, y} in world coords

  const clientRef = useRef(null);

  const topicStatus = {
    odom:   !!robotState,
    scan:   !!scanData,
    map:    !!mapData,
    cmdVel: !!cmdVel,
    camera: !!cameraData,
    voice:  !!voiceData,
  };

  const addLog = useCallback((msg) => {
    setLogs(prev => [...prev.slice(-(MAX_LOGS - 1)), { time: nowStr(), msg }]);
  }, []);

  const sendCommand = useCallback((data) => {
    clientRef.current?.send(data);
  }, []);

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
        addLog('Receiving map update');
        break;

      case 'velocity_command':
        if (msg.source === 'cmd_vel')    setCmdVel(msg);
        if (msg.source === 'cmd_vel_in') setCmdVelIn(msg);
        break;

      case 'voice_command':
        setVoiceData(msg);
        if (msg.command) {
          setVoiceHistory(prev => [...prev.slice(-(MAX_VOICE_HISTORY - 1)), msg.command]);
          addLog(`Voice command detected: ${msg.command}`);
        }
        break;

      case 'camera_frame':
        setCameraData(msg);
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

  return (
    <div className="app">
      <header className="header">
        <h1>Puzzlebot Live Dashboard</h1>
        <div className={`conn-badge ${connected ? 'conn-ok' : 'conn-err'}`}>
          {connected ? 'Connected' : 'Disconnected'}
        </div>
        {lastUpdate && (
          <span className="muted small">
            Last update: {new Date(lastUpdate * 1000).toLocaleTimeString()}
          </span>
        )}
      </header>

      <main className="main-grid">
        <section className="col-left">
          <SlamMap
            mapData={mapData}
            robotPose={robotState?.pose}
            trajectory={trajectory}
            mode={mode}
            goalMarker={goalMarker}
            onGoalPose={handleGoalPose}
          />
          <div className="col-left-bottom">
            <LidarView scanData={scanData} />
            <CameraPanel cameraData={cameraData} />
          </div>
        </section>

        <aside className="col-right">
          <StatusPanel
            connected={connected}
            lastUpdate={lastUpdate}
            topicStatus={topicStatus}
          />
          <ModePanel
            mode={mode}
            connected={connected}
            onModeChange={setMode}
            onSlamReset={handleSlamReset}
          />
          <TeleopPanel
            connected={connected}
            onCommand={sendCommand}
          />
          <WaypointPanel
            connected={connected}
            mode={mode}
            onCommand={sendCommand}
            addLog={addLog}
          />
          <VelocityPanel cmdVel={cmdVel} cmdVelIn={cmdVelIn} />
          <VoiceCommandPanel voiceData={voiceData} history={voiceHistory} />
          <LogsPanel logs={logs} />
        </aside>
      </main>
    </div>
  );
}
