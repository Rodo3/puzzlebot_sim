import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createWebSocketClient } from './services/websocketClient.js';
import StatusPanel       from './components/StatusPanel.jsx';
import SlamMap           from './components/SlamMap.jsx';
import LidarView         from './components/LidarView.jsx';
import VelocityPanel     from './components/VelocityPanel.jsx';
import VoiceCommandPanel from './components/VoiceCommandPanel.jsx';
import LogsPanel         from './components/LogsPanel.jsx';

const WS_URL = import.meta.env.VITE_WS_URL ?? 'ws://localhost:8000/ws';
const MAX_TRAJECTORY = 500;
const MAX_LOGS = 50;
const MAX_VOICE_HISTORY = 20;

function nowStr() {
  return new Date().toLocaleTimeString();
}

export default function App() {
  const [connected,   setConnected]   = useState(false);
  const [lastUpdate,  setLastUpdate]  = useState(null);
  const [robotState,  setRobotState]  = useState(null);
  const [scanData,    setScanData]    = useState(null);
  const [mapData,     setMapData]     = useState(null);
  const [cmdVel,      setCmdVel]      = useState(null);
  const [cmdVelIn,    setCmdVelIn]    = useState(null);
  const [voiceData,   setVoiceData]   = useState(null);
  const [trajectory,  setTrajectory]  = useState([]);
  const [voiceHistory, setVoiceHistory] = useState([]);
  const [logs,        setLogs]        = useState([]);

  const topicStatus = {
    odom:   !!robotState,
    scan:   !!scanData,
    map:    !!mapData,
    cmdVel: !!cmdVel,
    voice:  !!voiceData,
  };

  const addLog = useCallback((msg) => {
    setLogs(prev => [...prev.slice(-(MAX_LOGS - 1)), { time: nowStr(), msg }]);
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
    return () => client.close();
  }, [handleMessage, addLog]);

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
          />
          <LidarView scanData={scanData} />
        </section>

        <aside className="col-right">
          <StatusPanel
            connected={connected}
            lastUpdate={lastUpdate}
            topicStatus={topicStatus}
          />
          <VelocityPanel cmdVel={cmdVel} cmdVelIn={cmdVelIn} />
          <VoiceCommandPanel voiceData={voiceData} history={voiceHistory} />
          <LogsPanel logs={logs} />
        </aside>
      </main>
    </div>
  );
}
