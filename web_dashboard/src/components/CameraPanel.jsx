import React from 'react';

export default function CameraPanel({ cameraData }) {
  if (!cameraData) {
    return (
      <div className="panel camera-panel">
        <h3>Camera</h3>
        <div className="camera-placeholder">Waiting for /camera/image/compressed…</div>
      </div>
    );
  }

  return (
    <div className="panel camera-panel">
      <h3>Camera</h3>
      <img
        className="camera-img"
        src={`data:image/jpeg;base64,${cameraData.data}`}
        alt="robot camera"
      />
    </div>
  );
}
