/**
 * Manages the WebSocket connection to the ROS bridge.
 * Automatically reconnects on disconnect.
 * Parses incoming JSON and dispatches to registered handlers.
 */

const RECONNECT_DELAY_MS = 2000;
const MAX_RECONNECT_DELAY_MS = 30000;

export function createWebSocketClient(url, handlers = {}) {
  let ws = null;
  let reconnectDelay = RECONNECT_DELAY_MS;
  let stopped = false;

  function connect() {
    if (stopped) return;
    ws = new WebSocket(url);

    ws.onopen = () => {
      reconnectDelay = RECONNECT_DELAY_MS;
      handlers.onConnect?.();
    };

    ws.onclose = () => {
      handlers.onDisconnect?.();
      if (!stopped) {
        setTimeout(connect, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY_MS);
      }
    };

    ws.onerror = () => {
      // onclose fires after onerror — reconnect handled there.
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handlers.onMessage?.(msg);
      } catch {
        // Ignore malformed JSON.
      }
    };
  }

  connect();

  return {
    close() {
      stopped = true;
      ws?.close();
    },
    send(data) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(data));
      }
    },
  };
}
