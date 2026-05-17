"""Async WebSocket server using FastAPI + uvicorn.

Clients connect to ws://<host>:<port>/ws and receive broadcast JSON messages.
A /health HTTP endpoint is available for liveness checks.
"""

import asyncio
import json
import logging
from typing import Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class WebSocketServer:
    def __init__(self, host: str = '0.0.0.0', port: int = 8000):
        self._host = host
        self._port = port
        self._clients: Set[WebSocket] = set()
        self._app = self._build_app()
        self._server: uvicorn.Server | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _build_app(self) -> FastAPI:
        app = FastAPI(title='Puzzlebot Web Bridge')

        @app.get('/health')
        async def health():
            return JSONResponse({'status': 'ok', 'clients': len(self._clients)})

        @app.websocket('/ws')
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._clients.add(websocket)
            logger.info('WebSocket client connected. Total: %d', len(self._clients))
            try:
                while True:
                    # Keep connection alive; we only send, not receive.
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            except Exception as exc:
                logger.debug('WebSocket receive error: %s', exc)
            finally:
                self._clients.discard(websocket)
                logger.info('WebSocket client disconnected. Total: %d', len(self._clients))

        return app

    async def broadcast(self, payload: dict) -> None:
        if not self._clients:
            return
        message = json.dumps(payload)
        dead: Set[WebSocket] = set()
        for ws in list(self._clients):
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    def broadcast_sync(self, payload: dict) -> None:
        """Thread-safe bridge from ROS callbacks into the asyncio event loop."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(payload), self._loop)

    def start(self) -> None:
        """Launch uvicorn in a background thread so it doesn't block ROS spin."""
        import threading

        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            config = uvicorn.Config(
                self._app,
                host=self._host,
                port=self._port,
                log_level='warning',
                loop='asyncio',
            )
            self._server = uvicorn.Server(config)
            self._loop.run_until_complete(self._server.serve())

        thread = threading.Thread(target=_run, daemon=True, name='ws-server')
        thread.start()
        logger.info('WebSocket server started on %s:%d', self._host, self._port)

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
