import json
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

logger = logging.getLogger("uvicorn.error")

router = APIRouter(tags=["Live pose"])


def _parse_metrics_message(raw: str) -> tuple[Optional[dict[str, Any]], Optional[str], bool]:
    """
    Returns (payload, error, is_ping).
    Payload matches BiomechanicsFrame / pose_capture.py frame_data keys.
    """
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"invalid_json: {e}", False

    if not isinstance(msg, dict):
        return None, "expected_object", False

    if msg.get("type") == "ping":
        return None, None, True

    if msg.get("type") == "metrics":
        payload = msg.get("payload")
        if not isinstance(payload, dict):
            return None, "missing_payload", False
        return payload, None, False

    # Bare frame for simple clients
    if "elbow_angle" in msg and "hip_angle" in msg:
        return msg, None, False

    return None, "unknown_message", False


@router.websocket("/ws/live-pose")
async def live_pose_websocket(
    websocket: WebSocket,
    session_id: Optional[int] = Query(None),
):
    """
    Client sends: {"type":"metrics","payload":{...}} (same fields as pose_capture.py overlay).
    Server echoes metrics so the web UI can drive Real-Time Analysis from the socket.
    """
    await websocket.accept()
    await websocket.send_json(
        {
            "type": "hello",
            "session_id": session_id,
            "ts": time.time(),
        }
    )

    try:
        while True:
            raw = await websocket.receive_text()
            payload, err, is_ping = _parse_metrics_message(raw)

            if is_ping:
                await websocket.send_json({"type": "pong", "ts": time.time()})
                continue

            if err:
                await websocket.send_json({"type": "error", "detail": err})
                continue

            if payload is None:
                continue

            await websocket.send_json(
                {
                    "type": "metrics",
                    "session_id": session_id,
                    "payload": payload,
                    "ts": time.time(),
                }
            )
    except WebSocketDisconnect:
        logger.info("live_pose ws disconnected session_id=%s", session_id)
    except Exception as e:
        logger.exception("live_pose ws error: %s", e)

