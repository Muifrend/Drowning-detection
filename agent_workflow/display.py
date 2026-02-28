"""Rendering utilities for live overlay and tactical minimap."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

try:
    from . import config
except ImportError:
    import config  # type: ignore


STATE_COLORS = {
    "MONITOR": (255, 255, 255),
    "ALERT": (0, 255, 255),
    "DISPATCH": (0, 165, 255),
    "ESCALATE": (0, 0, 255),
    "RESOLVED": (0, 255, 0),
}


def _normalize_detections(detections: dict | list | None) -> list[dict]:
    if detections is None:
        return []
    if isinstance(detections, list):
        return detections
    if isinstance(detections, dict):
        if isinstance(detections.get("detections"), list):
            return detections["detections"]
        if isinstance(detections.get("objects"), list):
            return detections["objects"]
    return []


def _draw_text_lines(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.6,
    thickness: int = 2,
    max_chars: int = 80,
) -> None:
    x0, y0 = origin
    words = text.split()
    lines: list[str] = []
    current = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) <= max_chars:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))

    for idx, line in enumerate(lines):
        y = y0 + idx * 24
        cv2.putText(
            image,
            line,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_overlay(
    frame: np.ndarray,
    detections: dict,
    agent_state: str,
    dispatch_plan: dict | None,
    explanation: str,
) -> np.ndarray:
    output = frame.copy()
    items = _normalize_detections(detections)

    for det in items:
        bbox = det.get("bbox") or det.get("box")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        label = str(det.get("label", "unknown")).lower()
        p_distress = float(det.get("p_distress", det.get("score", 0.0)))

        color = (0, 255, 0)
        if label == "drowning":
            color = (0, 0, 255)
        elif label == "swimming":
            color = (0, 255, 0)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        text = f"{label} p={p_distress:.2f}"
        text_y = max(20, y1 - 10)
        cv2.putText(output, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    state_text = str(agent_state).upper()
    state_color = STATE_COLORS.get(state_text, (255, 255, 255))
    cv2.putText(output, state_text, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, state_color, 3, cv2.LINE_AA)

    if state_text == "ESCALATE":
        # Flashing full-frame danger overlay while in ESCALATE.
        if int(time.time() * 4) % 2 == 0:
            red = np.zeros_like(output)
            red[:, :] = (0, 0, 255)
            output = cv2.addWeighted(output, 0.7, red, 0.3, 0.0)

    if dispatch_plan and dispatch_plan.get("eta_seconds") is not None:
        eta = float(dispatch_plan["eta_seconds"])
        dispatch_ts = dispatch_plan.get("dispatch_ts")
        if dispatch_ts is not None:
            eta = max(0.0, eta - (time.time() - float(dispatch_ts)))
        else:
            eta = max(0.0, eta)
        eta_text = f"ETA: {eta:.1f}s"
        (w, _), _ = cv2.getTextSize(eta_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        cv2.putText(
            output,
            eta_text,
            (max(10, output.shape[1] - w - 20), 40),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if explanation:
        _draw_text_lines(
            output,
            explanation,
            (20, max(40, output.shape[0] - 60)),
            (0, 255, 255),
            scale=0.6,
            thickness=2,
            max_chars=100,
        )

    return output


def _pool_to_minimap(coord: tuple[float, float]) -> tuple[int, int]:
    pad = 20
    x = int(pad + (coord[0] / max(config.POOL_W, 1e-6)) * (config.MINIMAP_W - 2 * pad))
    y = int(pad + (coord[1] / max(config.POOL_L, 1e-6)) * (config.MINIMAP_H - 2 * pad))
    return x, y


def _draw_dashed_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 10,
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = int(math.hypot(dx, dy))
    if length == 0:
        return

    for i in range(0, length, dash_len * 2):
        a = i / length
        b = min(i + dash_len, length) / length
        p1 = (int(start[0] + dx * a), int(start[1] + dy * a))
        p2 = (int(start[0] + dx * b), int(start[1] + dy * b))
        cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)


def draw_minimap(
    swimmer_positions: list,
    lifeguard_positions: dict,
    victim_pos: tuple | None,
    jump_point: tuple | None,
    dispatch_plan: dict | None,
    agent_state: str = "MONITOR",
) -> np.ndarray:
    canvas = np.zeros((config.MINIMAP_H, config.MINIMAP_W, 3), dtype=np.uint8)

    pad = 20
    cv2.rectangle(
        canvas,
        (pad, pad),
        (config.MINIMAP_W - pad, config.MINIMAP_H - pad),
        (255, 255, 255),
        2,
    )

    for swimmer in swimmer_positions or []:
        sx, sy = _pool_to_minimap((float(swimmer[0]), float(swimmer[1])))
        cv2.circle(canvas, (sx, sy), 4, (255, 0, 0), -1)

    for label, coord in (lifeguard_positions or {}).items():
        lx, ly = _pool_to_minimap((float(coord[0]), float(coord[1])))
        cv2.circle(canvas, (lx, ly), 6, (0, 255, 255), -1)
        cv2.putText(canvas, str(label), (lx + 8, ly - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    if victim_pos is not None:
        vx, vy = _pool_to_minimap((float(victim_pos[0]), float(victim_pos[1])))
        radius = 5 + int(2 * (1 + math.sin(time.time() * 5.0)))
        cv2.circle(canvas, (vx, vy), radius, (0, 0, 255), -1)

    if jump_point is not None:
        jx, jy = _pool_to_minimap((float(jump_point[0]), float(jump_point[1])))
        cv2.line(canvas, (jx - 7, jy - 7), (jx + 7, jy + 7), (0, 255, 0), 2)
        cv2.line(canvas, (jx - 7, jy + 7), (jx + 7, jy - 7), (0, 255, 0), 2)

    if dispatch_plan:
        lifeguard = dispatch_plan.get("lifeguard")
        if lifeguard in lifeguard_positions and jump_point is not None:
            lxy = _pool_to_minimap(tuple(lifeguard_positions[lifeguard]))
            jxy = _pool_to_minimap((float(jump_point[0]), float(jump_point[1])))
            _draw_dashed_line(canvas, lxy, jxy, (255, 255, 255), thickness=2)

        if jump_point is not None and victim_pos is not None:
            jxy = _pool_to_minimap((float(jump_point[0]), float(jump_point[1])))
            vxy = _pool_to_minimap((float(victim_pos[0]), float(victim_pos[1])))
            _draw_dashed_line(canvas, jxy, vxy, (255, 0, 0), thickness=2)

    eta_text = f"ETA: {dispatch_plan['eta_seconds']:.1f}s" if dispatch_plan else "ETA: --"
    lg_text = f"LG-{dispatch_plan['lifeguard']}" if dispatch_plan else "LG: --"
    footer = f"{eta_text} | {lg_text} | State: {agent_state}"
    cv2.putText(
        canvas,
        footer,
        (12, config.MINIMAP_H - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return canvas


def build_display_frame(video_frame: np.ndarray, minimap_frame: np.ndarray) -> np.ndarray:
    video_w = max(1, config.FRAME_W - config.MINIMAP_W)
    video_resized = cv2.resize(video_frame, (video_w, config.FRAME_H))
    minimap_resized = cv2.resize(minimap_frame, (config.MINIMAP_W, config.FRAME_H))
    return np.concatenate([video_resized, minimap_resized], axis=1)
