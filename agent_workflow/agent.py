"""Agent state machine and dispatch planning for drowning response."""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

try:
    from . import config
except ImportError:
    import config  # type: ignore


class State(Enum):
    MONITOR = "MONITOR"
    ALERT = "ALERT"
    DISPATCH = "DISPATCH"
    ESCALATE = "ESCALATE"
    RESOLVED = "RESOLVED"


@dataclass
class WorldState:
    state: State = State.MONITOR
    p_distress: float = 0.0
    p_unresponsive: float = 0.0
    threat_detected: bool = False
    time_in_risk: float = 0.0


def compute_priority(p_distress, p_unresponsive, time_in_risk, eta):
    a, b, c, d = 0.4, 0.3, 0.2, 0.1
    return a * p_distress + b * p_unresponsive + c * time_in_risk - d * eta


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _sample_pool_boundary(n_points: int = 80) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    w = config.POOL_W
    l = config.POOL_L
    perimeter = 2.0 * (w + l)

    for i in range(n_points):
        s = (i / n_points) * perimeter
        if s < w:
            points.append((s, 0.0))
        elif s < w + l:
            points.append((w, s - w))
        elif s < 2 * w + l:
            points.append((w - (s - (w + l)), l))
        else:
            points.append((0.0, l - (s - (2 * w + l))))
    return points


def _point_to_segment_dist(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx * vx + vy * vy
    if vv <= 1e-9:
        return _euclidean(p, a)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    proj = (ax + t * vx, ay + t * vy)
    return _euclidean(p, proj)


def _gaussian_penalty(distance: float, sigma: float = 1.5, amplitude: float = 2.0) -> float:
    return amplitude * math.exp(-(distance * distance) / (2.0 * sigma * sigma))


def _lifeguard_label(pos: tuple[float, float]) -> str:
    if _euclidean(pos, config.LIFEGUARD_A) <= _euclidean(pos, config.LIFEGUARD_B):
        if _euclidean(pos, config.LIFEGUARD_A) < 1e-6:
            return "A"
    if _euclidean(pos, config.LIFEGUARD_B) < 1e-6:
        return "B"
    return "A" if _euclidean(pos, config.LIFEGUARD_A) <= _euclidean(pos, config.LIFEGUARD_B) else "B"


def select_jump_point(
    lifeguard_pos: tuple,
    victim_pos: tuple,
    swimmer_positions: list,
) -> dict:
    best: dict | None = None
    lg = _lifeguard_label((float(lifeguard_pos[0]), float(lifeguard_pos[1])))

    for j in _sample_pool_boundary(80):
        deck_time = _euclidean(lifeguard_pos, j) / max(config.DECK_SPEED, 1e-6)
        swim_dist = _euclidean(j, victim_pos)

        crowd_penalty = 0.0
        for swimmer in swimmer_positions:
            d = _point_to_segment_dist((float(swimmer[0]), float(swimmer[1])), j, victim_pos)
            crowd_penalty += _gaussian_penalty(d, sigma=1.5, amplitude=2.0)

        swim_time = (swim_dist / max(config.SWIM_SPEED, 1e-6)) * (1.0 + crowd_penalty)
        total_time = deck_time + swim_time

        candidate = {
            "jump_point": (float(j[0]), float(j[1])),
            "eta_seconds": float(total_time),
            "deck_time": float(deck_time),
            "swim_time": float(swim_time),
            "lifeguard": lg,
        }

        if best is None or candidate["eta_seconds"] < best["eta_seconds"]:
            best = candidate

    return best or {
        "jump_point": (0.0, 0.0),
        "eta_seconds": float("inf"),
        "deck_time": float("inf"),
        "swim_time": float("inf"),
        "lifeguard": lg,
    }


class Agent:
    def __init__(self) -> None:
        self.world = WorldState()
        self._lock = threading.Lock()

        self._distress_window: deque[float] = deque(maxlen=config.TEMPORAL_WINDOW)
        self._risk_start_ts: float | None = None
        self._no_threat_start_ts: float | None = None
        self._unresponsive_start_ts: float | None = None

        self._ack_requested = False

        self.last_dispatch_plan: dict | None = None
        self.last_victim_pos: tuple[float, float] | None = None
        self.last_swimmer_positions: list[tuple[float, float]] = []
        self.last_ems_payload: dict | None = None
        self._ems_triggered_for_incident = False

    @staticmethod
    def _extract_objects(detections: dict | list | None) -> list[dict]:
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

    def _transition(self, new_state: State) -> None:
        if self.world.state == new_state:
            return
        old = self.world.state
        self.world.state = new_state
        print(f"STATE TRANSITION: {old.value} -> {new_state.value}")

    def lifeguard_acknowledged(self) -> None:
        with self._lock:
            self._ack_requested = True

    def reset(self) -> None:
        with self._lock:
            self.world = WorldState()
            self._distress_window.clear()
            self._risk_start_ts = None
            self._no_threat_start_ts = None
            self._unresponsive_start_ts = None
            self.last_dispatch_plan = None
            self.last_victim_pos = None
            self.last_swimmer_positions = []
            self.last_ems_payload = None
            self._ems_triggered_for_incident = False
            self._ack_requested = False
            print("Agent reset")

    def process(self, detections: dict | list | None, frame_count: int) -> dict:
        with self._lock:
            now = datetime.now(timezone.utc).timestamp()
            objects = self._extract_objects(detections)

            p_distress_candidates = []
            p_unresponsive_candidates = []
            threat_detected = False
            victim_bbox = None

            for obj in objects:
                label = str(obj.get("label", "")).lower()
                p_distress = float(obj.get("p_distress", obj.get("score", 0.0)))
                p_unresponsive = float(obj.get("p_unresponsive", 0.0))

                p_distress_candidates.append(p_distress)
                p_unresponsive_candidates.append(p_unresponsive)

                if label == "drowning" or p_distress > config.ALERT_THRESHOLD:
                    threat_detected = True

                if victim_bbox is None and label == "drowning":
                    victim_bbox = obj.get("bbox")

            raw_p_distress = max(p_distress_candidates) if p_distress_candidates else 0.0
            raw_p_unresponsive = max(p_unresponsive_candidates) if p_unresponsive_candidates else 0.0

            self._distress_window.append(raw_p_distress)
            p_distress = sum(self._distress_window) / max(len(self._distress_window), 1)
            p_unresponsive = raw_p_unresponsive

            self.world.p_distress = p_distress
            self.world.p_unresponsive = p_unresponsive
            self.world.threat_detected = threat_detected

            if threat_detected:
                if self._risk_start_ts is None:
                    self._risk_start_ts = now
                self.world.time_in_risk = now - self._risk_start_ts
                self._no_threat_start_ts = None
            else:
                self.world.time_in_risk = 0.0
                self._risk_start_ts = None
                if self._no_threat_start_ts is None:
                    self._no_threat_start_ts = now
                self._unresponsive_start_ts = None
                self._ems_triggered_for_incident = False

            if p_unresponsive > 0.7 and threat_detected:
                if self._unresponsive_start_ts is None:
                    self._unresponsive_start_ts = now
            else:
                self._unresponsive_start_ts = None

            if self._ack_requested:
                self._transition(State.MONITOR)
                self._ack_requested = False

            if self._no_threat_start_ts is not None and (now - self._no_threat_start_ts) >= 3.0:
                self._transition(State.RESOLVED)
            else:
                if self.world.state == State.MONITOR and p_distress > config.ALERT_THRESHOLD:
                    self._transition(State.ALERT)
                if self.world.state == State.ALERT and p_distress > config.DISPATCH_THRESHOLD:
                    self._transition(State.DISPATCH)
                if self.world.state == State.DISPATCH:
                    if self._unresponsive_start_ts is not None and (now - self._unresponsive_start_ts) > config.UNRESPONSIVE_SECONDS:
                        self._transition(State.ESCALATE)
                    elif p_distress > config.ESCALATE_THRESHOLD:
                        self._transition(State.ESCALATE)

            explanation = (
                f"state={self.world.state.value} p_distress={p_distress:.2f} "
                f"p_unresponsive={p_unresponsive:.2f} risk={self.world.time_in_risk:.1f}s"
            )

            return {
                "state": self.world.state.value,
                "p_distress": p_distress,
                "p_unresponsive": p_unresponsive,
                "time_in_risk": self.world.time_in_risk,
                "threat_detected": threat_detected,
                "explanation": explanation,
                "priority": compute_priority(p_distress, p_unresponsive, self.world.time_in_risk, eta=0.0),
                "frame_count": frame_count,
            }

    def dispatch(self, victim_pos, swimmer_positions) -> dict:
        with self._lock:
            self.last_victim_pos = (float(victim_pos[0]), float(victim_pos[1]))
            self.last_swimmer_positions = [
                (float(p[0]), float(p[1])) for p in (swimmer_positions or [])
            ]

            plan_a = select_jump_point(config.LIFEGUARD_A, self.last_victim_pos, self.last_swimmer_positions)
            plan_b = select_jump_point(config.LIFEGUARD_B, self.last_victim_pos, self.last_swimmer_positions)

            best = plan_a if plan_a["eta_seconds"] <= plan_b["eta_seconds"] else plan_b
            best = dict(best)
            best["state"] = self.world.state.value
            best["dispatch_ts"] = time.time()
            best[
                "explanation"
            ] = (
                f"Dispatching Lifeguard {best['lifeguard']}: ETA {best['eta_seconds']:.1f}s via jump at "
                f"({best['jump_point'][0]:.1f},{best['jump_point'][1]:.1f}m). "
                f"p_distress={self.world.p_distress:.2f} sustained {self.world.time_in_risk:.1f}s."
            )

            self.last_dispatch_plan = best
            return best

    def check_ems(self, p_unresponsive, time_in_risk) -> dict | None:
        with self._lock:
            p_unresponsive = float(p_unresponsive)
            time_in_risk = float(time_in_risk)
            p_distress = float(self.world.p_distress)

            triggered = (p_unresponsive > 0.7 and time_in_risk > 5.0) or (
                p_distress > 0.9 and time_in_risk > 15.0
            )

            if not triggered or self._ems_triggered_for_incident:
                return None

            dispatch_plan = self.last_dispatch_plan or {}
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "victim_pool_coords": self.last_victim_pos,
                "p_distress": p_distress,
                "p_unresponsive": p_unresponsive,
                "dispatched_lifeguard": dispatch_plan.get("lifeguard"),
                "eta_seconds": dispatch_plan.get("eta_seconds"),
                "status": "EMS_TRIGGERED",
            }

            alerts_path = Path("alerts.log")
            with alerts_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")

            print("\033[91mEMS TRIGGERED\033[0m")
            self._ems_triggered_for_incident = True
            self.last_ems_payload = payload
            return payload
