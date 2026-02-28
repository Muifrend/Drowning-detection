"""Main runtime orchestrator for local real-time drowning detection."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

try:
    from . import config
    from .agent import Agent
    from .capture import CameraCapture
    from .display import build_display_frame, draw_minimap, draw_overlay
except ImportError:
    import config  # type: ignore
    from agent import Agent  # type: ignore
    from capture import CameraCapture  # type: ignore
    from display import build_display_frame, draw_minimap, draw_overlay  # type: ignore

# Ensure local agent_workflow directory can import detect.py directly.
sys.path.insert(0, os.path.dirname(__file__))

try:
    from detect import MODEL_LOADED, analyze_frame  # type: ignore
    if MODEL_LOADED:
        print("Real inference active")
    else:
        print("WARNING: Model failed to load - mock mode")
except ImportError as e:
    print(f"FATAL: detect.py import failed: {e}")
    MODEL_LOADED = False

    def analyze_frame(image):
        return {
            "detections": [],
            "threat_detected": False,
            "threat_count": 0,
        }


def pixel_to_pool(px: tuple, frame_shape: tuple) -> tuple:
    pool_x = (px[0] / frame_shape[1]) * config.POOL_W
    pool_y = (px[1] / frame_shape[0]) * config.POOL_L
    return (pool_x, pool_y)


def _normalize_detections(result) -> dict:
    if isinstance(result, dict):
        if isinstance(result.get("detections"), list):
            return result
        if isinstance(result.get("objects"), list):
            return {"detections": result["objects"]}
    if isinstance(result, list):
        return {"detections": result}
    return {"detections": []}


def _extract_positions(detections: dict, frame_shape: tuple) -> tuple[list, tuple | None]:
    swimmers: list[tuple[float, float]] = []
    victim: tuple[float, float] | None = None

    for det in detections.get("detections", []):
        bbox = det.get("bbox") or det.get("box")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        pos = pixel_to_pool((cx, cy), frame_shape)
        label = str(det.get("label", "")).lower()
        p_distress = float(det.get("p_distress", det.get("score", 0.0)))

        if label == "swimming":
            swimmers.append(pos)
        if victim is None and (label == "drowning" or p_distress > config.ALERT_THRESHOLD):
            victim = pos

    return swimmers, victim


def _capture_worker(capture: CameraCapture, shared: dict, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        frame = capture.get_frame()
        if frame is None:
            time.sleep(0.005)
            continue
        with shared["lock"]:
            shared["frame"] = frame
            shared["frame_count"] += 1


def _inference_worker(
    agent: Agent,
    camera: CameraCapture,
    shared: dict,
    stop_event: threading.Event,
    run_log_path: Path,
) -> None:
    last_inferred_id = -1
    last_logged_frame = -1

    while not stop_event.is_set():
        current_frame, current_id = camera.get_frame_with_id()

        if current_frame is None:
            time.sleep(0.01)
            continue

        if current_id - last_inferred_id < max(1, int(config.INFERENCE_EVERY)):
            time.sleep(0.005)
            continue

        last_inferred_id = current_id
        frame = current_frame
        frame_count = current_id

        # Convert to PIL
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Run inference
        raw = analyze_frame(pil_image)
        if not isinstance(raw, dict):
            raw = {"detections": [], "threat_detected": False, "threat_count": 0}
        detections = _normalize_detections(raw)
        detections["threat_detected"] = bool(raw.get("threat_detected", False))
        detections["threat_count"] = int(raw.get("threat_count", 0))

        # Extract p_distress from detections
        p_distress = 0.0
        if detections["detections"]:
            p_distress = max(float(d.get("p_distress", 0.0)) for d in detections["detections"])

        # Update shared state
        with shared["lock"]:
            shared["detections"] = detections
            shared["p_distress"] = p_distress

        # Run agent
        actions = agent.process(detections, p_distress)
        with shared["lock"]:
            shared["agent_actions"] = actions
            try:
                shared["agent_state"] = str(agent.current_state.value)
            except AttributeError:
                shared["agent_state"] = str(actions.get("state", "MONITOR"))

        swimmers: list[tuple[float, float]] = []
        victim: tuple[float, float] | None = None
        dispatch_plan = None
        # Run dispatch if threat
        if detections.get("threat_detected"):
            victim_px = None
            for d in detections["detections"]:
                if d.get("is_threat"):
                    bbox = d.get("bbox", [])
                    if len(bbox) == 4:
                        victim_px = ((int(bbox[0]) + int(bbox[2])) // 2, (int(bbox[1]) + int(bbox[3])) // 2)
                        break

            if victim_px:
                victim = pixel_to_pool(victim_px, frame.shape)
                for d in detections["detections"]:
                    if not d.get("is_threat"):
                        bbox = d.get("bbox", [])
                        if len(bbox) != 4:
                            continue
                        px = ((int(bbox[0]) + int(bbox[2])) // 2, (int(bbox[1]) + int(bbox[3])) // 2)
                        swimmers.append(pixel_to_pool(px, frame.shape))

                dispatch_plan = agent.dispatch(victim, swimmers)
                actions["explanation"] = dispatch_plan.get("explanation", actions.get("explanation", ""))
                ems_payload = agent.check_ems(actions.get("p_unresponsive", 0.0), actions.get("time_in_risk", 0.0))
                if ems_payload:
                    actions["ems_payload"] = ems_payload

        with shared["lock"]:
            shared["agent_actions"] = actions
            shared["dispatch_plan"] = dispatch_plan
            shared["swimmer_positions"] = swimmers
            shared["victim_pos"] = victim

        if frame_count > 0 and frame_count % 100 == 0 and frame_count != last_logged_frame:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "frame": frame_count,
                "state": actions.get("state", "MONITOR"),
                "p_distress": p_distress,
                "eta": None if dispatch_plan is None else float(dispatch_plan.get("eta_seconds", 0.0)),
                "dispatch_plan": dispatch_plan,
            }
            with run_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            last_logged_frame = frame_count


def _display_worker(
    agent: Agent,
    shared: dict,
    stop_event: threading.Event,
    headless: bool,
    max_seconds: float,
) -> None:
    start = time.time()
    target_dt = 1.0 / 15.0

    while not stop_event.is_set():
        loop_start = time.time()

        with shared["lock"]:
            frame = None if shared["frame"] is None else shared["frame"].copy()
            detections = dict(shared.get("detections", {}))
            actions = dict(shared.get("agent_actions", {}))
            dispatch_plan = None if shared.get("dispatch_plan") is None else dict(shared["dispatch_plan"])
            swimmers = list(shared.get("swimmer_positions", []))
            victim = shared.get("victim_pos")
            agent_state = shared.get("agent_state", "MONITOR")

        if frame is None:
            time.sleep(0.01)
            continue

        state = actions.get("state", "MONITOR")
        explanation = actions.get("explanation", "")
        overlay = draw_overlay(frame, detections, state, dispatch_plan, explanation)
        minimap = draw_minimap(
            swimmers,
            {"A": config.LIFEGUARD_A, "B": config.LIFEGUARD_B},
            victim,
            None if dispatch_plan is None else dispatch_plan.get("jump_point"),
            dispatch_plan,
            agent_state=agent_state,
        )
        combined = build_display_frame(overlay, minimap)

        if not headless:
            cv2.imshow("Lifeguard Agent", combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                stop_event.set()
            elif key in (ord("a"), ord("A")):
                agent.lifeguard_acknowledged()
            elif key in (ord("r"), ord("R")):
                agent.reset()

        if max_seconds > 0 and (time.time() - start) >= max_seconds:
            stop_event.set()

        elapsed = time.time() - loop_start
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=config.SOURCE)
    parser.add_argument("--pool_w", type=float, default=config.POOL_W)
    parser.add_argument("--pool_l", type=float, default=config.POOL_L)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    args = parser.parse_args()

    if not args.headless and not os.environ.get("DISPLAY"):
        print("No DISPLAY detected; switching to headless mode.")
        args.headless = True

    source = int(args.source) if isinstance(args.source, str) and args.source.isdigit() else args.source
    if isinstance(source, str):
        source = str(Path(source))

    # Update config at runtime using CLI overrides.
    config.POOL_W = args.pool_w
    config.POOL_L = args.pool_l

    model_mode = "paligemma2-live" if MODEL_LOADED else "fallback-empty"
    print(
        f"Starting Lifeguard Agent | source={source} | pool=({config.POOL_W}m x {config.POOL_L}m) | model={model_mode}"
    )

    capture = CameraCapture(source)
    agent = Agent()
    stop_event = threading.Event()

    shared = {
        "frame": None,
        "detections": {},
        "agent_actions": {},
        "dispatch_plan": None,
        "p_distress": 0.0,
        "swimmer_positions": [],
        "victim_pos": None,
        "frame_count": 0,
        "lock": threading.Lock(),
    }

    run_log_path = Path("run_log.json")

    threads = [
        threading.Thread(target=_capture_worker, args=(capture, shared, stop_event), daemon=True),
        threading.Thread(
            target=_inference_worker,
            args=(agent, capture, shared, stop_event, run_log_path),
            daemon=True,
        ),
        threading.Thread(
            target=_display_worker,
            args=(agent, shared, stop_event, bool(args.headless), float(args.max_seconds)),
            daemon=True,
        ),
    ]

    for t in threads:
        t.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Shutting down...")
        stop_event.set()
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)
        capture.stop()
        cv2.destroyAllWindows()

        with shared["lock"]:
            final_state = shared.get("agent_actions", {}).get("state", "MONITOR")
            total_frames = int(shared.get("frame_count", 0))
            final_dispatch = shared.get("dispatch_plan")

        print("Final state summary:")
        print(f"  frames={total_frames}")
        print(f"  state={final_state}")
        print(f"  dispatch_plan={final_dispatch}")


if __name__ == "__main__":
    main()
