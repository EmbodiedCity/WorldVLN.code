"""Serialized simulator executor with observed poses and a versioned SE(3) contract."""
from __future__ import annotations

import json
import math
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from .pose_math import ACTION_CONTRACT, compose_body_delta, pose_to_matrix
except ImportError:
    from pose_math import ACTION_CONTRACT, compose_body_delta, pose_to_matrix


class SessionConflict(ValueError):
    pass


def read_pose(env):
    ue = env.unwrapped.unrealcv
    player = env.unwrapped.player_list[0]
    pose = list(ue.get_obj_location(player)) + list(ue.get_obj_rotation(player))
    pose_to_matrix(pose)
    return [float(value) for value in pose]


def _angle_error(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


class Executor:
    def __init__(self, *, env, task_json_root, yaw_offset_deg, image_codec,
                 jpeg_quality, resolve_task, capture_image, encode_image):
        self.env = env
        self.task_json_root = task_json_root
        self.yaw_offset_deg = float(yaw_offset_deg)
        self.image_codec = image_codec
        self.jpeg_quality = int(jpeg_quality)
        self.resolve_task = resolve_task
        self.capture_image = capture_image
        self.encode_image = encode_image
        self.settings = getattr(env.unwrapped, "_service_settings", {})
        self.session = None
        self.lock = threading.Lock()
        # Own target actors for this executor's lifetime. /objects enumeration
        # can omit a hidden/pending-kill actor even while its UObject name is
        # occupied; spawning a second actor with that name fatally crashes UE.
        prefix = uuid.uuid4().hex[:12]
        self._target_names = {1: f"WorldVLN_Character21_{prefix}", 2: f"WorldVLN_Character22_{prefix}"}
        self._spawned_targets = set()
        self._transport_unavailable = False

    def _setting(self, key, default):
        return self.settings.get(key, default)

    def run_bounded(self, operation, timeout_s, *args):
        """UnrealCV's installed request(timeout=...) ignores its timeout.

        A stalled socket must not hold the service lock forever. On a timeout,
        permanently fail this executor closed; a still-blocked daemon must never
        share the socket with a new rollout. /abort can still release ownership.
        """
        if self._transport_unavailable:
            raise RuntimeError("simulator transport unavailable; restart service/UE")
        finished = threading.Event()
        result = {}
        def invoke():
            try:
                result["value"] = operation(*args)
            except BaseException as error:
                result["error"] = error
            finally:
                finished.set()
        threading.Thread(target=invoke, daemon=True).start()
        if not finished.wait(float(timeout_s)):
            self._transport_unavailable = True
            raise RuntimeError("simulator operation timed out; restart service/UE before continuing")
        if "error" in result:
            raise result["error"]
        return result["value"]

    def _capture(self):
        # Drop rendered frames after the asynchronous actor setters/readback.
        image = None
        for _ in range(int(self._setting("stabilization_renders", 2))):
            time.sleep(float(self._setting("stabilization_sleep_s", 0.05)))
            image = self.capture_image(self.env, cam_id=0)
        if image is None or min(image.size) <= 0:
            raise RuntimeError("UnrealCV returned an empty image")
        return self.encode_image(image, codec=self.image_codec, quality=self.jpeg_quality)

    def _execute(self, target):
        pose_to_matrix(target)
        ue = self.env.unwrapped.unrealcv
        player = self.env.unwrapped.player_list[0]
        ue.set_obj_location(player, target[:3])
        # Collection's drone BP controller expects yaw-180. The object readback
        # is already in dataset world axes; never add this offset to observations.
        ue.set_rotation(player, target[4] + self.yaw_offset_deg)
        # The controller handles yaw; the native setter additionally supports
        # roll/pitch where available. Unsupported orientation fails readback.
        ue.set_obj_rotation(player, target[3:6])
        deadline = time.monotonic() + float(self._setting("pose_timeout_s", 5.0))
        observed = None
        while time.monotonic() < deadline:
            observed = read_pose(self.env)
            position_error = math.dist(observed[:3], target[:3])
            angle_error = max(_angle_error(observed[i], target[i]) for i in (3, 4, 5))
            if (position_error <= float(self._setting("pose_pos_tol_cm", 2.0)) and
                    angle_error <= float(self._setting("pose_angle_tol_deg", 1.5))):
                image = self._capture()
                # Read again after the render barrier; return what the UE actor
                # actually reached, never the intended command or a cumsum.
                rendered_pose = read_pose(self.env)
                rendered_position_error = math.dist(rendered_pose[:3], target[:3])
                rendered_angle_error = max(_angle_error(rendered_pose[i], target[i]) for i in (3, 4, 5))
                if (rendered_position_error <= float(self._setting("pose_pos_tol_cm", 2.0)) and
                        rendered_angle_error <= float(self._setting("pose_angle_tol_deg", 1.5))):
                    return rendered_pose, image
            time.sleep(0.025)
        raise RuntimeError(f"UE pose did not settle: commanded={target}, observed={observed}")

    def _place_objects(self, info):
        kind = 0 if info is None else int(info.get("use_obj", 0))
        if kind not in (0, 1, 2):
            raise ValueError("use_obj must be 0, 1 or 2")
        ue = self.env.unwrapped.unrealcv
        # Hide existing actors without dropping them below KillZ or enabling
        # car physics. Only create the kind actually required by the task.
        for existing_kind in self._spawned_targets:
            existing_name = self._target_names[existing_kind]
            location = [float(v) for v in ue.get_obj_location(existing_name)]
            if len(location) != 3 or not all(math.isfinite(v) for v in location):
                raise RuntimeError(f"Owned target actor is unavailable: {existing_name}")
            ue.set_phy(existing_name, 0)
            ue.set_hide_obj(existing_name)
        if kind == 0:
            return None if info is None else {"use_obj": 0}
        position = [float(v) for v in info["obj_pos"][:3]]
        rotation = [float(v) for v in info.get("obj_rot", [0, 0, 0])[:3]]
        name = self._target_names[kind]
        if kind == 2:
            position[2] = 0.0
        if kind not in self._spawned_targets:
            cls = "bp_character_C" if kind == 1 else "BP_BaseCar_C"
            ue.new_obj(cls, name, position)
            ue.set_phy(name, 0)
            # A synchronous read fences asynchronous spawn commands. If UE no
            # longer owns it later, fail rather than respawn the occupied name.
            created = [float(v) for v in ue.get_obj_location(name)]
            if len(created) != 3 or not all(math.isfinite(v) for v in created):
                raise RuntimeError(f"Task target actor could not be created: {name}")
            self._spawned_targets.add(kind)
        ue.set_appearance(name, int(info["obj_id"]) if kind == 1 else 2)
        ue.set_phy(name, 0)
        ue.set_obj_location(name, position)
        ue.set_obj_rotation(name, rotation)
        ue.set_show_obj(name)
        time.sleep(float(self._setting("object_settle_s", 0.25)))
        actual = [float(v) for v in ue.get_obj_location(name)]
        if math.dist(actual, position) > float(self._setting("object_verify_tol_cm", 50.0)):
            raise RuntimeError(f"Task object placement failed: {name}, wanted={position}, actual={actual}")
        return {"name": name, "use_obj": kind, "commanded_position": position, "observed_position": actual}

    def health(self):
        if self._transport_unavailable:
            raise RuntimeError("simulator transport unavailable; restart service/UE")
        pose = read_pose(self.env)
        self._capture()
        session = self.session
        return {
            "status": "ok", "ue_connected": True,
            "world_pose": pose, "action_contract": ACTION_CONTRACT, "world_poses_source": "simulator_observed",
            "active_session": session["session_id"] if session else "",
            "frame_index": session["frame_index"] if session else 0,
        }

    def reset(self, payload):
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise ValueError("session_id is required")
        active = self.session
        if active and not active["done"]:
            raise SessionConflict(f"active session {active['session_id']}; finish it or abort the matching session")
        prompt, pose, info, task_id, task_name = self.resolve_task(payload, self.task_json_root)
        pose_to_matrix(pose)
        max_frames = int(payload.get("max_frames", 49))
        if max_frames < 1 or max_frames > 10000:
            raise ValueError("max_frames must be between 1 and 10000")
        # Invalidate state before mutating UE; a failed reset cannot keep an old
        # session whose state no longer corresponds to the simulator.
        self.session = None
        self.env.reset()
        ue = self.env.unwrapped.unrealcv
        player = self.env.unwrapped.player_list[0]
        ue.set_viewport(player)
        ue.set_phy(player, 0)
        object_result = self._place_objects(info)
        observed, image = self._execute(pose)
        self.session = {
            "session_id": session_id, "task_id": str(task_id), "pose": observed,
            "frame_index": 1, "max_frames": max_frames, "done": max_frames <= 1,
            "failed": False,
        }
        print("[pose_readback] " + json.dumps({"session_id": session_id, "frame_indices": [1],
              "pose_errors": [self._pose_error(observed, pose)]}), flush=True)
        return {
            "session_id": session_id, "task_id": str(task_id),
            "task_json_name": str(task_name), "instruction": prompt,
            "images_base64": [image], "frame_indices": [1],
            "world_poses": [observed], "commanded_world_poses": [list(pose)],
            "done": max_frames <= 1, "object_verification": object_result,
            "action_contract": ACTION_CONTRACT, "world_poses_source": "simulator_observed",
            "pose_errors": [self._pose_error(observed, pose)],
        }

    @staticmethod
    def _pose_error(observed, commanded):
        return {
            "position_cm": math.dist(observed[:3], commanded[:3]),
            "roll_deg": _angle_error(observed[3], commanded[3]),
            "yaw_deg": _angle_error(observed[4], commanded[4]),
            "pitch_deg": _angle_error(observed[5], commanded[5]),
        }

    def abort(self, payload):
        """Release only the caller's session; never move or reset the UE actor."""
        session = self.session
        if session is None:
            return {"status": "ok", "aborted": False, "active_session": ""}
        session_id = str(payload.get("session_id", "")).strip()
        if session_id != session["session_id"]:
            raise SessionConflict("active session mismatch; abort requires the current session_id")
        self.session = None
        print("[env_service] aborted session " + session_id, flush=True)
        return {"status": "ok", "aborted": True, "session_id": session_id, "active_session": ""}

    def step(self, payload):
        session = self.session
        if session is None:
            raise SessionConflict("no active session; call /reset first")
        if str(payload.get("session_id", "")) != session["session_id"]:
            raise SessionConflict("active session mismatch")
        if session["failed"]:
            raise SessionConflict("session failed during UE execution; reset is required")
        if session["done"]:
            raise SessionConflict("session has already completed")
        if payload.get("action_contract", ACTION_CONTRACT) != ACTION_CONTRACT:
            raise ValueError("action_contract mismatch; expected previous_body SE3 cm/deg")
        actions = payload.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("actions must be a non-empty list")
        if len(actions) > session["max_frames"] - session["frame_index"]:
            raise ValueError("actions exceed the remaining episode frame budget")
        # Validate the whole batch before applying any action.
        for action in actions:
            pose_to_matrix(action)
        images, indices, poses, commanded = [], [], [], []
        try:
            for action in actions:
                target = compose_body_delta(session["pose"], action)
                observed, image = self._execute(target)
                session["pose"] = observed
                session["frame_index"] += 1
                images.append(image)
                indices.append(session["frame_index"])
                poses.append(observed)
                commanded.append(target)
        except Exception:
            session["failed"] = True
            raise
        session["done"] = session["frame_index"] >= session["max_frames"]
        print("[pose_readback] " + json.dumps({"session_id": session["session_id"], "frame_indices": indices,
              "pose_errors": [self._pose_error(actual, target) for actual, target in zip(poses, commanded)]}), flush=True)
        return {
            "session_id": session["session_id"], "task_id": session["task_id"],
            "images_base64": images, "frame_indices": indices, "world_poses": poses,
            "commanded_world_poses": commanded, "done": session["done"],
            "action_contract": ACTION_CONTRACT, "world_poses_source": "simulator_observed",
            "pose_errors": [self._pose_error(actual, target) for actual, target in zip(poses, commanded)],
        }


def run_env_service(*, host, port, env, task_json_root, yaw_offset_deg,
                    action_frame, body_apply_order, image_codec, jpeg_quality,
                    resolve_task, capture_image, encode_image):
    if action_frame != "body" or body_apply_order != "trans_first":
        raise ValueError("service requires --action_frame body --body_apply_order trans_first")
    executor = Executor(
        env=env, task_json_root=task_json_root, yaw_offset_deg=yaw_offset_deg,
        image_codec=image_codec, jpeg_quality=jpeg_quality, resolve_task=resolve_task,
        capture_image=capture_image, encode_image=encode_image,
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, message, *args):
            print("[env_service] " + message % args, flush=True)

        def respond(self, code, payload):
            raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path.rstrip("/") != "/health":
                self.respond(404, {"error": "unknown path"})
                return
            if not executor.lock.acquire(timeout=1.0):
                self.respond(503, {"status": "busy", "error": "simulator operation in progress"})
                return
            try:
                try:
                    self.respond(200, executor.run_bounded(executor.health, 5.0))
                except Exception as error:
                    self.respond(503, {"status": "error", "ue_connected": False, "error": str(error)})
            finally:
                executor.lock.release()

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 8 * 1024 * 1024:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(size).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("payload must be a JSON object")
                with executor.lock:
                    if self.path.rstrip("/") == "/reset":
                        result = executor.run_bounded(executor.reset, 30.0, payload)
                    elif self.path.rstrip("/") == "/step_actions":
                        count = len(payload.get("actions", []))
                        timeout = max(30.0, count * (float(executor._setting("pose_timeout_s", 5.0)) + 1.0))
                        result = executor.run_bounded(executor.step, timeout, payload)
                    elif self.path.rstrip("/") == "/abort":
                        result = executor.abort(payload)
                    else:
                        self.respond(404, {"error": "unknown path"})
                        return
                    self.respond(200, result)
            except SessionConflict as error:
                self.respond(409, {"error": str(error)})
            except (ValueError, KeyError, TypeError) as error:
                self.respond(400, {"error": str(error)})
            except Exception as error:
                self.respond(503, {"error": str(error)})

    # Validate the simulator before claiming readiness or opening the socket.
    executor.run_bounded(executor.health, 5.0)
    server = ThreadingHTTPServer((str(host), int(port)), Handler)
    print(f"[env_service] listening on http://{host}:{port}; {ACTION_CONTRACT}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        env.close()
