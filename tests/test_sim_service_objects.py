"""CPU regressions for actor lifetime and disconnected UnrealCV fail-closed behavior."""
import sys
import threading
import time
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "action_aware_grpo" / "runtime"
sys.path.insert(0, str(ROOT))
from sim_service import Executor


class FakeUE:
    def __init__(self):
        # Legacy names are occupied even when an enumeration would omit them.
        self.positions = {"BP_Character_21": [1., 2., 3.], "BP_Character_22": [4., 5., 6.]}
        self.spawns = []
        self.hidden = set()
        self.appearances = {}

    def get_objects(self):
        raise AssertionError("Actor creation must not rely on an incomplete enumeration")

    def new_obj(self, cls, name, position):
        if name in self.positions:
            raise RuntimeError("UE fatal would occur: rename on top of occupied UObject")
        self.positions[name] = list(position)
        self.spawns.append((cls, name))

    def get_obj_location(self, name):
        return list(self.positions[name])

    def set_obj_location(self, name, position):
        self.positions[name] = list(position)

    def set_phy(self, name, state):
        assert state == 0

    def set_hide_obj(self, name):
        self.hidden.add(name)

    def set_show_obj(self, name):
        self.hidden.discard(name)

    def set_appearance(self, name, value):
        self.appearances[name] = value

    def set_obj_rotation(self, name, rotation):
        pass


def make_executor():
    ue = FakeUE()
    env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(
        unrealcv=ue, _service_settings={"object_settle_s": 0.0}))
    return Executor(env=env, task_json_root="unused", yaw_offset_deg=-180,
                    image_codec="jpeg", jpeg_quality=90,
                    resolve_task=None, capture_image=None, encode_image=None), ue


def info(kind):
    return {"use_obj": kind, "obj_id": 19, "obj_pos": [10., 20., 128.], "obj_rot": [0., 0., 0.]}


class ActorLifetimeTests(unittest.TestCase):
    def test_repeated_resets_reuse_only_required_actor(self):
        executor, ue = make_executor()
        first = executor._place_objects(info(1))
        self.assertEqual(len(ue.spawns), 1)  # no unused car
        self.assertEqual(ue.spawns[0][0], "bp_character_C")
        for _ in range(4):
            self.assertEqual(executor._place_objects(info(1))["name"], first["name"])
        self.assertEqual(len(ue.spawns), 1)
        car = executor._place_objects(info(2))
        self.assertEqual(len(ue.spawns), 2)
        self.assertEqual(ue.spawns[1][0], "BP_BaseCar_C")
        self.assertEqual(executor._place_objects(info(2))["name"], car["name"])
        self.assertEqual(len(ue.spawns), 2)
        self.assertIn(first["name"], ue.hidden)
        self.assertEqual(ue.appearances[first["name"]], 19)
        self.assertEqual(ue.appearances[car["name"]], 2)
        self.assertEqual(ue.positions[car["name"]][2], 0.)
        self.assertNotEqual(first["name"], "BP_Character_21")
        self.assertNotEqual(car["name"], "BP_Character_22")

    def test_no_object_hides_without_killz_or_spawn(self):
        executor, ue = make_executor()
        first = executor._place_objects(info(1))
        before = dict(ue.positions)
        executor._place_objects(None)
        self.assertIn(first["name"], ue.hidden)
        self.assertEqual(ue.positions, before)
        self.assertEqual(len(ue.spawns), 1)

    def test_missing_owned_actor_fails_without_respawning_occupied_name(self):
        executor, ue = make_executor()
        first = executor._place_objects(info(1))
        del ue.positions[first["name"]]
        with self.assertRaises(KeyError):
            executor._place_objects(info(1))
        self.assertEqual(len(ue.spawns), 1)

    def test_executors_do_not_share_actor_names(self):
        first, _ = make_executor()
        second, _ = make_executor()
        self.assertTrue(set(first._target_names.values()).isdisjoint(second._target_names.values()))

    def test_stalled_transport_times_out_and_fails_closed_but_allows_abort(self):
        executor, _ = make_executor()
        release = threading.Event()
        start = time.monotonic()
        try:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                executor.run_bounded(lambda: release.wait(5), 0.01)
            self.assertLess(time.monotonic() - start, 0.5)
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                executor.run_bounded(lambda: None, 1)
            executor.session = {"session_id": "owned"}
            self.assertTrue(executor.abort({"session_id": "owned"})["aborted"])
        finally:
            release.set()


if __name__ == "__main__":
    unittest.main(verbosity=2)
