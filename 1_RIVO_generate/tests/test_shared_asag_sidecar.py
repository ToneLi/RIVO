import threading
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from grpo_plugin_service import build_app


class _FakeProbe:
    max_cached_sessions = 3

    def __init__(self):
        self.analyze_calls = []
        self.released = []

    def estimated_kv_gib_per_session(self):
        return 1.25

    def analyze(
        self,
        history_token_ids,
        probe_token_ids,
        previous_span,
        current_span,
        session_id,
    ):
        self.analyze_calls.append(
            (
                history_token_ids,
                probe_token_ids,
                previous_span,
                current_span,
                session_id,
            )
        )
        return {
            "entropy": 2.0,
            "previous_attention": 0.1,
            "current_attention": 0.2,
        }

    def release_session(self, session_id):
        self.released.append(session_id)
        return True


class _FakeEngine:
    def __init__(self):
        self.checkpoint = Path("/checkpoint")
        self.checkpoint_step = 50
        self.alpha = 1.0
        self.lock = threading.RLock()
        self.asag_probe = _FakeProbe()

    def control(self, request):
        return {"action": "CONTINUE"}

    def reroute(self, request):
        return {"query": "query"}


class SharedASAGSidecarTests(unittest.TestCase):
    def setUp(self):
        self.engine = _FakeEngine()
        self.client = TestClient(build_app(self.engine))

    def test_health_reports_shared_host_asag(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["shared_host_asag"])
        self.assertEqual(payload["asag_max_cached_sessions"], 3)

    def test_analyze_and_release_use_sidecar_probe(self):
        response = self.client.post(
            "/analyze",
            json={
                "session_id": "worker-0:q1",
                "history_token_ids": [1, 2, 3],
                "probe_token_ids": [4],
                "previous_span": {"start": 0, "end": 1},
                "current_span": {"start": 1, "end": 3},
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["current_attention"], 0.2)
        self.assertEqual(self.engine.asag_probe.analyze_calls[-1][-1], "worker-0:q1")

        response = self.client.delete("/sessions/worker-0:q1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"released": True})
        self.assertEqual(self.engine.asag_probe.released, ["worker-0:q1"])


if __name__ == "__main__":
    unittest.main()
