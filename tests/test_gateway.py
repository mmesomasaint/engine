import os
import sys
import unittest
import asyncio
from types import SimpleNamespace
from fastapi import BackgroundTasks, HTTPException
from fastapi.testclient import TestClient

# Ensure the source module path is in sys.path for local tests
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import gateway


class TestGateway(unittest.TestCase):
    def setUp(self):
        gateway.jobs_db.clear()
        gateway.vector_cache_db.clear()
        self.orig_get_embedding = gateway.get_embedding
        self.orig_architect_app_invoke = gateway.architect_app.invoke
        self.orig_embed_content = gateway.client.models.embed_content

    def tearDown(self):
        gateway.get_embedding = self.orig_get_embedding
        gateway.architect_app.invoke = self.orig_architect_app_invoke
        gateway.client.models.embed_content = self.orig_embed_content

    def test_cosine_similarity(self):
        self.assertAlmostEqual(gateway.cosine_similarity([1, 0], [1, 0]), 1.0)

    def test_get_embedding_from_embeddings_values(self):
        saved = gateway.client.models.embed_content

        class FakeResponse:
            embeddings = [SimpleNamespace(values=[0.1, 0.2, 0.3])]

        gateway.client.models.embed_content = lambda **kwargs: FakeResponse()
        try:
            self.assertEqual(gateway.get_embedding("hello"), [0.1, 0.2, 0.3])
        finally:
            gateway.client.models.embed_content = saved

    def test_get_embedding_from_data_embedding(self):
        saved = gateway.client.models.embed_content

        class FakeResponse:
            data = [{"embedding": [0.7, 0.8]}]

        gateway.client.models.embed_content = lambda **kwargs: FakeResponse()
        try:
            self.assertEqual(gateway.get_embedding("hello"), [0.7, 0.8])
        finally:
            gateway.client.models.embed_content = saved

    def test_get_embedding_raises_value_error(self):
        saved = gateway.client.models.embed_content

        class FakeResponse:
            pass

        gateway.client.models.embed_content = lambda **kwargs: FakeResponse()
        try:
            with self.assertRaises(ValueError):
                gateway.get_embedding("hello")
        finally:
            gateway.client.models.embed_content = saved

    def test_start_generation_with_cache_hit(self):
        gateway.vector_cache_db.append({"vector": [0.1, 0.2], "schema": "{}"})
        gateway.get_embedding = lambda text: [0.1, 0.2]

        client = TestClient(gateway.app)
        payload = {"client_name": "Acme", "client_url": "https://x", "client_request": "req"}
        response = client.post("/api/generate-os-optimized", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "completed_from_cache")

    def test_start_generation_with_cache_miss(self):
        gateway.get_embedding = lambda text: [0.0, 1.0]

        client = TestClient(gateway.app)
        payload = {"client_name": "Acme", "client_url": "https://x", "client_request": "req"}
        response = client.post("/api/generate-os-optimized", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertIn("job_id", response.json())

    def test_start_generation_route(self):
        client = TestClient(gateway.app)
        payload = {"client_name": "Acme", "client_url": "https://x", "client_request": "req"}
        response = client.post("/api/generate-os", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertIn("job_id", response.json())

    def test_check_status_not_found(self):
        with self.assertRaises(HTTPException):
            asyncio.run(gateway.check_status("does-not-exist"))

    def test_check_status_found(self):
        gateway.jobs_db["123"] = {"status": "pending", "result": None}
        self.assertEqual(asyncio.run(gateway.check_status("123")), {"status": "pending", "result": None})

    def test_run_langgraph_agent_success(self):
        gateway.jobs_db["job-1"] = {"status": "pending", "result": None}
        original = gateway.architect_app.invoke
        gateway.architect_app.invoke = lambda state: {"final": "ok"}

        try:
            gateway.run_langgraph_agent("job-1", SimpleNamespace(client_name="Acme", client_url="x", client_request="y"))
            self.assertEqual(gateway.jobs_db["job-1"]["status"], "completed")
            self.assertEqual(gateway.jobs_db["job-1"]["result"], {"final": "ok"})
        finally:
            gateway.architect_app.invoke = original

    def test_run_langgraph_agent_failure(self):
        gateway.jobs_db["job-2"] = {"status": "pending", "result": None}
        original = gateway.architect_app.invoke
        def raising(state):
            raise RuntimeError("boom")
        gateway.architect_app.invoke = raising

        try:
            gateway.run_langgraph_agent("job-2", SimpleNamespace(client_name="Acme", client_url="x", client_request="y"))
            self.assertEqual(gateway.jobs_db["job-2"]["status"], "failed")
            self.assertIn("boom", gateway.jobs_db["job-2"]["error"])
        finally:
            gateway.architect_app.invoke = original


if __name__ == "__main__":
    unittest.main()
