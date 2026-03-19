import os
import sys
import unittest
from dotenv import load_dotenv
from types import SimpleNamespace
from langgraph.graph import END

# Provide dummy API key for local/tests before module import
load_dotenv()
os.environ.setdefault("GEMINI_API_KEY", "dummy")
os.environ.setdefault("GOOGLE_API_KEY", "dummy")

# Ensure the source module path is in sys.path for local tests
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import architect


class TestArchitect(unittest.TestCase):
    def test_provision_notion_workspace(self):
        result = architect.provision_notion_workspace.func("Acme", "{}")
        self.assertIn("Successfully built the workspace for Acme", result)

    def test_scrape_client_website(self):
        result = architect.scrape_client_website.func("https://example.com")
        self.assertIn("Client Profile", result)

    def test_researcher_node(self):
        state = {
            "client_name": "Acme",
            "client_url": "https://example.com",
            "business_context": "",
            "client_request": "test",
            "current_schema": "",
            "review_feedback": "",
            "iteration_count": 0,
            "final_approval": False,
            "deployment_status": ""
        }

        original_scraper = architect.scrape_client_website
        architect.scrape_client_website = SimpleNamespace(invoke=lambda args: "SCRAPED")

        try:
            output = architect.researcher_node(state)
            self.assertEqual(output, {"business_context": "SCRAPED"})
        finally:
            architect.scrape_client_website = original_scraper

    def test_planner_node(self):
        state = {
            "client_name": "Acme",
            "client_url": "https://example.com",
            "business_context": "Data",
            "client_request": "Build schema",
            "current_schema": "",
            "review_feedback": "",
            "iteration_count": 0,
            "final_approval": False,
            "deployment_status": ""
        }

        original_llm = architect.llm
        architect.llm = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content='{"tbl":"x"}'))

        try:
            output = architect.planner_node(state)
            self.assertEqual(output["current_schema"], '{"tbl":"x"}')
            self.assertEqual(output["iteration_count"], 1)
        finally:
            architect.llm = original_llm

    def test_reviewer_node_approved(self):
        state = {
            "current_schema": "{}",
            "client_request": "something"
        }

        original_llm = architect.llm
        architect.llm = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content="APPROVED"))

        try:
            output = architect.reviewer_node(state)
            self.assertTrue(output["final_approval"])
            self.assertEqual(output["review_feedback"], "")
        finally:
            architect.llm = original_llm

    def test_reviewer_node_rejected(self):
        state = {
            "current_schema": "{}",
            "client_request": "something"
        }

        original_llm = architect.llm
        architect.llm = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content="Needs more relations"))

        try:
            output = architect.reviewer_node(state)
            self.assertFalse(output["final_approval"])
            self.assertEqual(output["review_feedback"], "Needs more relations")
        finally:
            architect.llm = original_llm

    def test_executor_node(self):
        state = {
            "client_name": "Acme",
            "current_schema": "{}"
        }

        original_tool = architect.provision_notion_workspace
        architect.provision_notion_workspace = SimpleNamespace(invoke=lambda args: "DEPLOYED")

        try:
            output = architect.executor_node(state)
            self.assertEqual(output["deployment_status"], "DEPLOYED")
        finally:
            architect.provision_notion_workspace = original_tool

    def test_should_continue(self):
        self.assertEqual(architect.should_continue({"final_approval": True, "iteration_count": 0}), "executor")
        self.assertEqual(architect.should_continue({"final_approval": False, "iteration_count": 3}), END)
        self.assertEqual(architect.should_continue({"final_approval": False, "iteration_count": 1}), "planner")


if __name__ == "__main__":
    unittest.main()
