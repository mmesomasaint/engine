import os
import json
import requests
from dotenv import load_dotenv 
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# ---------------------------------------------------------
# THE STATE
# ---------------------------------------------------------
class ArchitectState(TypedDict):
    brief_id: str
    client_name: str
    client_request: str
    notion_token: str
    base_page_id: str
    current_schema: str
    review_feedback: str
    iteration_count: int
    final_approval: bool
    deployment_status: str
    live_notion_url: str | None

class ExecutionResult(TypedDict):
    success: bool
    message: str
    live_notion_url: str | None

# ---------------------------------------------------------
# THE NOTION TOOL
# ---------------------------------------------------------
@tool
def provision_notion_workspace(client_name: str, schema_json: str, notion_token: str, base_page_id: str) -> ExecutionResult:
    """Executes API calls to Notion to build a custom workspace."""
    print(f"[SYSTEM ACTION] Provisioning Notion OS for {client_name}...")
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }
    try:
        architecture = json.loads(schema_json)
        
        # Build Dashboard Page
        dashboard_payload = {
            "parent": {"type": "page_id", "page_id": base_page_id},
            "properties": {"title": [{"text": {"content": f"{client_name} - Optima OS"}}]}
        }
        res = requests.post("https://api.notion.com/v1/pages", headers=headers, json=dashboard_payload)
        res.raise_for_status()
        dashboard_id = res.json()["id"]
        dashboard_url = res.json()["url"]
        
        db_list = architecture.get("databases", []) if isinstance(architecture, dict) else architecture
        if not isinstance(db_list, list) or len(db_list) == 0:
            raise ValueError("AI generated an empty or invalid database list.")

        db_name_to_real_id = {}
        pending_relations = []

        # PASS 1: Build base databases
        for db in db_list:
            db_name = db.get("name", "Untitled")
            clean_properties = {}
            for prop_name, prop_val in db.get("properties", {}).items():
                if "relation" in prop_val:
                    pending_relations.append({
                        "source_db_name": db_name,
                        "prop_name": prop_name,
                        "target_db_name": prop_val["relation"].get("database_id")
                    })
                else:
                    clean_properties[prop_name] = prop_val
            
            db_payload = {
                "parent": {"type": "page_id", "page_id": dashboard_id},
                "title": [{"text": {"content": db_name}}],
                "properties": clean_properties 
            }
            db_res = requests.post("https://api.notion.com/v1/databases", headers=headers, json=db_payload)
            db_res.raise_for_status() 
            db_name_to_real_id[db_name] = db_res.json()["id"]

        # PASS 2: Build Relations
        for rel in pending_relations:
            source_id = db_name_to_real_id.get(rel["source_db_name"])
            target_id = db_name_to_real_id.get(rel["target_db_name"])
            if source_id and target_id:
                patch_payload = {
                    "properties": {
                        rel["prop_name"]: {
                            "relation": {"database_id": target_id, "single_property": {}}
                        }
                    }
                }
                requests.patch(f"https://api.notion.com/v1/databases/{source_id}", headers=headers, json=patch_payload).raise_for_status()

        return {"success": True, "message": "Successfully built.", "live_notion_url": dashboard_url}
        
    except Exception as e:
        print(f"[NOTION EXECUTOR SYSTEM ERROR]: {str(e)}")
        return {"success": False, "message": str(e), "live_notion_url": None}

# ---------------------------------------------------------
# GRAPH NODES
# ---------------------------------------------------------
def planner_node(state: ArchitectState) -> Dict[str, Any]:
    print(f"\n--- PLANNER: Iteration {state['iteration_count']} ---")
    feedback_context: str = f"\nCritique to fix:\n{state['review_feedback']}" if state.get("review_feedback") else ""

    prompt: str = f"""You are an elite Notion System Architect. 
    Design a robust, relational database architecture to solve this operational requirement:
    {state['client_request']}
    
    CRITICAL RULES:
    1. Output MUST be a raw JSON array of database objects. No markdown blocks, no formatting.
    2. Properties must use Notion's strict nested objects (e.g. "Status": {{ "select": {{ "options": [...] }} }}).
    3. For relations, use the EXACT string name of the target database (e.g., "Related DB": {{ "relation": {{ "database_id": "Projects" }} }}).
    
    {feedback_context}"""

    response = llm.invoke([SystemMessage(content="You are an expert system architect."), HumanMessage(content=prompt)])
    
    # --- THE MARKDOWN STRIPPER ---
    # Forcefully remove ```json and ``` if the AI stubbornly included them
    raw_content: str = str(response.content).strip()
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]
    if raw_content.startswith("```"):
        raw_content = raw_content[3:]
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]
        
    cleaned_schema = raw_content.strip()

    return {"current_schema": cleaned_schema, "iteration_count": state["iteration_count"] + 1}


def reviewer_node(state: ArchitectState) -> Dict[str, Any]:
    print("\n--- QA REVIEWER: Inspecting Schema ---")
    prompt: str = f"""You are a ruthless QA Engineer. Review this Notion JSON schema:
    {state['current_schema']}
    
    Client requested: {state['client_request']}
    
    Rules: 
    1. Must have at least two tables. 
    2. Must be 100% valid JSON syntax.
    If it passes, reply EXACTLY with 'APPROVED'. Otherwise, detail what is missing."""
    
    response = llm.invoke([SystemMessage(content="You are a strict QA reviewer."), HumanMessage(content=prompt)])
    result: str = str(response.content).strip()
    
    if "APPROVED" in result:
        print("[QA RESULT] Schema Approved!")
        return {"final_approval": True, "review_feedback": ""}
        
    # --- THE TELEMETRY LOG ---
    print(f"[QA REJECTION REASON] {result}")
    return {"final_approval": False, "review_feedback": result}

def executor_node(state: ArchitectState):
    print("\n--- EXECUTOR: Deploying API ---")
    result = provision_notion_workspace.invoke({
        "client_name": state["client_name"], 
        "schema_json": state["current_schema"],
        "notion_token": state["notion_token"], 
        "base_page_id": state["base_page_id"] 
    })
    return {"deployment_status": result["message"], "live_notion_url": result["live_notion_url"]}

def should_continue(state: ArchitectState):
    if state["final_approval"]: return "executor"
    if state["iteration_count"] >= 3: return END
    return "planner"

# ---------------------------------------------------------
# COMPILE GRAPH
# ---------------------------------------------------------
workflow = StateGraph(ArchitectState)
workflow.add_node("planner", planner_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("executor", executor_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "reviewer")
workflow.add_conditional_edges("reviewer", should_continue)
workflow.add_edge("executor", END)

architect_app = workflow.compile()