import os
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv 
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

# Ensure your GEMINI_API_KEY is still set in your environment
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# ---------------------------------------------------------
# DEFINE THE STATE (The Shared Memory)
# ---------------------------------------------------------
class ArchitectState(TypedDict):
    portal_id: str
    client_name: str
    client_url: str          # NEW: URL to scrape
    business_context: str    # NEW: Data returned from the scraper
    client_request: str
    notion_token: str          # NEW: Notion API token for deployment
    base_page_id: str
    current_schema: str
    review_feedback: str
    iteration_count: int
    final_approval: bool
    deployment_status: str   # NEW: Did the Notion API succeed?
    live_notion_url: str | None

# Define the type for execution result
class ExecutionResult(TypedDict):
    success: bool
    message: str
    live_notion_url: str | None

# Note: In your node logic, ensure you pass the token and base page ID to this function.
@tool
def provision_notion_workspace(client_name: str, schema_json: str, notion_token: str, base_page_id: str) -> ExecutionResult:
    """
    Executes an API call to Notion to build a custom workspace.
    Use this tool ONLY AFTER the QA Reviewer has approved the schema.
    """
    print(f"[SYSTEM ACTION] Provisioning Notion OS for {client_name}...")
    
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }
    
    try:
        # Parse the AI's validated JSON schema
        architecture = json.loads(schema_json)
        
        # Build the Master Portal Page (The Dashboard)
        dashboard_payload = {
            "parent": {"type": "page_id", "page_id": base_page_id},
            "properties": {
                "title": [{"text": {"content": f"{client_name} - Client Portal"}}]
            }
        }
        
        res = requests.post("https://api.notion.com/v1/pages", headers=headers, json=dashboard_payload)
        res.raise_for_status()
        
        dashboard_id = res.json()["id"]
        dashboard_url = res.json()["url"]
        
        # Defensively check the structure
        db_list = architecture.get("databases", []) if isinstance(architecture, dict) else architecture
        
        if not isinstance(db_list, list) or len(db_list) == 0:
            raise ValueError("AI generated an empty or invalid database list.")

        # Dictionaries to store state for Pass 2
        db_name_to_real_id = {}
        pending_relations = []

        # ---------------------------------------------------------
        # PASS 1: Create all databases (Stripping out relations)
        # ---------------------------------------------------------
        for db in db_list:
            db_name = db.get("name", "Untitled")
            print(f"🔨 [SYSTEM] PASS 1: Building base database: {db_name}")
            
            clean_properties = {}
            for prop_name, prop_val in db.get("properties", {}).items():
                if "relation" in prop_val:
                    # Save this relation for Pass 2
                    target_db_name = prop_val["relation"].get("database_id")
                    pending_relations.append({
                        "source_db_name": db_name,
                        "prop_name": prop_name,
                        "target_db_name": target_db_name
                    })
                else:
                    # Keep safe properties (title, rich_text, select, etc.)
                    clean_properties[prop_name] = prop_val
            
            db_payload = {
                "parent": {"type": "page_id", "page_id": dashboard_id},
                "title": [{"text": {"content": db_name}}],
                "properties": clean_properties 
            }
            
            db_res = requests.post("https://api.notion.com/v1/databases", headers=headers, json=db_payload)
            db_res.raise_for_status() 
            
            # Store the real Notion ID!
            real_db_id = db_res.json()["id"]
            db_name_to_real_id[db_name] = real_db_id
            print(f"✅ [SYSTEM] Created {db_name}")

        # ---------------------------------------------------------
        # PASS 2: Establish the Relations
        # ---------------------------------------------------------
        for rel in pending_relations:
            source_id = db_name_to_real_id.get(rel["source_db_name"])
            target_id = db_name_to_real_id.get(rel["target_db_name"])
            
            if source_id and target_id:
                print(f"🔗 [SYSTEM] PASS 2: Linking '{rel['source_db_name']}' -> '{rel['target_db_name']}'")
                
                patch_payload = {
                    "properties": {
                        rel["prop_name"]: {
                            "relation": {
                                "database_id": target_id,
                                "single_property": {}
                            }
                        }
                    }
                }
                patch_res = requests.patch(f"https://api.notion.com/v1/databases/{source_id}", headers=headers, json=patch_payload)
                patch_res.raise_for_status()
  
        print("[SUCCESS] Notion workspace provisioned successfully!")
        return {
            "success": True,
            "message": f"Successfully built the workspace for {client_name}. Live URL: {dashboard_url}",
            "live_notion_url": dashboard_url
        } 
        
    except json.JSONDecodeError:
        print("[NOTION EXECUTOR] AI generated invalid JSON.")
        return {
            "success": False,
            "message": "Error: The schema provided was not valid JSON. Please fix the formatting.",
            "live_notion_url": None
        }
    except requests.exceptions.HTTPError as e:
        # THE FIX: Print the exact reason Notion rejected our payload
        print(f"[NOTION EXECUTOR FATAL REJECTION]: {e.response.text}")
        return {
            "success": False,
            "message": f"Notion API Error: {e.response.text}",
            "live_notion_url": None
        }
    except Exception as e:
        print(f"[NOTION EXECUTOR SYSTEM ERROR]: {str(e)}")
        return {
            "success": False,
            "message": f"System Error during provisioning: {str(e)}",
            "live_notion_url": None
        }

@tool
def scrape_client_website(url: str) -> str:
    """
    Scrapes a client's website to extract their business model, team size, and core services.
    """
    print(f"\n[SYSTEM ACTION] Scraping data from {url}...")
    
    try:
        # We use a standard browser User-Agent so websites don't block our Python script
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Strip out the code we don't need the AI to read
        for script in soup(["script", "style", "nav", "footer"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        
        # Return the first 3000 characters to keep context windows clean and cheap
        return f"Website Content: {text[:3000]}"
        
    except requests.exceptions.RequestException as e:
        return f"Error: Could not scrape website. The client URL might be invalid or protected. Detail: {str(e)}"

# We give the Planner agent access to the tool
llm_with_tools = llm.bind_tools([provision_notion_workspace])

# 2. NEW NODE: THE RESEARCHER
def researcher_node(state: ArchitectState):
    print(f"\n--- RESEARCHER: Investigating {state['client_name']} ---")
    
    # Execute the scraping tool
    scraped_data = scrape_client_website.invoke({"url": state["client_url"]})
    
    return {"business_context": scraped_data}


# ---------------------------------------------------------
# 2. NODE A: THE PLANNER
# ---------------------------------------------------------
def planner_node(state: ArchitectState):
    print(f"\n--- PLANNER: Iteration {state['iteration_count']} ---")
    
    # If there is feedback from the reviewer, the planner needs to know about it.
    feedback_context = ""
    if state.get("review_feedback"):
        feedback_context = f"\nCritique to fix:\n{state['review_feedback']}"

    prompt = f"""You are an elite Notion System Architect. 
    Design a robust, relational database architecture for this client request:
    {state['client_request']}
    
    Business Context from Website: {state['business_context']}
    
    CRITICAL NOTION API RULES FOR DATABASE PROPERTIES:
    When defining the "properties" object for a database, you MUST use Notion's strict nested configuration objects. Do not use a "type" key for standard text fields.
    
    CORRECT FORMAT EXAMPLE:
    "properties": {{
      "Name": {{ "title": {{}} }},
      "Client Email": {{ "email": {{}} }},
      "Description": {{ "rich_text": {{}} }},
      "Status": {{ "select": {{ "options": [ {{"name": "Active", "color": "green"}}, {{"name": "Pending", "color": "yellow"}} ] }} }},
      "Cost": {{ "number": {{ "format": "dollar" }} }}
    }}
    
    INCORRECT FORMAT (DO NOT USE):
    "properties": {{ "Name": {{ "type": "title" }} }}
    
    OVERALL STRUCTURE:
    Your output must be a single array of database objects. Example:
    [
      {{
        "name": "Database Name",
        "properties": {{ ... }}
      }}
    ]

    CRITICAL RULE FOR RELATIONS:
    Because you are generating this schema before the databases exist, you do not know the real Notion database IDs. 
    If a property is a relation, you MUST set the "database_id" to the EXACT string name of the target database.
    
    RELATION FORMAT EXAMPLE:
    "Related Campaign": {{ "relation": {{ "database_id": "Campaigns" }} }}

    Output ONLY the raw JSON array detailing the databases and properties. Do NOT wrap the output in markdown formatting (do not use ```json).
    
    {feedback_context}"""

    response = llm.invoke([SystemMessage(content="You are an expert system architect."), HumanMessage(content=prompt)])
    
    # Update the state with the new schema and increment the counter
    return {
        "current_schema": response.content,
        "iteration_count": state["iteration_count"] + 1
    }

# ---------------------------------------------------------
# 3. NODE B: THE REVIEWER
# ---------------------------------------------------------
def reviewer_node(state: ArchitectState):
    print("\n--- REVIEWER: Inspecting Schema ---")
    
    prompt = f"""You are a ruthless QA Engineer for Notion systems.
    Review this proposed schema:
    {state['current_schema']}
    
    Client originally asked for: {state['client_request']}
    
    Rules for approval:
    1. Must have at least two related tables.
    2. Must include specific Notion property types (e.g., 'relation', 'select', 'date').
    
    If it passes, reply EXACTLY with 'APPROVED'.
    If it fails, reply with a detailed critique of what is missing or wrong."""

    response = llm.invoke([SystemMessage(content="You are a strict QA reviewer."), HumanMessage(content=prompt)])
    
    result = (str)(response.content).strip()
    
    if result == "APPROVED":
        print("QA Passed! Schema is perfect.")
        return {"final_approval": True, "review_feedback": ""}
    else:
        print("QA Failed. Sending back to Planner.")
        return {"final_approval": False, "review_feedback": result}

# 3. NEW NODE: THE EXECUTOR
def executor_node(state: ArchitectState):
    print("\n--- EXECUTOR: Deploying to Production ---")
    
    # Execute the Notion API tool using the approved schema
    result = provision_notion_workspace.invoke({
        "client_name": state["client_name"], 
        "schema_json": state["current_schema"],
        "notion_token": state["notion_token"], 
        "base_page_id": state["base_page_id"] 
    })
    
    return {
        "deployment_status": result["message"], 
        "live_notion_url": result["live_notion_url"]
    }

# ---------------------------------------------------------
# 4. THE ROUTER (The Conditional Edge)
# ---------------------------------------------------------
def should_continue(state: ArchitectState):
    # If QA approved it, move to DEPLOYMENT instead of END
    if state["final_approval"]:
        return "executor"
    # If we loop too many times, fail gracefully
    if state["iteration_count"] >= 3:
        return END
    # Otherwise, back to the Planner
    return "planner"

# 5. RECOMPILE THE GRAPH
workflow = StateGraph(ArchitectState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("planner", planner_node)      # Your existing node
workflow.add_node("reviewer", reviewer_node)    # Your existing node
workflow.add_node("executor", executor_node)

# The new flow: Research -> Plan -> Review
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "planner")
workflow.add_edge("planner", "reviewer")

# The loop: Review -> (Should Continue?) -> Planner OR Executor
workflow.add_conditional_edges("reviewer", should_continue)

# The finish line: Executor -> END
workflow.add_edge("executor", END)

architect_app = workflow.compile()

# ---------------------------------------------------------
# 6. TEST EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    initial_state: ArchitectState = {
        "portal_id": "",
        "client_name": "",
        "client_url": "",
        "business_context": "",
        "notion_token": "",
        "base_page_id": "",
        "client_request": "I need a system to track my freelance clients, their invoices, and the status of their current deliverables.",
        "current_schema": "",
        "review_feedback": "",
        "iteration_count": 0,
        "final_approval": False,
        "deployment_status": "",
        "live_notion_url": ""
    }
    
    print("Starting Autonomous Architecture Process...")
    final_state = architect_app.invoke(initial_state)
    
    print("\n==============================================")
    print("FINAL DELIVERABLE READY FOR NEXT.JS")
    print("==============================================")
    print(final_state["current_schema"])