import os
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from typing import Dict, Any, Optional
from src.architect import architect_app, ArchitectState

load_dotenv()

# Initialize Admin Supabase Client with strict None checks
SUPABASE_URL: Optional[str] = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY: Optional[str] = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("Missing Supabase Environment Variables in Python.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://crispy-journey-4p4r5vp97g7376p9-3000.app.github.dev", 
        "https://optima-portals-six.vercel.app", 
        "https://portal.optimalogic.studio"
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WebhookPayload(BaseModel):
    brief_id: str

def run_langgraph_agent(brief_id: str) -> None:
    print(f"[INIT] Waking Optima Engine for Brief ID: {brief_id}")
    
    try:
        # Fetch the Operational Brief
        brief_record = supabase.table("operational_briefs").select("*").eq("id", brief_id).maybe_single().execute()
        
        # TYPE GUARD
        if not brief_record or not isinstance(brief_record.data, dict):
            raise ValueError(f"Database Error: Could not locate brief {brief_id} or data is invalid.")
            
        brief: Dict[str, Any] = brief_record.data
            
        user_id: str = str(brief.get("user_id", ""))
        company_name: str = str(brief.get("company_name", "Unknown Company"))
        bottleneck: str = str(brief.get("primary_bottleneck", ""))
        tools: str = str(brief.get("current_tools", ""))

        if not user_id:
            raise ValueError("Data Integrity Error: Brief missing user_id.")

        # Fetch the Agency's Notion Keys
        integration_record = supabase.table("agency_integrations").select("*").eq("agency_id", user_id).maybe_single().execute()
        
        # TYPE GUARD
        if not integration_record or not isinstance(integration_record.data, dict):
            raise ValueError("Integration Error: Agency has not connected their Notion account.")
            
        integration: Dict[str, Any] = integration_record.data
        
        if not integration.get("notion_access_token"):
            raise ValueError("Integration Error: Agency missing Notion Access Token.")
            
        notion_token: str = str(integration.get("notion_access_token", ""))
        base_page_id: str = str(integration.get("base_notion_page_id", ""))

        client_request_formatted: str = f"Primary Bottleneck: {bottleneck}. Tools to integrate: {tools}."

        # Boot LangGraph Agents
        print("[COMPUTE] Booting LangGraph Agents...")
        initial_state: ArchitectState = {
            "brief_id": brief_id,
            "client_name": company_name,
            "client_request": client_request_formatted,
            "notion_token": notion_token,
            "base_page_id": base_page_id,
            "current_schema": "",
            "review_feedback": "",
            "iteration_count": 0,
            "final_approval": False,
            "deployment_status": "",
            "live_notion_url": None
        }
        
        # Execute the Graph (returns a dict representing the final state)
        final_state: Dict[str, Any] = architect_app.invoke(initial_state)
        
        if not final_state.get("final_approval"):
            raise RuntimeError("Graph Error: Agents failed to achieve architectural approval.")
        
        # Update the dashboard
        deployment_msg: str = str(final_state.get("deployment_status", "Architecture Deployed Successfully."))
        live_url: Optional[str] = final_state.get("live_notion_url")

        # If live_url is missing, the deployment failed at the Notion API level
        if not live_url or live_url == "#":
            supabase.table("operational_briefs").update({
                "status": "failed",
                "data_relationships": f"Execution Failed: {deployment_msg}",
                "live_notion_url": None
            }).eq("id", brief_id).execute()
            
            print(f"❌ [FAILED] Architecture failed during Notion API deployment for {company_name}")
        else:
            supabase.table("operational_briefs").update({
                "status": "completed",
                "data_relationships": deployment_msg,
                "live_notion_url": live_url
            }).eq("id", brief_id).execute()
            
            print(f"✅ [SUCCESS] Architecture provisioned for {company_name}")
        
    except Exception as e:
        error_message: str = str(e)
        print(f"[FATAL EXCEPTION] {error_message}")
        
        try:
            supabase.table("operational_briefs").update({
                "status": "failed",
                "data_relationships": f"System Error: {error_message}",
            }).eq("id", brief_id).execute()
        except Exception as db_fallback_error:
            print(f"[CATASTROPHIC] Failed to update database with error state: {str(db_fallback_error)}")


@app.post("/api/generate-os")
async def start_generation(payload: WebhookPayload, background_tasks: BackgroundTasks) -> Dict[str, str]:
    background_tasks.add_task(run_langgraph_agent, payload.brief_id)
    return {"status": "Processing Engine Pipeline", "brief_id": payload.brief_id}

# Health Check Endpoint
@app.get("/api/")
async def test_entry():
    return {"message": "Gateway is up and running!"}