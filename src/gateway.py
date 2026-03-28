import os
import uuid
import json
import numpy as np
from typing import Optional, Union
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.architect import architect_app, ArchitectState, provision_notion_workspace
from pydantic import BaseModel
from supabase import create_client, Client
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# ---------------------------------------------------------
# INITIALIZE SUPABASE ADMIN CLIENT
# We use the Service Role Key here to bypass RLS, because 
# the Python server acts as the absolute system admin.
# ---------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("Missing Supabase Environment Variables in Python.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

client = genai.Client()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://optima-portals-six.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the embedding model
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

class ClientRequest(BaseModel):
    portal_id: Union[str, int]
    client_name: str
    client_url: Optional[str] = None  # NEW: URL to scrape 
    client_request: str
    notion_token: str  # NEW: Notion API token
    base_page_id: str

# ---------------------------------------------------------
# THE BACKGROUND WORKER
# ---------------------------------------------------------
def run_langgraph_agent(portal_id: str, request_data: ClientRequest):
    """
    Runs LangGraph, checks the vector cache, and writes the final result to Supabase.
    Engineered with defensive data validation for production.
    """
    print(f"[INIT] Booting OS Generation for Portal ID: {portal_id}")
    
    try:
        # ---------------------------------------------------------
        # PRE-FLIGHT CHECK 1: Validate Portal & Agency Identity
        # ---------------------------------------------------------
        portal_record = supabase.table("active_portals").select("agency_id").eq("id", portal_id).maybe_single().execute()
        
        # Guard clause: Ensure data exists and is a dictionary
        if not portal_record or not portal_record.data or not isinstance(portal_record.data, dict):
            raise ValueError(f"Database Error: Could not locate a valid active portal for ID {portal_id}.")
            
        agency_id = portal_record.data.get("agency_id")
        if not agency_id:
            raise ValueError("Data Integrity Error: Portal record exists but is missing an associated agency_id.")

        # ---------------------------------------------------------
        # PRE-FLIGHT CHECK 2: Validate Notion Integrations
        # ---------------------------------------------------------
        integration_record = supabase.table("agency_integrations").select("base_notion_page_id").eq("agency_id", agency_id).maybe_single().execute()
        
        if not integration_record or not integration_record.data or not isinstance(integration_record.data, dict):
            raise ValueError("Integration Error: Agency has not connected their Notion account.")
            
        base_page_id = integration_record.data.get("base_notion_page_id")
        if not base_page_id:
            raise ValueError("Configuration Error: Agency has not set a Master Notion Page URL in Settings.")

        # ---------------------------------------------------------
        # EXECUTION PHASE 1: Semantic Caching
        # ---------------------------------------------------------
        print("[SYSTEM] Checking Semantic Cache for identical previous architectures...")
        cached_schema = check_semantic_cache(request_data.client_request)

        if cached_schema:
            print("[CACHE HIT] Bypassing LangGraph. Fast-tracking deployment...")
            
            # This now returns my TypedDict!
            result_payload = provision_notion_workspace.invoke({
                "client_name": request_data.client_name,
                "schema_json": cached_schema,
                "notion_token": request_data.notion_token,
                "base_page_id": base_page_id
            })
            
            if isinstance(result_payload, dict):
                deployment_msg = result_payload.get("message", "Deployed via Cache")
                live_url = result_payload.get("live_notion_url") or "#"
            else:
                # Extract the link from the raw string
                deployment_msg = str(result_payload)
                live_url = deployment_msg.split("Live URL: ")[-1].strip() if "Live URL: " in deployment_msg else "#"
            
        else:
            # ---------------------------------------------------------
            # EXECUTION PHASE 2: LangGraph Orchestration
            # ---------------------------------------------------------
            print("[CACHE MISS] Booting LangGraph Agents for full compute...")
            initial_state: ArchitectState = {
                "portal_id": portal_id,
                "notion_token": request_data.notion_token,
                "base_page_id": f"{base_page_id}", 
                "client_name": request_data.client_name,
                "client_url": request_data.client_url or "",
                "client_request": request_data.client_request,
                "business_context": "",
                "current_schema": "",
                "review_feedback": "",
                "iteration_count": 0,
                "final_approval": False,
                "deployment_status": "",
                "live_notion_url": ""
            }
            
            # Execute the Graph
            final_state = architect_app.invoke(initial_state)
            
            if not final_state.get("final_approval"):
                raise RuntimeError("Graph Error: Agents failed to achieve architectural approval after maximum iterations.")
            
            # Archive the new intelligence
            save_to_semantic_cache(request_data.client_request, final_state["current_schema"])
            
            # The LangGraph executor node should ideally be passing your TypedDict to the state
            deployment_data = final_state.get("deployment_status", {})
            
            # Defensively check if the node gave us your TypedDict or a string fallback
            if isinstance(deployment_data, dict):
                deployment_msg = deployment_data.get("message", "Deployed successfully.")
                live_url = deployment_data.get("live_notion_url") or "#"
            else:
                deployment_msg = str(deployment_data)
                # Rescue the URL from the flattened string using Python string splitting
                if "Live URL: " in deployment_msg:
                    live_url = deployment_msg.split("Live URL: ")[-1].strip()
                else:
                    live_url = "#"
            
        # ---------------------------------------------------------
        # POST-FLIGHT: Extract URL and Update Database
        # ---------------------------------------------------------
        supabase.table("active_portals").update({
            "status": "active",
            "deployment_status": deployment_msg,
            "live_notion_url": live_url
        }).eq("id", portal_id).execute()
        
        print(f"✅ [SUCCESS] Portal provisioned and DB updated for {portal_id}")
        
    except Exception as e:
        # ---------------------------------------------------------
        # GLOBAL FALLBACK: Graceful Failure Handling
        # ---------------------------------------------------------
        error_message = str(e)
        print(f"[FATAL EXCEPTION] {error_message}")
        
        # Ensure we always update the UI to stop the loading state
        try:
            supabase.table("active_portals").update({
                "status": "failed",
                "deployment_status": f"System Error: {error_message}"
            }).eq("id", portal_id).execute()
            print("[RECOVERY] Database successfully updated to 'failed' state.")
        except Exception as db_fallback_error:
            print(f"[CATASTROPHIC] Failed to update database with error state: {db_fallback_error}")

def check_semantic_cache(client_request: str) -> str | None:
    """
    Checks Supabase for a semantically identical previous request.
    Returns the cached JSON schema if a 95%+ match is found.
    """
    try:
        # Convert the new request into a mathematical vector
        query_vector = embeddings_model.embed_query(client_request)
        
        # Query Supabase using the RPC (Remote Procedure Call) we just created
        response = supabase.rpc(
            "match_cached_schema",
            {
                "query_embedding": query_vector,
                "match_threshold": 0.95, # 95% similarity required to trigger a cache hit
                "match_count": 1
            }
        ).execute()
        
        # Strictly validate the data type before interacting with it
        data = response.data
        
        if isinstance(data, list) and len(data) > 0:
            first_match = data[0]
            
            # Ensure the item inside the list is actually a dictionary
            if isinstance(first_match, dict):
                similarity = first_match.get('similarity', 0)
                print(f"[CACHE HIT] Similarity: {similarity}")
                
                schema = first_match.get('schema_json')
                if schema:
                    return f"{schema}"
            
        print("CACHE MISS. Routing to LangGraph Planner...")
        return None
        
    except Exception as e:
        print(f"Cache check failed: {e}")
        return None # Fail gracefully and let the AI generate it from scratch

def save_to_semantic_cache(client_request: str, schema_json: str):
    """
    Saves a newly generated and approved schema back to the database.
    """
    try:
        vector = embeddings_model.embed_query(client_request)
        supabase.table("semantic_cache").insert({
            "original_request": client_request,
            "schema_json": schema_json,
            "embedding": vector
        }).execute()
        print("New architecture saved to Semantic Cache.")
    except Exception as e:
        print(f"Failed to save to cache: {e}")

# ---------------------------------------------------------
# ROUTE 1: INTAKE ROUTE
# Returns instantly
# ---------------------------------------------------------
@app.post("/api/generate-os")
async def start_generation(req: ClientRequest, background_tasks: BackgroundTasks):
    # Hand the heavy lifting off to FastAPI's background thread
    background_tasks.add_task(run_langgraph_agent, f"{req.portal_id}", req)
    
    # Return the ID to Next.js immediately so the UI doesn't freeze
    return {"portal_id": req.portal_id, "message": "Graph execution started in background."}
