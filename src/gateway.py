import uuid
import numpy as np
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from architect import architect_app, ArchitectState
from pydantic import BaseModel

load_dotenv()
client = genai.Client()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# A mocked database storing previous OS architectures and their vector embeddings
vector_cache_db = [] 

# In production, this would be a Redis database or PostgreSQL (Supabase).
# For now, we use an in-memory dictionary to track our AI jobs.
jobs_db = {}

class ClientRequest(BaseModel):
    client_name: str
    client_url: str
    client_request: str

# ---------------------------------------------------------
# THE BACKGROUND WORKER
# ---------------------------------------------------------
def run_langgraph_agent(job_id: str, request_data: ClientRequest):
    """This function runs invisibly in the background."""
    
    # Initialize the LangGraph state for this specific client
    initial_state: ArchitectState = {
        "client_name": request_data.client_name,
        "client_url": request_data.client_url,
        "client_request": request_data.client_request,
        "business_context": "",
        "current_schema": "",
        "review_feedback": "",
        "iteration_count": 0,
        "final_approval": False,
        "deployment_status": ""
    }
    
    try:
        # Update status to processing
        jobs_db[job_id]["status"] = "processing"
        
        # Execute the Graph (This takes 20+ seconds)
        final_state = architect_app.invoke(initial_state)
        
        # Save the results
        jobs_db[job_id]["status"] = "completed"
        jobs_db[job_id]["result"] = final_state
        
    except Exception as e:
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["error"] = str(e)


def get_embedding(text: str) -> list[float]:
    """Converts text into an array of numbers representing its semantic meaning."""
    # Using Gemini's embedding model
    result = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )

    embeddings = getattr(result, "embeddings", None)
    if embeddings:
        first = embeddings[0]
        values = getattr(first, "values", None)
        if values is not None:
            return values
        if isinstance(first, dict) and "values" in first:
            return first["values"]

    data = getattr(result, "data", None)
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            if "embedding" in first:
                return first["embedding"]
            if "values" in first:
                return first["values"]

    embedding_attr = getattr(result, "embedding", None)
    if embedding_attr is not None:
        return embedding_attr

    raise ValueError("Unable to parse embedding from model response")

def cosine_similarity(vec1, vec2):
    """Measures how similar two vectors are (returns 0.0 to 1.0)"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ---------------------------------------------------------
# THE NEW OPTIMIZED INTAKE ROUTE
# ---------------------------------------------------------
@app.post("/api/generate-os-optimized")
async def start_generation_with_cache(req: ClientRequest, background_tasks: BackgroundTasks):
    
    # 1. Convert the user's request into a vector
    current_request_vector = get_embedding(req.client_request)
    
    # 2. Check the cache for semantic matches
    for cached_item in vector_cache_db:
        similarity_score = cosine_similarity(current_request_vector, cached_item["vector"])
        
        # 3. If it is 95% similar to a past request, BYPASS THE AI
        if similarity_score >= 0.95:
            print(f"♻️ CACHE HIT! Similarity: {similarity_score}. Bypassing LangGraph.")
            
            # Immediately trigger the Executor node (Notion API) with the cached schema
            # We save 20 seconds of graph looping and 100% of the LLM generation costs.
            return {
                "status": "completed_from_cache", 
                "schema": cached_item["schema"],
                "message": "Provisioned instantly from semantic cache."
            }

    # 4. If no match is found, proceed with the expensive LangGraph background job
    print("🧠 CACHE MISS. Booting up LangGraph Agents...")
    job_id = str(uuid.uuid4())
    jobs_db[job_id] = {"status": "pending", "result": None}
    background_tasks.add_task(run_langgraph_agent, job_id, req)
    
    return {"job_id": job_id, "message": "Graph execution started."}

# ---------------------------------------------------------
# ROUTE 1: INTAKE (Returns instantly)
# ---------------------------------------------------------
@app.post("/api/generate-os")
async def start_generation(req: ClientRequest, background_tasks: BackgroundTasks):
    # Generate a unique tracking ID for this execution
    job_id = str(uuid.uuid4())
    
    # Initialize the job in our database
    jobs_db[job_id] = {"status": "pending", "result": None}
    
    # Hand the heavy lifting off to FastAPI's background thread
    background_tasks.add_task(run_langgraph_agent, job_id, req)
    
    # Return the ID to Next.js immediately so the UI doesn't freeze
    return {"job_id": job_id, "message": "Graph execution started in background."}


# ---------------------------------------------------------
# ROUTE 2: THE STATUS CHECKER
# ---------------------------------------------------------
@app.get("/api/status/{job_id}")
async def check_status(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return jobs_db[job_id]