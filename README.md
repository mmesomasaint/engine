# 🧠 Optima Engine: LangGraph AI Orchestrator

The autonomous Python backend for Optima Logic. This engine translates natural language business requests into highly structured, relational Notion architectures. It features multi-agent QA loops, a two-pass deployment system, and mathematical semantic caching to completely bypass LLM compute costs on repeat queries.

## 🏗 Tech Stack
* **Framework:** FastAPI (Python)
* **AI Orchestration:** LangGraph & LangChain
* **LLM & Embeddings:** Google Gemini (`gemini-embedding-001` with 3072 dimensions)
* **Vector Database:** Supabase (pgvector)
* **Deployment:** Render

## ⚙️ Core System Architecture
1. **Semantic Router:** Converts incoming requests into 3072-dimensional vectors. If a 95%+ mathematical match is found in the pgvector cache, it bypasses the AI entirely and fast-tracks deployment.
2. **Multi-Agent Planner & Reviewer:** A LangGraph cyclical node structure where a Planner drafts the Notion JSON schema, and a Reviewer verifies strict API compliance, forcing rewrites if necessary.
3. **Two-Pass Notion Executor:** * *Pass 1:* Physically builds the databases via the Notion API and extracts live UUIDs.
   * *Pass 2:* Dynamically patches the databases to map complex relations (e.g., linking a "Bug Tracker" to "Active Sprints").

## 🛠 Local Development

1. **Virtual Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: .\venv\Scripts\activate

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

3. **Environment Variables**
   Create a .env file in the root directory. Never commit this file.
   ```bash
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_key # Bypasses RLS
   GOOGLE_API_KEY=your_gemini_api_key

4. **Run The Server**
   Run the server
   ```bash
   uvicorn gateway:app --reload --port 8000