import os
import asyncio
from typing import List, Optional
import json
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import httpx
from tavily import TavilyClient
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Load environment variables
load_dotenv()

# Check for API keys
if not os.getenv("TAVILY_API_KEY"):
    print("Warning: TAVILY_API_KEY is not set.")
if not os.getenv("GEMINI_API_KEY"):
    print("Warning: GEMINI_API_KEY is not set. Gemini is required for this free-tier setup.")

# Initialize clients
tavily_api_key = os.getenv("TAVILY_API_KEY", "")

if tavily_api_key:
    tavily_client = TavilyClient(api_key=tavily_api_key)
else:
    tavily_client = None

# Model Configuration
MODEL = 'gemini-2.5-flash'


# ---------------------------------------------------------
# Define Models
# ---------------------------------------------------------

class DoctorSearchInput(BaseModel):
    name: str = Field(description="Name of the doctor")
    hospital: str = Field(description="Hospital they are associated with")
    additional_context: Optional[str] = Field(default=None, description="Any additional context provided by the user to refine the search")

class DoctorSearchResult(BaseModel):
    profile_url: Optional[str] = Field(description="The URL found for the doctor's profile. None if not found.")
    source_platform: Optional[str] = Field(description="The name of the platform the profile was found on (e.g., Practo, Lybrate, Apollo, NMR, SMC).")
    found_name: Optional[str] = Field(description="The name of the doctor as it appeared in the search results.")
    found_specialty: Optional[str] = Field(description="The specialty of the doctor as it appeared in the search results.")
    confidence_reasoning: str = Field(description="Brief explanation of why this URL was chosen or why it was not found.")

class ExtractedCollege(BaseModel):
    name: str = Field(description="Name of the college or university")
    degree: Optional[str] = Field(default=None, description="The degree obtained (e.g., MBBS, MD, MS)")
    year: Optional[int] = Field(default=None, description="The year of graduation if available")

class DoctorExtractedData(BaseModel):
    """Data extracted directly from the Firecrawl scrape"""
    colleges: List[ExtractedCollege] = Field(default_factory=list, description="List of colleges where the doctor studied, including degree and year.")
    registrations: List[str] = Field(default_factory=list, description="Registration details, such as registration number and medical council.")

class CollegeInfo(BaseModel):
    name: str = Field(description="Name of the college or university")
    degree: Optional[str] = Field(default=None, description="The degree obtained")
    year: Optional[int] = Field(default=None, description="The year of graduation")
    is_government: bool = Field(description="True if it is a government/public institution")
    is_private: bool = Field(description="True if it is a private institution")

class DoctorFinalProfile(BaseModel):
    """The final structured output of the agent"""
    name: str
    hospital: str
    profile_url: str
    source_platform: str
    colleges: List[CollegeInfo] = Field(description="Colleges the doctor studied at, along with institution type")
    registrations: List[str] = Field(description="Medical council registrations")

# ---------------------------------------------------------
# Define Agents
# ---------------------------------------------------------

# 1. Search Agent
search_agent = Agent(
    MODEL,
    output_type=DoctorSearchResult,
    system_prompt=(
        "You are an expert search assistant. Your goal is to find a reliable online profile URL "
        "for a specific medical doctor based on the user's provided name and hospital.\n\n"
        "You must prioritize sources in this exact order:\n"
        "1. Practo\n"
        "2. The specific Hospital's official website (e.g., Apollo, Fortis, Manipal)\n"
        "3. Lybrate\n"
        "4. State Medical Council (SMC) directories or National Medical Register (NMR) indexers\n"
        "5. JustDial\n\n"
        "Use the `search_profile_url` tool to query the web. "
        "Analyze the search snippets to ensure the URL matches the doctor's identity. "
        "If you find a matching URL, extract the doctor's name, specialty, the exact URL, and identify the `source_platform` (e.g., 'Practo', 'Apollo Hospitals', 'Lybrate')."
    )
)

@search_agent.tool
async def search_profile_url(ctx: RunContext[None], query: str) -> str:
    """Searches the web using Tavily to find doctor profiles across multiple trusted healthcare domains."""
    if not tavily_client:
        return "Error: Tavily client not initialized (missing API key)."
    try:
        # If the LLM generates a very generic query without a site constraint, we can append trusted medical domains to guide the search.
        # But we also want to allow the LLM to write "site:apollo.com" if it chooses to.
        search_result = tavily_client.search(query=query, search_depth="basic")
        results = search_result.get('results', [])
        
        formatted_results = []
        for r in results:
             formatted_results.append(f"URL: {r['url']}\nContent: {r['content']}\n")
        return "\n---\n".join(formatted_results) if formatted_results else "No results found."
    except Exception as e:
         return f"Error searching: {e}"


# 2. Extraction Agent
extraction_agent = Agent(
    MODEL,
    output_type=DoctorExtractedData,
    system_prompt=(
        "You are an expert data extractor. I will provide you with the raw markdown text "
        "scraped from a doctor's Practo profile. Your job is to extract their educational "
        "history (where they studied for their degrees) and their medical registrations. "
        "For each college, extract the college name, the degree obtained (e.g. MBBS, MD), and the graduation year if available. "
        "Return the exact text of the registrations. "
        "If they are not found, return empty lists."
    )
)

# 3. Enrichment Agent
enrichment_agent = Agent(
    MODEL,
    output_type=CollegeInfo,
    system_prompt=(
        "You are an expert researcher. I will provide you with a JSON object representing a medical college in India, including the degree and year. "
        "You must determine if the college is a government (public) institution or a private institution. "
        "Use the `search_college_type` tool to search the web for evidence using ONLY the college name. "
        "Based on the search results, return the CollegeInfo, preserving the original name, degree, and year, while setting is_government and is_private appropriately."
    )
)

@enrichment_agent.tool
async def search_college_type(ctx: RunContext[None], college_data: str) -> str:
    """Uses Tavily to search the web and gather context about a specific medical college. Input should be the college name."""
    query = f"Is {college_data} a government or private medical college in India?"
    if not tavily_client:
        return "Error: Tavily client not initialized."
    try:
        search_result = tavily_client.search(query=query, search_depth="basic")
        snippets = [result['content'] for result in search_result.get('results', [])]
        return "\n".join(snippets)
    except Exception as e:
        return f"Error searching: {e}"

# ---------------------------------------------------------
# Application Flow: FastAPI WebSocket Endpoint
# ---------------------------------------------------------

app = FastAPI(title="Doctor Data Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/extract")
async def extract_doctor_data(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Step 1: Wait for Initial Search Request
        data = await websocket.receive_text()
        payload = json.loads(data)
        
        name = payload.get("name", "").strip()
        hospital = payload.get("hospital", "").strip()
        additional_context = payload.get("additional_context", None)
        
        if not name or not hospital:
            await websocket.send_json({"type": "error", "message": "Name and Hospital are required."})
            await websocket.close()
            return
            
        profile_url = None
        source_platform = None
        tried_urls = []
        
        # Outer Loop: If extraction fails or finds 0 colleges, we come back here to find a NEW profile
        while True:
            # Loop for Search & Confirm
            while True:
                await websocket.send_json({"type": "status", "message": f"Searching multiple sources for {name}..."})
                
                prompt = f"Find a profile URL for Doctor Name: {name}, Hospital: {hospital}."
                if additional_context:
                    prompt += f" Additional Context: {additional_context}"
                
                # Instruct the agent to ignore profiles that previously yielded 0 colleges
                if tried_urls:
                    prompt += f" DO NOT return any of these URLs: {', '.join(tried_urls)}"
                    
                try:
                    result = await search_agent.run(prompt)
                    search_data = result.output
                    
                    if not search_data.profile_url:
                         await websocket.send_json({
                             "type": "search_failed", 
                             "reasoning": search_data.confidence_reasoning
                         })
                         
                         # Wait for refined context from user
                         resp = await websocket.receive_text()
                         resp_payload = json.loads(resp)
                         
                         if resp_payload.get("action") == "abort":
                             return
                             
                         additional_context = resp_payload.get("additional_context")
                    else:
                         await websocket.send_json({
                             "type": "search_result",
                             "data": search_data.model_dump()
                         })
                         
                         # Wait for user confirmation
                         resp = await websocket.receive_text()
                         resp_payload = json.loads(resp)
                         
                         if resp_payload.get("action") == "confirm":
                             profile_url = search_data.profile_url
                             source_platform = search_data.source_platform
                             break
                         elif resp_payload.get("action") == "refine":
                             additional_context = resp_payload.get("additional_context")
                         else:
                             return # Aborted
                             
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"Search agent failed: {str(e)}"})
                    return

            # Stage 2: Scrape & Extract
            await websocket.send_json({"type": "status", "message": f"Fetching markdown from {source_platform} via Jina..."})
            
            try:
                async with httpx.AsyncClient() as client:
                    jina_url = f"https://r.jina.ai/{profile_url}"
                    response = await client.get(jina_url, timeout=30.0)
                    
                    if response.status_code != 200:
                        await websocket.send_json({"type": "error", "message": f"Failed to get markdown content. Status code: {response.status_code}"})
                        return
                    markdown_content = response.text
                    
                if not markdown_content:
                    await websocket.send_json({"type": "error", "message": "Scraped markdown content is empty."})
                    return
                    
                await websocket.send_json({"type": "status", "message": "Analyzing profile with Gemini..."})
                extract_result = await extraction_agent.run(markdown_content)
                extracted_data = extract_result.output
                
                # Stage 3: Enrichment
                if not extracted_data.colleges:
                    await websocket.send_json({"type": "status", "message": f"No educational data found on {source_platform}. Adding to blocklist and searching for an alternative profile..."})
                    tried_urls.append(profile_url)
                    continue # Loop back to the very top (Stage 1 search) to find a new URL
                    
                await websocket.send_json({"type": "status", "message": f"Found {len(extracted_data.colleges)} colleges. Routing to Tavily..."})
                
                final_colleges = []
                for college_obj in extracted_data.colleges:
                     await websocket.send_json({"type": "status", "message": f"Determining type for college: {college_obj.name}..."})
                     # Pass the whole object as json so the LLM retains degree and year
                     enrich_result = await enrichment_agent.run(college_obj.model_dump_json())
                     final_colleges.append(enrich_result.output)
                     
                # Construct final profile
                final_profile = DoctorFinalProfile(
                     name=name,
                     hospital=hospital,
                     profile_url=profile_url,
                     source_platform=source_platform,
                     colleges=final_colleges,
                     registrations=extracted_data.registrations
                )
                
                await websocket.send_json({
                    "type": "final_result",
                    "data": final_profile.model_dump()
                })
                
                # End the outer loop after a successful extraction
                break
                
            except Exception as e:
                await websocket.send_json({"type": "error", "message": f"Extraction failed: {str(e)}"})
                break
            
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Unexpected connection error: {e}")

# ---------------------------------------------------------
# Serve React Frontend (Monolith)
# ---------------------------------------------------------

# Ensure we don't crash on local dev if the 'dist' folder doesn't exist yet
dist_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
assets_path = os.path.join(dist_path, "assets")

if os.path.exists(assets_path):
    app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

@app.get("/{catchall:path}")
def serve_react_app(catchall: str):
    file_path = os.path.join(dist_path, catchall)
    
    # If the user is requesting a specific file like favicon.ico
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # Otherwise, fall back to the React index.html for client-side routing
    index_path = os.path.join(dist_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
        
    return {"error": "Frontend build not found. During local development, you can ignore this and continue using localhost:5173 for the frontend."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
