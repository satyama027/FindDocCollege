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

class ConsistencyResult(BaseModel):
    is_same_doctor: bool = Field(description="True if both profiles definitively refer to the same individual doctor.")
    confidence_reasoning: str = Field(description="Explanation of why these profiles match or do not match based on name, specialty, city, and hospital.")

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
        "⭐ CRITICAL PRIORITY: You MUST always prioritize finding and returning a Practo (practo.com) profile FIRST.\n"
        "Even if the user explicitly names a specific hospital (like Cloudnine or Apollo), you MUST still search for and prefer the doctor's Practo profile over the hospital's official website.\n\n"
        "Fallback sources, in exact order, if and ONLY if a Practo profile cannot be found:\n"
        "2. The specific Hospital's official website (e.g., Apollo, Fortis, Manipal)\n"
        "3. Lybrate\n"
        "4. State Medical Council (SMC) directories or National Medical Register (NMR) indexers\n"
        "5. JustDial\n\n"
        "Use the `search_profile_url` tool to query the web. "
        "Analyze the search snippets to ensure the URL matches the doctor's identity. "
        "If you find a matching URL, extract the doctor's name, specialty, the exact URL, and identify the `source_platform` (e.g., 'Practo', 'Apollo Hospitals', 'Lybrate').\n"
        "Return the EXACT profile URL without any line breaks or spaces."
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

# 4. Consistency Agent
consistency_agent = Agent(
    MODEL,
    output_type=ConsistencyResult,
    system_prompt=(
        "You are a medical data verification assistant. You will be given metadata from a primary, confirmed doctor profile "
        "and metadata from a newly scraped profile. Your job is to determine if both profiles refer to the EXACT SAME human being "
        "by comparing their names, specialties, cities, and hospital affiliations."
    )
)
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
            
        # Track global aggregation state
        aggregated_colleges = []
        extracted_degrees = set()
        total_urls_checked = 0
        max_urls = 10
        registrations = []
        tried_urls = []
        platform_retries = {}
        
        primary_source_platform = None
        primary_url_confirmed = False
        
        while total_urls_checked < max_urls and len(extracted_degrees) < 2:
            # -------------------------------------------------------------
            # Stage 1: Search & Confirm
            # -------------------------------------------------------------
            while True:
                if not primary_url_confirmed:
                    await websocket.send_json({"type": "status", "message": f"Searching primary source for {name}..."})
                else:
                    await websocket.send_json({"type": "status", "message": f"Searching background source #{total_urls_checked + 1} for missing education details..."})
                    
                # Check if we are retrying a specific platform
                if not primary_url_confirmed and primary_source_platform and platform_retries.get(primary_source_platform, 0) == 1:
                    prompt = f"The URL you provided for {primary_source_platform} returned no educational data. Please search again and provide a CORRECTED URL for {primary_source_platform}, or a different profile link on the same site for {name} at {hospital}."
                elif primary_url_confirmed and source_platform and platform_retries.get(source_platform, 0) == 1:
                    prompt = f"The URL you provided for {source_platform} returned no educational data. Please search again and provide a CORRECTED URL..."
                else:
                    prompt = f"Find a profile URL for Doctor Name: {name}, Hospital: {hospital}."
                    
                if additional_context:
                    prompt += f" Additional Context: {additional_context}"
                if tried_urls:
                    prompt += f" DO NOT return any of these URLs: {', '.join(tried_urls)}"
                    
                try:
                    result = await search_agent.run(prompt)
                    search_data = result.output
                    
                    if not search_data.profile_url:
                        if not primary_url_confirmed:
                             await websocket.send_json({"type": "search_failed", "reasoning": search_data.confidence_reasoning})
                             resp = await websocket.receive_text()
                             resp_payload = json.loads(resp)
                             if resp_payload.get("action") == "abort": return
                             additional_context = resp_payload.get("additional_context")
                             continue
                        else:
                             # If we can't find anything else in the background, break out and enrich what we have
                             break
                    else:
                         search_data.profile_url = search_data.profile_url.strip().replace(' ', '').replace('\n', '').replace('\r', '')
                         
                         if search_data.profile_url in tried_urls:
                             continue # Force LLM to generate a new URL instead of failing or alerting the user
                             
                         if not primary_url_confirmed:
                             # Ask user for manual confirmation ONLY until we have established a primary profile
                             await websocket.send_json({"type": "search_result", "data": search_data.model_dump()})
                             resp = await websocket.receive_text()
                             resp_payload = json.loads(resp)
                             if resp_payload.get("action") == "confirm":
                                 profile_url = search_data.profile_url
                                 source_platform = search_data.source_platform
                                 primary_source_platform = source_platform
                                 primary_url_confirmed = True
                                 break
                             elif resp_payload.get("action") == "refine":
                                 additional_context = resp_payload.get("additional_context")
                                 continue
                             else:
                                 return # Aborted
                         else:
                             # Background search automatically proceeds
                             profile_url = search_data.profile_url
                             source_platform = search_data.source_platform
                             break
                             
                except Exception as e:
                    if not primary_url_confirmed:
                        await websocket.send_json({"type": "error", "message": f"Search agent failed: {str(e)}"})
                        return
                    else:
                        break # Give up on background search if search agent errors
                        
            # If we broke out of the search loop because we can't find more URLs
            if not profile_url or profile_url in tried_urls:
                break
                
            tried_urls.append(profile_url)
            total_urls_checked += 1
            
            # -------------------------------------------------------------
            # Stage 2: Scrape & Verify Consistency
            # -------------------------------------------------------------
            await websocket.send_json({"type": "status", "message": f"Fetching markdown from {source_platform} via Jina..."})
            try:
                async with httpx.AsyncClient() as client:
                    jina_url = f"https://r.jina.ai/{profile_url}"
                    response = await client.get(jina_url, timeout=30.0)
                    if response.status_code != 200:
                        continue # Silently fail background scrape
                    markdown_content = response.text
                if not markdown_content:
                    continue # Silently fail if empty
                    
                # Consistency Check (Only against established primary profiles, i.e., URLs > 1 AND valid colleges found)
                if aggregated_colleges:
                    await websocket.send_json({"type": "status", "message": f"Verifying {source_platform} profile consistency..."})
                    primary_context = f"Doctor: {name}, Hospital: {hospital}, Source: {primary_source_platform}"
                    new_context = f"New Scraped URL: {profile_url}, Content Preview: {markdown_content[:2000]}"
                    consistency_prompt = f"Primary Profile Metadata:\n{primary_context}\n\nNew Profile Data:\n{new_context}"
                    
                    consistency = await consistency_agent.run(consistency_prompt)
                    if not consistency.output.is_same_doctor:
                        await websocket.send_json({"type": "status", "message": f"Skipping {source_platform}: Identified as a different doctor."})
                        continue
                        
                # -------------------------------------------------------------
                # Stage 3: Extract & Deduplicate
                # -------------------------------------------------------------
                await websocket.send_json({"type": "status", "message": f"Extracting education from {source_platform}..."})
                extract_result = await extraction_agent.run(markdown_content)
                extracted_data = extract_result.output
                
                # Handling empty extraction and Retry Logic
                if not extracted_data.colleges:
                    if platform_retries.get(source_platform, 0) < 1:
                        platform_retries[source_platform] = platform_retries.get(source_platform, 0) + 1
                        await websocket.send_json({"type": "status", "message": f"No educational data found on {source_platform}. Retrying to find a valid URL for this platform..."})
                        continue
                    else:
                        await websocket.send_json({"type": "status", "message": f"Retries exhausted. No educational data found on {source_platform}. Adding to blocklist and searching alternative platforms..."})
                        continue # Already added to tried_urls, loop back to find a new URL
                        
                # Quality Check: Ensure at least one college is properly named
                has_valid_college = False
                for c in extracted_data.colleges:
                    name_lower = c.name.lower().strip()
                    if len(name_lower) > 2 and "not specified" not in name_lower and "unspecified" not in name_lower:
                        has_valid_college = True
                        break
                        
                if not has_valid_college:
                    if platform_retries.get(source_platform, 0) < 1:
                        platform_retries[source_platform] = platform_retries.get(source_platform, 0) + 1
                        await websocket.send_json({"type": "status", "message": f"Data found on {source_platform} lacks institution names. Retrying to find a better URL..."})
                        continue
                    else:
                        await websocket.send_json({"type": "status", "message": f"Retries exhausted. Data found on {source_platform} lacks institution names. Rejecting and searching for an alternative..."})
                        continue

                # Merge logic
                if not registrations and extracted_data.registrations:
                    registrations = extracted_data.registrations
                    
                for c in extracted_data.colleges:
                    if not c.degree: continue
                    has_valid_name = len(c.name.lower().strip()) > 2 and "not specified" not in c.name.lower() and "unspecified" not in c.name.lower()
                    if has_valid_name and c.degree not in extracted_degrees:
                        extracted_degrees.add(c.degree)
                        aggregated_colleges.append(c)
                        
            except Exception as e:
                # Silently catch exceptions in background threads, unless it's the very first URL
                if not primary_url_confirmed:
                    await websocket.send_json({"type": "error", "message": f"Extraction failed on primary URL: {str(e)}"})
                    return
                continue

        # -------------------------------------------------------------
        # Stage 4: Enrichment & Final Result
        # -------------------------------------------------------------
        if not aggregated_colleges:
             await websocket.send_json({"type": "error", "message": "Failed to extract any valid colleges from all searched sources."})
             return
             
        await websocket.send_json({"type": "status", "message": f"Found {len(aggregated_colleges)} unique colleges across {total_urls_checked} sources. Routing to Tavily..."})
        
        final_colleges = []
        for college_obj in aggregated_colleges:
             await websocket.send_json({"type": "status", "message": f"Determining type for college: {college_obj.name}..."})
             enrich_result = await enrichment_agent.run(college_obj.model_dump_json())
             final_colleges.append(enrich_result.output)
             
        final_profile = DoctorFinalProfile(
             name=name,
             hospital=hospital,
             profile_url=tried_urls[0] if tried_urls else "", # primary URL
             source_platform=primary_source_platform or "",
             colleges=final_colleges,
             registrations=registrations
        )
        
        await websocket.send_json({"type": "status", "message": f"Operation complete. Hit {len(extracted_degrees)} distinct degrees after checking {total_urls_checked} URLs."})
        await websocket.send_json({
            "type": "final_result",
            "data": final_profile.model_dump()
        })
        
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
