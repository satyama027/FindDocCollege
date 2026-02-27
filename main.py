import os
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import httpx
from tavily import TavilyClient
from dotenv import load_dotenv

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
    practo_url: Optional[str] = Field(description="The Practo URL found for the doctor. None if not found.")
    found_name: Optional[str] = Field(description="The name of the doctor as it appeared in the search results.")
    found_specialty: Optional[str] = Field(description="The specialty of the doctor as it appeared in the search results.")
    confidence_reasoning: str = Field(description="Brief explanation of why this URL was chosen or why it was not found.")

class DoctorExtractedData(BaseModel):
    """Data extracted directly from the Firecrawl scrape"""
    colleges: List[str] = Field(description="List of college or university names where the given doctor studied.")
    registrations: List[str] = Field(description="Registration details, such as registration number and medical council.")

class CollegeInfo(BaseModel):
    name: str = Field(description="Name of the college or university")
    is_government: bool = Field(description="True if it is a government/public institution")
    is_private: bool = Field(description="True if it is a private institution")

class DoctorFinalProfile(BaseModel):
    """The final structured output of the agent"""
    name: str
    hospital: str
    practo_url: str
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
        "You are an expert search assistant. Your goal is to find the correct Practo profile URL "
        "for a specific doctor based on the user's provided details. "
        "Use the `search_practo_url` tool to query the web. "
        "Analyze the search snippets to ensure the URL matches the doctor's name, hospital, and any additional context. "
        "If you find a matching URL, extract the doctor's name and specialty from the snippet and return them."
    )
)

@search_agent.tool
async def search_practo_url(ctx: RunContext[None], query: str) -> str:
    """Searches the web using Tavily to find Practo profile URLs."""
    if not tavily_client:
        return "Error: Tavily client not initialized (missing API key)."
    try:
        # Force site:practo.com in the query if the agent didn't include it
        if "site:practo.com" not in query:
             query += " site:practo.com"
        search_result = tavily_client.search(query=query, search_depth="basic")
        results = search_result.get('results', [])
        # Return a formatted string of the top results so the agent can analyze them
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
        "colleges (where they studied for their degrees) and their medical registrations. "
        "Return the exact names of the colleges and the exact text of the registrations."
        "If they are not found, return empty lists."
    )
)

# 3. Enrichment Agent
enrichment_agent = Agent(
    MODEL,
    output_type=CollegeInfo,
    system_prompt=(
        "You are an expert researcher. I will provide you with the name of a medical college in India. "
        "You must determine if it is a government (public) institution or a private institution. "
        "Use the `search_college_type` tool to search the web for evidence. "
        "Based on the search results, return the CollegeInfo, setting is_government and is_private appropriately."
    )
)

@enrichment_agent.tool
async def search_college_type(ctx: RunContext[None], college_name: str) -> str:
    """Uses Tavily to search the web and gather context about a specific medical college."""
    query = f"Is {college_name} a government or private medical college in India?"
    if not tavily_client:
        return "Error: Tavily client not initialized."
    try:
        search_result = tavily_client.search(query=query, search_depth="basic")
        snippets = [result['content'] for result in search_result.get('results', [])]
        return "\n".join(snippets)
    except Exception as e:
        return f"Error searching: {e}"

# ---------------------------------------------------------
# Application Flow
# ---------------------------------------------------------

async def interactive_search(name: str, hospital: str) -> str:
    """Handles the interactive loop to find and confirm the Practo URL."""
    additional_context = None
    
    while True:
        print(f"\n[Search] Looking for Practo URL for {name} at {hospital}...")
        
        # Build prompt for the search agent
        prompt = f"Find the Practo URL for Doctor Name: {name}, Hospital: {hospital}."
        if additional_context:
            prompt += f" Additional Context: {additional_context}"
            
        try:
            result = await search_agent.run(prompt)
            search_data = result.output
            
            if not search_data.practo_url:
                print(f"[Search Failed] Could not find a matching URL. Reason: {search_data.confidence_reasoning}")
            else:
                print("\n[Search Result]")
                print(f"URL:       {search_data.practo_url}")
                print(f"Name:      {search_data.found_name}")
                print(f"Specialty: {search_data.found_specialty}")
                print(f"Reasoning: {search_data.confidence_reasoning}")
            
            user_input = input("\nIs this the correct doctor? (y / n / provide more context to retry): ").strip().lower()
            
            if user_input == 'y' or user_input == 'yes':
                if search_data.practo_url:
                     return search_data.practo_url
                else:
                     print("Cannot proceed without a valid URL. Please provide more context.")
                     additional_context = input("Enter additional details (e.g., city, specific clinic): ")
            elif user_input == 'n' or user_input == 'no':
                 additional_context = input("Please enter additional details to refine the search (e.g., city, specific clinic): ")
            else:
                 # User provided context directly
                 additional_context = user_input
                 
        except Exception as e:
            print(f"[Agent Error] Search agent failed: {e}")
            return None


async def run_extraction_flow(name: str, hospital: str, practo_url: str) -> DoctorFinalProfile:
    """Runs the Jina scraping, LLM extraction, and enrichment pipeline."""
    
    print(f"\n[Scrape] Fetching markdown from {practo_url} using Jina Reader...")
    
    try:
        # Request markdown via Jina Reader API
        async with httpx.AsyncClient() as client:
            jina_url = f"https://r.jina.ai/{practo_url}"
            response = await client.get(jina_url, timeout=30.0)
            
            if response.status_code != 200:
                print(f"[Scrape Error] Failed to get markdown content. Status code: {response.status_code}")
                return None
                
            markdown_content = response.text
            
        if not markdown_content:
             print("[Scrape Error] Markdown content is empty.")
             return None
             
        print("\n[Extract] Analyzing profile with Gemini...")
        extract_result = await extraction_agent.run(markdown_content)
        extracted_data = extract_result.output
        
        print(f"[Extract Success] Found {len(extracted_data.colleges)} colleges and {len(extracted_data.registrations)} registrations.")
        
        final_colleges = []
        for college in extracted_data.colleges:
             print(f"\n[Enrich] Determining type for college: {college}...")
             enrich_result = await enrichment_agent.run(college)
             final_colleges.append(enrich_result.output)
             
        # Construct and return final profile
        return DoctorFinalProfile(
             name=name,
             hospital=hospital,
             practo_url=practo_url,
             colleges=final_colleges,
             registrations=extracted_data.registrations
        )
             
    except Exception as e:
        print(f"[Pipeline Error] Extraction failed: {e}")
        return None

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

async def main():
    print("=== Doctor Data Extraction Agent ===")
    
    # Get basic details
    name = input("Enter Doctor's Name: ").strip()
    if not name:
        name = "Dr. Sample" # Default for quick testing
        
    hospital = input("Enter Hospital Name: ").strip()
    if not hospital:
        hospital = "Apollo Hospital"
    
    # Stage 1: Search & Confirm
    practo_url = await interactive_search(name, hospital)
    
    if not practo_url:
        print("Flow aborted.")
        return
        
    # Stage 2 & 3: Extract & Enrich
    final_profile = await run_extraction_flow(name, hospital, practo_url)
    
    if final_profile:
        print("\n=== Final Extracted Profile ===")
        print(final_profile.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())
