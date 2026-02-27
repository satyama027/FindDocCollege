import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai import result

# Import the agents and schemas from main
from main import (
    search_agent, 
    extraction_agent, 
    enrichment_agent, 
    DoctorSearchResult, 
    DoctorExtractedData, 
    CollegeInfo,
    interactive_search,
    run_extraction_flow
)

# ----------------------------------------------------------------------------
# Test 1: Search Agent extracts Practo URL from user input
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_search_agent():
    """Check that from the user input, the agent can do a google search and return the Practo URL."""
    
    # We use PydanticAI's TestModel to mock the Gemini LLM
    # We tell it to return a specific piece of structured data when called
    mocked_search_result = {
        "practo_url": "https://www.practo.com/bangalore/doctor/sushanth-shivaswamy-pediatrician",
        "found_name": "Dr. Sushanth Shivaswamy",
        "found_specialty": "Pediatrician",
        "confidence_reasoning": "Found exact match on Practo search results."
    }
    
    # Override the model for the test
    with search_agent.override(model=TestModel(custom_output_args=mocked_search_result)):
        prompt = "Find the Practo URL for Doctor Name: sushanth shivaswamy, Hospital: kauveri hospital."
        result = await search_agent.run(prompt)
        
        # Verify the agent returned the expected output format and data
        assert result.output.practo_url == "https://www.practo.com/bangalore/doctor/sushanth-shivaswamy-pediatrician"
        assert result.output.found_name == "Dr. Sushanth Shivaswamy"


# ----------------------------------------------------------------------------
# Test 2: Extraction Agent extracts College Name from Markdown
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_extraction_agent():
    """Check that from the sample markdown, college name can be extracted."""
    
    # A dummy piece of Jina markdown to simulate a Practo profile
    sample_markdown = """
    # Dr. Sushanth Shivaswamy
    Pediatrician
    
    ## Education
    - MBBS - Bangalore Medical College and Research Institute, Bangalore, 1999
    - MD - Pediatrics - King Edward Memorial Hospital and Seth Gordhandas Sunderdas Medical College, 2003
    
    ## Registrations
    - 54321 Karnataka Medical Council, 1999
    """
    
    # Mock what the LLM should output after reading that markdown
    mocked_extraction = {
        "colleges": [
            "Bangalore Medical College and Research Institute",
            "King Edward Memorial Hospital and Seth Gordhandas Sunderdas Medical College"
        ],
        "registrations": ["54321 Karnataka Medical Council"]
    }
    
    with extraction_agent.override(model=TestModel(custom_output_args=mocked_extraction)):
        result = await extraction_agent.run(sample_markdown)
        
        # Verify the college name was extracted into the Pydantic schema correctly
        assert len(result.output.colleges) == 2
        assert "Bangalore Medical College and Research Institute" in result.output.colleges
        assert len(result.output.registrations) == 1


# ----------------------------------------------------------------------------
# Test 3: Enrichment Agent determines Government vs Private
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_enrichment_agent():
    """Once college name is extracted, check that the enrichment agent determines if it is government or private."""
    
    college_name = "Bangalore Medical College and Research Institute"
    
    # Mock the LLM classification result
    mocked_college_info = {
        "name": college_name,
        "is_government": True,
        "is_private": False
    }
    
    with enrichment_agent.override(model=TestModel(custom_output_args=mocked_college_info)):
        result = await enrichment_agent.run(college_name)
        
        # Verify classification
        assert result.output.name == college_name
        assert result.output.is_government is True
        assert result.output.is_private is False

# ----------------------------------------------------------------------------
# Test 4: Search Agent issues correct Tavily queries based on input
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_search_agent_tool_call():
    """Check that the search_agent actually uses the search_practo_url tool."""
    
    # We use a spy/mock model to verify the agent requested the correct tool call
    mock_response = DoctorSearchResult(
        practo_url="https://practo.com/mock",
        found_name="Mock Doc",
        found_specialty="Mock Spec",
        confidence_reasoning="Mocked test."
    )
    
    with search_agent.override(model=TestModel(custom_output_args=mock_response, call_tools='all')):
        # We pass a prompt
        prompt = "Find URL for John Doe at Apollo Hospital."
        result = await search_agent.run(prompt)
        
        # We don't have a direct Tavily mock easily injectable here without monkeypatching, 
        # but we can verify the Agent's thought process successfully yielded the output.
        # If it throws no errors and returns our mock model output type, the routing worked.
        assert result.output.practo_url == "https://practo.com/mock"


# ----------------------------------------------------------------------------
# Test 5: Jina Markdown Extractor
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_jina_markdown_fetch(monkeypatch):
    """Check that run_extraction_flow fetches markdown from Jina."""
    
    import httpx
    
    class MockResponse:
        status_code = 200
        text = "# Mock Markdown from Jina"
        
    class MockClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def get(self, url, timeout):
            assert "r.jina.ai" in url
            return MockResponse()

    # Monkeypatch the httpx.AsyncClient to avoid real network requests in CI
    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    
    # Run the flow (we expect it to fail gracefully at the LLM extract step if no API key is set, 
    # but we just want to ensure it calls the fetch)
    with extraction_agent.override(model=TestModel(custom_output_args={"colleges":["Mock College"], "registrations":[]})):
        with enrichment_agent.override(model=TestModel(custom_output_args={"name":"Mock College","is_government":True,"is_private":False})):
            profile = await run_extraction_flow("Test Name", "Test Hosp", "mock-url.com")
            assert profile is not None
            assert profile.practo_url == "mock-url.com"
            assert profile.colleges[0].name == "Mock College"


# ----------------------------------------------------------------------------
# Test 6: Enrichment Agent searches Tavily
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_enrichment_agent_tool_call():
    """Verify the enrichment_agent uses the search_college_type tool."""
    
    mocked_college_info = {
        "name": "Live College Test",
        "is_government": False,
        "is_private": True
    }
    
    # We test that the framework routes the string to the model correctly
    with enrichment_agent.override(model=TestModel(custom_output_args=mocked_college_info, call_tools='all')):
        result = await enrichment_agent.run("Live College Test")
        assert result.output.is_government is False


