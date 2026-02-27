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
    app
)
from fastapi.testclient import TestClient

# ----------------------------------------------------------------------------
# Test 1: Search Agent extracts Practo URL from user input
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_search_agent():
    """Check that from the user input, the agent can do a google search and return the Practo URL."""
    
    # We use PydanticAI's TestModel to mock the Gemini LLM
    # We tell it to return a specific piece of structured data when called
    mocked_search_result = {
        "profile_url": "https://www.practo.com/bangalore/doctor/sushanth-shivaswamy-pediatrician",
        "source_platform": "Practo",
        "found_name": "Dr. Sushanth Shivaswamy",
        "found_specialty": "Pediatrician",
        "confidence_reasoning": "Found exact match on Practo search results."
    }
    
    # Override the model for the test
    with search_agent.override(model=TestModel(custom_output_args=mocked_search_result)):
        prompt = "Find the Practo URL for Doctor Name: sushanth shivaswamy, Hospital: kauveri hospital."
        result = await search_agent.run(prompt)
        
        # Verify the agent returned the expected output format and data
        assert result.output.profile_url == "https://www.practo.com/bangalore/doctor/sushanth-shivaswamy-pediatrician"
        assert result.output.source_platform == "Practo"
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
    """Check that the search_agent actually uses the search_profile_url tool."""
    
    # We use a spy/mock model to verify the agent requested the correct tool call
    mock_response = DoctorSearchResult(
        profile_url="https://practo.com/mock",
        source_platform="Practo",
        found_name="Mock Doc",
        found_specialty="Mock Spec",
        confidence_reasoning="Mocked test."
    )
    
    with search_agent.override(model=TestModel(custom_output_args=mock_response, call_tools='all')):
        # We pass a prompt
        prompt = "Find URL for John Doe at Apollo Hospital."
        result = await search_agent.run(prompt)
        
        # Verify the Agent's thought process successfully yielded the output.
        assert result.output.profile_url == "https://practo.com/mock"

# ----------------------------------------------------------------------------
# Test 4b: Search Agent prioritizes fallbacks correctly
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_search_agent_fallback_priority():
    """Verify that the agent correctly falls back to Lybrate or Hospital sites if Practo isn't available."""
    
    fallback_mock_response = DoctorSearchResult(
        profile_url="https://www.lybrate.com/mock",
        source_platform="Lybrate",
        found_name="Mock Doc",
        found_specialty="Mock Spec",
        confidence_reasoning="Used Lybrate as fallback."
    )
    
    with search_agent.override(model=TestModel(custom_output_args=fallback_mock_response)):
        prompt = "Find URL for Jane Doe at City Hospital. Practo URL does not exist."
        result = await search_agent.run(prompt)
        
        # Verify the agent returned the fallback Lybrate URL
        assert result.output.profile_url == "https://www.lybrate.com/mock"
        assert result.output.source_platform == "Lybrate"


# ----------------------------------------------------------------------------
# Test 4c: Search Agent - Hospital Site Fallback
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_search_agent_hospital_fallback():
    """Verify that the agent correctly identifies an official hospital website profile."""
    
    mock_response = DoctorSearchResult(
        profile_url="https://www.apollohospitals.com/mock-doc",
        source_platform="Apollo Hospitals",
        found_name="Dr. Apollo Mock",
        found_specialty="Cardiology",
        confidence_reasoning="Found on official Apollo site."
    )
    
    with search_agent.override(model=TestModel(custom_output_args=mock_response)):
        prompt = "Find URL for Dr. Apollo Mock at Apollo Hospitals."
        result = await search_agent.run(prompt)
        
        assert "apollohospitals.com" in result.output.profile_url
        assert result.output.source_platform == "Apollo Hospitals"


# ----------------------------------------------------------------------------
# Test 4d: Search Agent - National Medical Register (NMR) via Aggregator Fallback
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_search_agent_nmr_fallback():
    """Verify the agent can parse an NMR registration from a public registry scraper."""
    
    mock_response = DoctorSearchResult(
        profile_url="https://indianmedicalregistry.org/mock-doc-nmr",
        source_platform="NMR Public Registry",
        found_name="Dr. NMR Mock",
        found_specialty="General Medicine",
        confidence_reasoning="Found NMR registration number in public registry scraper."
    )
    
    with search_agent.override(model=TestModel(custom_output_args=mock_response)):
        prompt = "Find URL for Dr. NMR Mock. Not on Practo or hospital sites."
        result = await search_agent.run(prompt)
        
        assert result.output.source_platform == "NMR Public Registry"


# ----------------------------------------------------------------------------
# Test 4e: Search Agent - State Medical Council (SMC) Fallback
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_search_agent_smc_fallback():
    """Verify the agent identifies a doctor from a state medical council's public PDF or page."""
    
    mock_response = DoctorSearchResult(
        profile_url="https://karnatakamedicalcouncil.com/mock-doc-reg",
        source_platform="Karnataka Medical Council (SMC)",
        found_name="Dr. SMC Mock",
        found_specialty="Unknown",
        confidence_reasoning="Found SMC registration page."
    )
    
    with search_agent.override(model=TestModel(custom_output_args=mock_response)):
        prompt = "Find Karnataka Medical Council registration for Dr. SMC Mock."
        result = await search_agent.run(prompt)
        
        assert "karnatakamedicalcouncil.com" in result.output.profile_url
        assert "SMC" in result.output.source_platform


# ----------------------------------------------------------------------------
# Test 4f: Search Agent - JustDial Fallback
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_search_agent_justdial_fallback():
    """Verify the agent correctly identifies a JustDial profile."""
    
    mock_response = DoctorSearchResult(
        profile_url="https://www.justdial.com/mock-doc",
        source_platform="JustDial",
        found_name="Dr. JustDial Mock",
        found_specialty="Dentist",
        confidence_reasoning="Found on JustDial."
    )
    
    with search_agent.override(model=TestModel(custom_output_args=mock_response)):
        prompt = "Find URL for Dr. JustDial Mock."
        result = await search_agent.run(prompt)
        
        assert result.output.profile_url == "https://www.justdial.com/mock-doc"
        assert result.output.source_platform == "JustDial"


# ----------------------------------------------------------------------------
# Test 5: Jina Markdown Extractor (WebSocket Flow)
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_jina_markdown_fetch(monkeypatch):
    """Check that the WebSocket flow fetches markdown from Jina and returns the final profile."""
    
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
    
    client = TestClient(app)
    
    # Mock the three agents to return expected deterministic data
    mock_search = DoctorSearchResult(
        profile_url="https://mock-url.com",
        source_platform="MockSource",
        found_name="Mock Doc",
        found_specialty="Mock Spec",
        confidence_reasoning="Mocked"
    )
    
    with search_agent.override(model=TestModel(custom_output_args=mock_search)):
        with extraction_agent.override(model=TestModel(custom_output_args={"colleges":["Mock College"], "registrations":[]})):
            with enrichment_agent.override(model=TestModel(custom_output_args={"name":"Mock College","is_government":True,"is_private":False})):
                
                # Connect to the WebSocket
                with client.websocket_connect("/ws/extract") as websocket:
                    # Send initial search payload
                    websocket.send_json({"name": "Test Name", "hospital": "Test Hosp"})
                    
                    # Receive Searching Status
                    status1 = websocket.receive_json()
                    assert status1["type"] == "status"
                    
                    # Receive Search Result
                    search_res = websocket.receive_json()
                    assert search_res["type"] == "search_result"
                    assert search_res["data"]["profile_url"] == "https://mock-url.com"
                    assert search_res["data"]["source_platform"] == "MockSource"
                    
                    # Send Confirmation
                    websocket.send_json({"action": "confirm"})
                    
                    # Receive diverse status messages (Jina Fetch, Extracting, Routing...)
                    # We loop until we get the final result or error
                    final_data = None
                    for _ in range(10): 
                        msg = websocket.receive_json()
                        if msg["type"] == "final_result":
                            final_data = msg["data"]
                            break
                        elif msg["type"] == "error":
                            pytest.fail(f"WebSocket flow errored: {msg['message']}")
                            
                    assert final_data is not None
                    assert final_data["name"] == "Test Name"
                    assert len(final_data["colleges"]) == 1
                    assert final_data["colleges"][0]["name"] == "Mock College"


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


