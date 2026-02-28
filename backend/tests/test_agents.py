import pytest
import json
from pydantic_ai.models.test import TestModel
from pydantic_ai import result

# Import the agents and schemas from main
from main import (
    search_agent, 
    extraction_agent, 
    enrichment_agent, 
    consistency_agent,
    DoctorSearchResult, 
    DoctorExtractedData, 
    CollegeInfo,
    ConsistencyResult,
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
            {"name": "Bangalore Medical College and Research Institute", "degree": "MBBS", "year": 1999},
            {"name": "King Edward Memorial Hospital and Seth Gordhandas Sunderdas Medical College", "degree": "MD - Pediatrics", "year": 2003}
        ],
        "registrations": ["54321 Karnataka Medical Council"]
    }
    
    with extraction_agent.override(model=TestModel(custom_output_args=mocked_extraction)):
        result = await extraction_agent.run(sample_markdown)
        
        # Verify the college name was extracted into the Pydantic schema correctly
        assert len(result.output.colleges) == 2
        assert result.output.colleges[0].name == "Bangalore Medical College and Research Institute"
        assert result.output.colleges[0].degree == "MBBS"
        assert result.output.colleges[0].year == 1999
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
        "degree": "MBBS",
        "year": 1999,
        "is_government": True,
        "is_private": False
    }
    
    with enrichment_agent.override(model=TestModel(custom_output_args=mocked_college_info)):
        result = await enrichment_agent.run(json.dumps({"name": college_name, "degree": "MBBS", "year": 1999}))
        
        # Verify classification
        assert result.output.name == college_name
        assert result.output.degree == "MBBS"
        assert result.output.year == 1999
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
        with extraction_agent.override(model=TestModel(custom_output_args={"colleges":[{"name": "Mock College", "degree": "MBBS", "year": 2000}, {"name": "Mock 2", "degree": "MD", "year": 2008}], "registrations":[]})):
            with enrichment_agent.override(model=TestModel(custom_output_args={"name":"Mock College","degree":"MBBS","year":2000,"is_government":True,"is_private":False})):
                
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
                    import time
                    start_t = time.time()
                    while True:
                        if time.time() - start_t > 5:
                            pytest.fail("Test timed out waiting for final result.")
                        msg = websocket.receive_json()
                        if msg["type"] == "final_result":
                            final_data = msg["data"]
                            break
                        elif msg["type"] == "error":
                            pytest.fail(f"WebSocket flow errored: {msg['message']}")
                            
                    assert final_data is not None
                    assert final_data["name"] == "Test Name"
                    assert len(final_data["colleges"]) == 2
                    assert final_data["colleges"][0]["name"] == "Mock College"


# ----------------------------------------------------------------------------
# Test 6: Enrichment Agent searches Tavily
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_enrichment_agent_tool_call():
    """Verify the enrichment_agent uses the search_college_type tool."""
    
    mocked_college_info = {
        "name": "Live College Test",
        "degree": "MD",
        "year": 2010,
        "is_government": False,
        "is_private": True
    }
    
    with enrichment_agent.override(model=TestModel(custom_output_args=mocked_college_info, call_tools='all')):
        result = await enrichment_agent.run(json.dumps({"name": "Live College Test", "degree": "MD", "year": 2010}))
        assert result.output.is_government is False


# ----------------------------------------------------------------------------
# Test 7: Extraction Fallback Loop
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_extraction_fallback_loop(monkeypatch):
    """Verify that if the first extracted profile yields 0 colleges, the agent searches again."""
    
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
            return MockResponse()

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    client = TestClient(app)
    
    class MockResponse:
        status_code = 200
        text = "# Mock Markdown from Jina"
        
    class MockClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def get(self, url, timeout):
            return MockResponse()

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    
    # Create Mocks
    mock_search_1 = DoctorSearchResult(
        profile_url="https://justdial.com/mock",
        source_platform="JustDial",
        found_name="Mock Doc",
        found_specialty="Mock Spec",
        confidence_reasoning="Found JustDial"
    )
    mock_search_2 = DoctorSearchResult(
        profile_url="https://lybrate.com/mock",
        source_platform="Lybrate",
        found_name="Mock Doc",
        found_specialty="Mock Spec",
        confidence_reasoning="Found Lybrate"
    )

    search_calls = [0]
    async def mock_search_run(prompt: str, **kwargs):
        class MockRunResult:
            def __init__(self, data):
                self.output = data
        if "JustDial" not in str(prompt) and "Lybrate" not in str(prompt) and search_calls[0] == 0:
            search_calls[0] += 1
            return MockRunResult(mock_search_1)
        search_calls[0] += 1
        return MockRunResult(mock_search_2)

    extract_calls = [0]
    async def mock_extract_run(content: str, **kwargs):
        class MockExtractResult:
            def __init__(self, data):
                self.output = DoctorExtractedData(**data)
        if extract_calls[0] == 0:
            extract_calls[0] += 1
            return MockExtractResult({"colleges": [], "registrations": []})
        return MockExtractResult({"colleges": [{"name": "Mock College", "degree": "MBBS", "year": 2005}, {"name": "Mock 2", "degree": "MD", "year": 2008}], "registrations": ["Mock Reg"]})

    async def mock_enrich_run(college_json: str, **kwargs):
        class MockEnrichResult:
            def __init__(self):
                self.output = CollegeInfo(name="Mock College", degree="MBBS", year=2005, is_government=True, is_private=False)
        return MockEnrichResult()

    import main
    monkeypatch.setattr(main.search_agent, "run", mock_search_run)
    monkeypatch.setattr(main.extraction_agent, "run", mock_extract_run)
    monkeypatch.setattr(main.enrichment_agent, "run", mock_enrich_run)
    
    with client.websocket_connect("/ws/extract") as websocket:
        # 1. Initial Search
        websocket.send_json({"name": "Test Name", "hospital": "Test Hosp"})
        
        # We need to loop receiving until we find the first search_result
        import time
        start = time.time()
        res1 = None
        while True:
            if time.time() - start > 5:
                pytest.fail("Test timed out waiting for FIRST search result.")
            msg = websocket.receive_json()
            if msg["type"] == "search_result":
                res1 = msg
                break
                
        assert res1["data"]["source_platform"] == "JustDial"
        
        # 2. Confirm First Result -> Leads to 0 Colleges
        websocket.send_json({"action": "confirm"})
        
        # Receive status messages until the loop triggers the SECOND search and extracts automatically
        start2 = time.time()
        final_data = None
        while True:
            if time.time() - start2 > 5:
                pytest.fail("Test timed out waiting for final result.")
                
            msg = websocket.receive_json()
            if msg["type"] == "final_result":
                final_data = msg["data"]
                break
            elif msg["type"] == "error":
                pytest.fail(f"Agent errored. {msg}")
                
        assert final_data["source_platform"] == "JustDial"
        assert len(final_data["colleges"]) == 2

# ----------------------------------------------------------------------------
# Test 8: Extraction Fallback for Poor Quality Data (No valid institution name)
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_extraction_fallback_quality_check(monkeypatch):
    """Verify that if the extracted college name is invalid ('unspecified'), it falls back to a new search."""
    
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
            return MockResponse()

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    client = TestClient(app)
    
    mock_search_1 = DoctorSearchResult(
        profile_url="https://cloudnine.com/mock",
        source_platform="Cloudnine Hospitals",
        found_name="Mock Doc",
        found_specialty="Mock Spec",
        confidence_reasoning="Found Cloudnine"
    )
    mock_search_2 = DoctorSearchResult(
        profile_url="https://practo.com/mock",
        source_platform="Practo",
        found_name="Mock Doc",
        found_specialty="Mock Spec",
        confidence_reasoning="Found Practo"
    )

    search_calls = [0]
    async def mock_search_run(prompt: str, **kwargs):
        class MockRunResult:
            def __init__(self, data):
                self.output = data
        if "cloudnine.com/mock" not in str(prompt) and search_calls[0] == 0:
            search_calls[0] += 1
            return MockRunResult(mock_search_1)
        search_calls[0] += 1
        return MockRunResult(mock_search_2)

    extract_calls = [0]
    async def mock_extract_run(content: str, **kwargs):
        class MockExtractResult:
            def __init__(self, data):
                self.output = DoctorExtractedData(**data)
        if extract_calls[0] == 0:
            extract_calls[0] += 1
            # First extraction returns a valid degree but NO VALID COLLEGE NAME
            return MockExtractResult({"colleges": [{"name": "not specified", "degree": "MBBS", "year": 2005}], "registrations": ["Mock Reg"]})
        # Second extraction returns a VALID college
        return MockExtractResult({"colleges": [{"name": "Mock Medical College", "degree": "MBBS", "year": 2005}, {"name": "Mock 2", "degree": "MD", "year": 2008}], "registrations": ["Mock Reg"]})

    async def mock_enrich_run(college_json: str, **kwargs):
        class MockEnrichResult:
            def __init__(self):
                self.output = CollegeInfo(name="Mock Medical College", degree="MBBS", year=2005, is_government=True, is_private=False)
        return MockEnrichResult()

    import main
    monkeypatch.setattr(main.search_agent, "run", mock_search_run)
    monkeypatch.setattr(main.extraction_agent, "run", mock_extract_run)
    monkeypatch.setattr(main.enrichment_agent, "run", mock_enrich_run)
    
    with client.websocket_connect("/ws/extract") as websocket:
        websocket.send_json({"name": "Test Name", "hospital": "Test Hosp"})
        
        import time
        start = time.time()
        res1 = None
        while True:
            if time.time() - start > 5:
                pytest.fail("Test timed out.")
            msg = websocket.receive_json()
            if msg["type"] == "search_result":
                res1 = msg
                break
                
        assert res1["data"]["source_platform"] == "Cloudnine Hospitals"
        websocket.send_json({"action": "confirm"})
        
        start2 = time.time()
        final_data = None
        while True:
            if time.time() - start2 > 5:
                pytest.fail("Test timed out waiting for final result.")
            msg = websocket.receive_json()
            if msg["type"] == "final_result":
                final_data = msg["data"]
                break
            elif msg["type"] == "error":
                pytest.fail(f"Agent errored. {msg}")
                
        assert final_data["source_platform"] == "Cloudnine Hospitals"
        assert len(final_data["colleges"]) == 2
        assert final_data["colleges"][0]["name"] == "Mock Medical College"

# ----------------------------------------------------------------------------
# Test 9: Retry Logic - Fixing a Malformed URL for the same platform
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_retry_malformed_url(monkeypatch):
    """Verify that if the first URL for an extracted profile yields 0 colleges, the system Retries the search instead of abandoning the platform."""
    
    import httpx
    
    class MockResponse:
        status_code = 200
        text = "# Mock Markdown from Jina"
        
    class MockClient:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc_val, exc_tb): pass
        async def get(self, url, timeout): return MockResponse()

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    client = TestClient(app)
    
    # Same platform, but the first search is "malformed" and the retry is "fixed"
    mock_search_malformed = DoctorSearchResult(
        profile_url="https://practo.com/malformed",
        source_platform="Practo",
        found_name="Mock Doc",
        found_specialty="Mock Spec",
        confidence_reasoning="Found Practo"
    )
    mock_search_fixed = DoctorSearchResult(
        profile_url="https://practo.com/fixed",
        source_platform="Practo",
        found_name="Mock Doc",
        found_specialty="Mock Spec",
        confidence_reasoning="Retried and found valid URL"
    )

    search_calls = [0]
    async def mock_search_run(prompt: str, **kwargs):
        class MockRunResult:
            def __init__(self, data): self.output = data
        if search_calls[0] == 0:
            search_calls[0] += 1
            return MockRunResult(mock_search_malformed)
        search_calls[0] += 1
        return MockRunResult(mock_search_fixed)

    extract_calls = [0]
    async def mock_extract_run(content: str, **kwargs):
        class MockExtractResult:
            def __init__(self, data): self.output = DoctorExtractedData(**data)
        if extract_calls[0] == 0:
            extract_calls[0] += 1
            return MockExtractResult({"colleges": [], "registrations": []})
        return MockExtractResult({"colleges": [{"name": "Mock Medical College", "degree": "MBBS", "year": 2005}, {"name": "Mock 2", "degree": "MD", "year": 2008}], "registrations": ["Mock Reg"]})

    async def mock_enrich_run(college_json: str, **kwargs):
        class MockEnrichResult:
            def __init__(self): self.output = CollegeInfo(name="Mock Medical College", degree="MBBS", year=2005, is_government=True, is_private=False)
        return MockEnrichResult()

    import main
    monkeypatch.setattr(main.search_agent, "run", mock_search_run)
    monkeypatch.setattr(main.extraction_agent, "run", mock_extract_run)
    monkeypatch.setattr(main.enrichment_agent, "run", mock_enrich_run)
    
    with client.websocket_connect("/ws/extract") as websocket:
        websocket.send_json({"name": "Test Name", "hospital": "Test Hosp"})
        
        # 1. First Attempt (Malformed Practo Data)
        res1 = None
        while True:
            msg = websocket.receive_json()
            if msg["type"] == "search_result":
                res1 = msg
                break
        assert res1["data"]["profile_url"] == "https://practo.com/malformed"
        websocket.send_json({"action": "confirm"})
        
        # 2. Key Assertion: Expect Status message indicating a Retry, followed by final result
        retry_detected = False
        final_data = None
        while True:
            msg = websocket.receive_json()
            if msg["type"] == "status" and "Retrying" in msg["message"]:
                retry_detected = True
            elif msg["type"] == "final_result":
                final_data = msg["data"]
                break
                
        assert retry_detected is True, "The system completely skipped the URL Retry logic!"
        assert final_data["source_platform"] == "Practo"
        assert len(final_data["colleges"]) == 2

# ----------------------------------------------------------------------------
# Test 10: Retry Logic - Rejection after Retries Exhausted
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rejection_after_retries_exhausted(monkeypatch):
    """Verify that if the Retry ALSO fails to produce valid colleges, the system finally rejects the platform and falls back."""
    
    import httpx
    
    class MockResponse:
        status_code = 200
        text = "# Mock Markdown from Jina"
        
    class MockClient:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc_val, exc_tb): pass
        async def get(self, url, timeout): return MockResponse()

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    client = TestClient(app)
    
    mock_search_practo1 = DoctorSearchResult(profile_url="https://practo.com/bad1", source_platform="Practo", found_name="Mock Doc", found_specialty="Mock Spec", confidence_reasoning="")
    mock_search_practo2 = DoctorSearchResult(profile_url="https://practo.com/bad2", source_platform="Practo", found_name="Mock Doc", found_specialty="Mock Spec", confidence_reasoning="")
    mock_search_lybrate = DoctorSearchResult(profile_url="https://lybrate.com/good", source_platform="Lybrate", found_name="Mock Doc", found_specialty="Mock Spec", confidence_reasoning="")

    search_calls = [0]
    async def mock_search_run(prompt: str, **kwargs):
        class MockRunResult:
            def __init__(self, data): self.output = data
        if search_calls[0] == 0:
            search_calls[0] += 1
            return MockRunResult(mock_search_practo1)
        elif search_calls[0] == 1:
            search_calls[0] += 1
            return MockRunResult(mock_search_practo2)
        return MockRunResult(mock_search_lybrate)

    extract_calls = [0]
    async def mock_extract_run(content: str, **kwargs):
        class MockExtractResult:
            def __init__(self, data): self.output = DoctorExtractedData(**data)
        if extract_calls[0] < 2:
            extract_calls[0] += 1
            return MockExtractResult({"colleges": [], "registrations": []})
        return MockExtractResult({"colleges": [{"name": "Mock Medical College", "degree": "MBBS", "year": 2005}, {"name": "Mock 2", "degree": "MD", "year": 2008}], "registrations": ["Mock Reg"]})

    async def mock_enrich_run(college_json: str, **kwargs):
        class MockEnrichResult:
            def __init__(self): self.output = CollegeInfo(name="Mock Medical College", degree="MBBS", year=2005, is_government=True, is_private=False)
        return MockEnrichResult()

    import main
    monkeypatch.setattr(main.search_agent, "run", mock_search_run)
    monkeypatch.setattr(main.extraction_agent, "run", mock_extract_run)
    monkeypatch.setattr(main.enrichment_agent, "run", mock_enrich_run)
    
    with client.websocket_connect("/ws/extract") as websocket:
        websocket.send_json({"name": "Test Name", "hospital": "Test Hosp"})
        
        # 1. First Attempt (Practo Bad)
        while True:
            if websocket.receive_json()["type"] == "search_result": break
        websocket.send_json({"action": "confirm"})
        
        # 2. Second Attempt (Retry - Practo Bad 2) is automatically scraped in background.
        
        # 3. Third Attempt (Lybrate - Good) and verify rejection triggered
        rejection_detected = False
        final_data = None
        while True:
            msg = websocket.receive_json()
            if msg["type"] == "status" and ("Retries exhausted" in msg["message"] or "exhausted" in msg["message"]):
                rejection_detected = True
            elif msg["type"] == "final_result":
                final_data = msg["data"]
                break
                
        assert rejection_detected is True, "System did not emit standard rejection message after retries failed."
        assert final_data["source_platform"] == "Practo"
        assert len(final_data["colleges"]) == 2


# ============================================================================
# PHASE 4: MULTI-URL AGGREGATION TESTS (TDD)
# ============================================================================

# Test 11: Combine information from multiple sources (Dr. Abhishek Bansal)
@pytest.mark.asyncio
async def test_aggregation_combines_sources_abhishek_bansal(monkeypatch):
    """Verify system combines MBBS from URL 1 and MD from URL 2 for Abhishek Bansal."""
    
    import httpx
    class MockResponse:
        status_code = 200
        text = "# Mock Markdown"
    class MockClient:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc_val, exc_tb): pass
        async def get(self, url, timeout): return MockResponse()
    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    client = TestClient(app)
    
    mock_search_1 = DoctorSearchResult(profile_url="https://practo.com/abhishek", source_platform="Practo", found_name="Abhishek Bansal", found_specialty="Pediatrician", confidence_reasoning="")
    mock_search_2 = DoctorSearchResult(profile_url="https://lybrate.com/abhishek", source_platform="Lybrate", found_name="Abhishek Bansal", found_specialty="Pediatrician", confidence_reasoning="")
    
    search_calls = [0]
    async def mock_search_run(prompt: str, **kwargs):
        class MockRunResult:
            def __init__(self, data): self.output = data
        search_calls[0] += 1
        return MockRunResult(mock_search_1 if search_calls[0] == 1 else mock_search_2)

    extract_calls = [0]
    async def mock_extract_run(content: str, **kwargs):
        class MockExtractResult:
            def __init__(self, data): self.output = DoctorExtractedData(**data)
        extract_calls[0] += 1
        if extract_calls[0] == 1:
            return MockExtractResult({"colleges": [{"name": "AIIMS", "degree": "MBBS", "year": 2010}], "registrations": []})
        return MockExtractResult({"colleges": [{"name": "PGI", "degree": "MD", "year": 2015}], "registrations": []})

    async def mock_enrich_run(college_json: str, **kwargs):
        class MockEnrichResult:
            def __init__(self, data): self.output = data
        import json
        c = json.loads(college_json)
        return MockEnrichResult(CollegeInfo(name=c["name"], degree=c["degree"], year=c["year"], is_government=True, is_private=False))

    async def mock_consistency_run(prompt: str, **kwargs):
        class MockConsistencyResult:
            def __init__(self): self.output = ConsistencyResult(is_same_doctor=True, confidence_reasoning="Matches")
        return MockConsistencyResult()

    import main
    monkeypatch.setattr(main.search_agent, "run", mock_search_run)
    monkeypatch.setattr(main.extraction_agent, "run", mock_extract_run)
    monkeypatch.setattr(main.enrichment_agent, "run", mock_enrich_run)
    monkeypatch.setattr(main.consistency_agent, "run", mock_consistency_run)
    
    with client.websocket_connect("/ws/extract") as websocket:
        websocket.send_json({"name": "Abhishek Bansal", "hospital": "Lucknow Hosp"})
        
        while True:
            msg = websocket.receive_json()
            print("STAGE 1 msg:", msg)
            if msg["type"] == "search_result": break
        websocket.send_json({"action": "confirm"})
        
        final_data = None
        while True:
            msg = websocket.receive_json()
            print("STAGE 2 msg:", msg)
            if msg["type"] == "final_result":
                final_data = msg["data"]
                break
            if msg["type"] == "error":
                pytest.fail(f"Received error: {msg['message']}")
        assert len(final_data["colleges"]) == 2
        degrees = [c["degree"] for c in final_data["colleges"]]
        assert "MBBS" in degrees and "MD" in degrees
        assert search_calls[0] == 2 # Proves it didn't stop at URL 1

# Test 12: Stops immediately after finding 2 degrees
@pytest.mark.asyncio
async def test_aggregation_stops_at_two_degrees(monkeypatch):
    import httpx
    class MockClient:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc_val, exc_tb): pass
        async def get(self, url, timeout):
            class MockResponse:
                status_code = 200
                text = "Mock Content"
            return MockResponse()
    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    client = TestClient(app)
    
    search_calls = [0]
    async def mock_search_run(*args, **kwargs):
        search_calls[0] += 1
        class MockRunResult:
            def __init__(self): self.output = DoctorSearchResult(profile_url=f"http://x-{search_calls[0]}", source_platform="X", found_name="X", found_specialty="X", confidence_reasoning="")
        return MockRunResult()

    async def mock_extract_run(*args, **kwargs):
        class MockExtractResult:
            def __init__(self): self.output = DoctorExtractedData(colleges=[
                {"name": "Coll1", "degree": "MBBS", "year": 2000},
                {"name": "Coll2", "degree": "MD", "year": 2005}
            ], registrations=[])
        return MockExtractResult()

    async def mock_enrich_run(college_json, **kwargs):
        import json
        c = json.loads(college_json)
        class MockEnrichResult:
            def __init__(self): self.output = CollegeInfo(name=c["name"], degree=c["degree"], year=c.get("year"), is_government=True, is_private=False)
        return MockEnrichResult()

    import main
    monkeypatch.setattr(main.search_agent, "run", mock_search_run)
    monkeypatch.setattr(main.extraction_agent, "run", mock_extract_run)
    monkeypatch.setattr(main.enrichment_agent, "run", mock_enrich_run)
    
    with client.websocket_connect("/ws/extract") as websocket:
        websocket.send_json({"name": "Test", "hospital": "Test"})
        while True:
            if websocket.receive_json()["type"] == "search_result": break
        websocket.send_json({"action": "confirm"})
        while True:
            msg = websocket.receive_json()
            if msg["type"] == "final_result": break
            if msg["type"] == "error":
                pytest.fail(f"Received error: {msg['message']}")
        assert search_calls[0] == 1 # Shouldn't search a second URL because first URL had 2 degrees

# Test 13: Filters inconsistent doctors
@pytest.mark.asyncio
async def test_aggregation_filters_inconsistent_doctors(monkeypatch):
    import httpx
    class MockClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, *args, **kwargs):
            class MockResponse:
                status_code = 200
                text = "Mock Content"
            return MockResponse()
    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    client = TestClient(app)
    
    search_calls = [0]
    async def mock_search_run(*args, **kwargs):
        search_calls[0] += 1
        class MockRunResult:
            def __init__(self): self.output = DoctorSearchResult(profile_url=f"http://x-{search_calls[0]}", source_platform="X", found_name="X", found_specialty="X", confidence_reasoning="")
        return MockRunResult()

    extract_calls = [0]
    async def mock_extract_run(*args, **kwargs):
        class MockExtractResult:
            def __init__(self, data): self.output = DoctorExtractedData(**data)
        extract_calls[0] += 1
        if extract_calls[0] == 1:
            return MockExtractResult({"colleges": [{"name": "AIIMS", "degree": "MBBS"}], "registrations": []})
        # URL 2 has MD, but we'll mock consistency to reject it
        return MockExtractResult({"colleges": [{"name": "PGI", "degree": "MD"}], "registrations": []})

    async def mock_consistency_run(*args, **kwargs):
        class MockConsistencyResult:
            def __init__(self): self.output = ConsistencyResult(is_same_doctor=False, confidence_reasoning="Different city")
        return MockConsistencyResult()

    async def mock_enrich_run(*args, **kwargs):
        class MockEnrichResult:
            def __init__(self): self.output = CollegeInfo(name="AIIMS", degree="MBBS", year=2000, is_government=True, is_private=False)
        return MockEnrichResult()

    import main
    monkeypatch.setattr(main.search_agent, "run", mock_search_run)
    monkeypatch.setattr(main.extraction_agent, "run", mock_extract_run)
    monkeypatch.setattr(main.enrichment_agent, "run", mock_enrich_run)
    monkeypatch.setattr(main.consistency_agent, "run", mock_consistency_run)
    
    with client.websocket_connect("/ws/extract") as websocket:
        websocket.send_json({"name": "Test", "hospital": "Test"})
        while True:
            if websocket.receive_json()["type"] == "search_result": break
        websocket.send_json({"action": "confirm"})
        
        final_data = None
        while True:
            msg = websocket.receive_json()
            if msg["type"] == "final_result": 
                final_data = msg["data"]
                break
            if msg["type"] == "error":
                pytest.fail(f"Received error: {msg['message']}")
            
        assert len(final_data["colleges"]) == 1
        assert final_data["colleges"][0]["degree"] == "MBBS" # The MD was rejected due to inconsistency

# Test 14: Deduplicates degrees (keeps first)
@pytest.mark.asyncio
async def test_aggregation_deduplicates_degrees(monkeypatch):
    import httpx
    class MockClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, *args, **kwargs):
            class MockResponse:
                status_code = 200
                text = "Mock Content"
            return MockResponse()
    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    client = TestClient(app)
    
    search_calls = [0]
    async def mock_search_run(*args, **kwargs):
        search_calls[0] += 1
        class MockRunResult:
            def __init__(self): self.output = DoctorSearchResult(profile_url=f"http://x-{search_calls[0]}", source_platform="X", found_name="X", found_specialty="X", confidence_reasoning="")
        return MockRunResult()

    extract_calls = [0]
    async def mock_extract_run(*args, **kwargs):
        class MockExtractResult:
            def __init__(self, data): self.output = DoctorExtractedData(**data)
        extract_calls[0] += 1
        if extract_calls[0] == 1:
            return MockExtractResult({"colleges": [{"name": "AIIMS", "degree": "MBBS"}], "registrations": []})
        # URL 2 has another MBBS and an MD
        return MockExtractResult({"colleges": [{"name": "MAMC", "degree": "MBBS"}, {"name": "PGI", "degree": "MD"}], "registrations": []})

    async def mock_consistency_run(*args, **kwargs):
        class MockConsistencyResult:
            def __init__(self): self.output = ConsistencyResult(is_same_doctor=True, confidence_reasoning="")
        return MockConsistencyResult()

    async def mock_enrich_run(college_json, **kwargs):
        import json
        c = json.loads(college_json)
        class MockEnrichResult:
            def __init__(self): self.output = CollegeInfo(name=c["name"], degree=c["degree"], year=2000, is_government=True, is_private=False)
        return MockEnrichResult()

    import main
    monkeypatch.setattr(main.search_agent, "run", mock_search_run)
    monkeypatch.setattr(main.extraction_agent, "run", mock_extract_run)
    monkeypatch.setattr(main.enrichment_agent, "run", mock_enrich_run)
    monkeypatch.setattr(main.consistency_agent, "run", mock_consistency_run)
    
    with client.websocket_connect("/ws/extract") as websocket:
        websocket.send_json({"name": "Test", "hospital": "Test"})
        while True:
            if websocket.receive_json()["type"] == "search_result": break
        websocket.send_json({"action": "confirm"})
        final_data = None
        while True:
            msg = websocket.receive_json()
            if msg["type"] == "final_result": 
                final_data = msg["data"]
                break
            if msg["type"] == "error": break
        
        assert len(final_data["colleges"]) == 2
        degrees = [c["degree"] for c in final_data["colleges"]]
        names = [c["name"] for c in final_data["colleges"]]
        assert "MBBS" in degrees and "MD" in degrees
        assert "AIIMS" in names
        assert "MAMC" not in names # Kept the first MBBS

# Test 15: Ignores empty results silently
@pytest.mark.asyncio
async def test_aggregation_ignores_empty_results(monkeypatch):
    import httpx
    class MockClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, *args, **kwargs):
            class MockResponse:
                status_code = 200
                text = "Mock Content"
            return MockResponse()
    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    client = TestClient(app)
    
    search_calls = [0]
    async def mock_search_run(*args, **kwargs):
        search_calls[0] += 1
        class MockRunResult:
            def __init__(self): self.output = DoctorSearchResult(profile_url=f"http://x-{search_calls[0]}", source_platform="X", found_name="X", found_specialty="X", confidence_reasoning="")
        return MockRunResult()

    extract_calls = [0]
    async def mock_extract_run(*args, **kwargs):
        class MockExtractResult:
            def __init__(self, data): self.output = DoctorExtractedData(**data)
        extract_calls[0] += 1
        if extract_calls[0] == 1:
            return MockExtractResult({"colleges": [{"name": "AIIMS", "degree": "MBBS"}], "registrations": []})
        if extract_calls[0] == 2:
            return MockExtractResult({"colleges": [], "registrations": []}) # empty
        return MockExtractResult({"colleges": [{"name": "PGI", "degree": "MD"}], "registrations": []})

    async def mock_consistency_run(*args, **kwargs):
        class MockConsistencyResult:
            def __init__(self): self.output = ConsistencyResult(is_same_doctor=True, confidence_reasoning="")
        return MockConsistencyResult()

    async def mock_enrich_run(college_json, **kwargs):
        import json
        c = json.loads(college_json)
        class MockEnrichResult:
            def __init__(self): self.output = CollegeInfo(name=c["name"], degree=c["degree"], year=2000, is_government=True, is_private=False)
        return MockEnrichResult()

    import main
    monkeypatch.setattr(main.search_agent, "run", mock_search_run)
    monkeypatch.setattr(main.extraction_agent, "run", mock_extract_run)
    monkeypatch.setattr(main.enrichment_agent, "run", mock_enrich_run)
    monkeypatch.setattr(main.consistency_agent, "run", mock_consistency_run)
    
    with client.websocket_connect("/ws/extract") as websocket:
        websocket.send_json({"name": "Test", "hospital": "Test"})
        while True:
            if websocket.receive_json()["type"] == "search_result": break
        websocket.send_json({"action": "confirm"})
        final_data = None
        while True:
            msg = websocket.receive_json()
            if msg["type"] == "final_result": 
                final_data = msg["data"]
                break
            if msg["type"] == "error": break
        
        assert len(final_data["colleges"]) == 2
        assert extract_calls[0] == 3 # Proves it skipped 2 and moved to 3

# Test 16: Stops at URL limit 10
@pytest.mark.asyncio
async def test_aggregation_stops_at_URL_limit(monkeypatch):
    import httpx
    class MockClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, *args, **kwargs):
            class MockResponse:
                status_code = 200
                text = "Mock Content"
            return MockResponse()
    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    client = TestClient(app)
    
    search_calls = [0]
    async def mock_search_run(*args, **kwargs):
        search_calls[0] += 1
        class MockRunResult:
            def __init__(self): self.output = DoctorSearchResult(profile_url=f"http://x-{search_calls[0]}", source_platform="X", found_name="X", found_specialty="X", confidence_reasoning="")
        return MockRunResult()

    async def mock_extract_run(*args, **kwargs):
        class MockExtractResult:
            def __init__(self): self.output = DoctorExtractedData(colleges=[{"name": "AIIMS", "degree": "MBBS"}], registrations=[])
        return MockExtractResult() # Never returns an MD, so it loops until limit

    async def mock_consistency_run(*args, **kwargs):
        class MockConsistencyResult:
            def __init__(self): self.output = ConsistencyResult(is_same_doctor=True, confidence_reasoning="")
        return MockConsistencyResult()

    async def mock_enrich_run(college_json, **kwargs):
        class MockEnrichResult:
            def __init__(self): self.output = CollegeInfo(name="AIIMS", degree="MBBS", year=2000, is_government=True, is_private=False)
        return MockEnrichResult()

    import main
    monkeypatch.setattr(main.search_agent, "run", mock_search_run)
    monkeypatch.setattr(main.extraction_agent, "run", mock_extract_run)
    monkeypatch.setattr(main.enrichment_agent, "run", mock_enrich_run)
    monkeypatch.setattr(main.consistency_agent, "run", mock_consistency_run)
    
    with client.websocket_connect("/ws/extract") as websocket:
        websocket.send_json({"name": "Test", "hospital": "Test"})
        while True:
            if websocket.receive_json()["type"] == "search_result": break
        websocket.send_json({"action": "confirm"})
        final_data = None
        while True:
            msg = websocket.receive_json()
            if msg["type"] == "final_result": 
                final_data = msg["data"]
                break
            if msg["type"] == "error": break
        
        assert len(final_data["colleges"]) == 1 # Only got MBBS
        assert search_calls[0] >= 10 # Hit the hard limit

# Test 17: Handles extraction exception
@pytest.mark.asyncio
async def test_aggregation_handles_extraction_errors(monkeypatch):
    import httpx
    class MockClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, *args, **kwargs):
            class MockResponse:
                status_code = 200
                text = "Mock Content"
            return MockResponse()
    monkeypatch.setattr(httpx, "AsyncClient", MockClient)
    client = TestClient(app)
    
    search_calls = [0]
    async def mock_search_run(*args, **kwargs):
        search_calls[0] += 1
        class MockRunResult:
            def __init__(self): self.output = DoctorSearchResult(profile_url=f"http://x-{search_calls[0]}", source_platform="X", found_name="X", found_specialty="X", confidence_reasoning="")
        return MockRunResult()

    extract_calls = [0]
    async def mock_extract_run(*args, **kwargs):
        class MockExtractResult:
            def __init__(self, data): self.output = DoctorExtractedData(**data)
        extract_calls[0] += 1
        if extract_calls[0] == 1:
            return MockExtractResult({"colleges": [{"name": "AIIMS", "degree": "MBBS"}], "registrations": []})
        if extract_calls[0] == 2:
            raise Exception("Mock timeout/crash")
        return MockExtractResult({"colleges": [{"name": "PGI", "degree": "MD"}], "registrations": []})

    async def mock_consistency_run(*args, **kwargs):
        class MockConsistencyResult:
            def __init__(self): self.output = ConsistencyResult(is_same_doctor=True, confidence_reasoning="")
        return MockConsistencyResult()

    async def mock_enrich_run(college_json, **kwargs):
        import json
        c = json.loads(college_json)
        class MockEnrichResult:
            def __init__(self): self.output = CollegeInfo(name=c["name"], degree=c["degree"], year=2000, is_government=True, is_private=False)
        return MockEnrichResult()

    import main
    monkeypatch.setattr(main.search_agent, "run", mock_search_run)
    monkeypatch.setattr(main.extraction_agent, "run", mock_extract_run)
    monkeypatch.setattr(main.enrichment_agent, "run", mock_enrich_run)
    monkeypatch.setattr(main.consistency_agent, "run", mock_consistency_run)
    
    with client.websocket_connect("/ws/extract") as websocket:
        websocket.send_json({"name": "Test", "hospital": "Test"})
        while True:
            if websocket.receive_json()["type"] == "search_result": break
        websocket.send_json({"action": "confirm"})
        final_data = None
        while True:
            msg = websocket.receive_json()
            if msg["type"] == "final_result": 
                final_data = msg["data"]
                break
            if msg["type"] == "error": break
        
        assert len(final_data["colleges"]) == 2
        assert extract_calls[0] == 3
