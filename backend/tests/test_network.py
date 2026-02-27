import pytest
import asyncio
import websockets
import json
import threading
import uvicorn
import time
from urllib.error import URLError
import urllib.request

from main import app

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="critical")

@pytest.fixture(scope="module", autouse=True)
def live_server():
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    # Wait for the server to actually start accepting connections
    max_retries = 20
    for _ in range(max_retries):
        try:
            urllib.request.urlopen("http://127.0.0.1:8001/docs")
            break
        except URLError:
            time.sleep(0.5)
    else:
        pytest.fail("Server did not start in time")
        
    yield

@pytest.mark.asyncio
async def test_websocket_network():
    """Verify that a real network WebSocket connection establishes properly and transfers JSON payloads."""
    uri = "ws://127.0.0.1:8001/ws/extract"
    try:
        async with websockets.connect(uri) as websocket:
            # Send an invalid payload that should return an error immediately
            await websocket.send(json.dumps({"name": "", "hospital": ""}))
            
            response = await websocket.recv()
            data = json.loads(response)
            
            assert data["type"] == "error"
            assert "Name and Hospital are required" in data["message"]
    except Exception as e:
        pytest.fail(f"Network connection failed: {e}")