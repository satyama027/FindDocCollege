# FindDocCollege

A web app that automatically searches the web to find where a doctor went to medical college and their medical council registration details. 

It uses AI agents (Gemini + Tavily search) to scrape trusted platforms like Practo, Lybrate, Apollo, and the National Medical Register to extract educational background information.

## How it works

1. Enter a doctor's name and hospital.
2. The Search Agent hunts down their official profile across trusted medical domains.
3. The Extraction Agent scrapes the profile to pull out their medical degrees, colleges, and registration numbers.
4. The Enrichment Agent searches the web to verify if the scraped colleges are government or private institutions.

## Tech Stack

*   **Backend:** Python, FastAPI, PydanticAI
*   **Frontend:** React, Vite
*   **AI/Web:** Gemini 2.5 Flash, Tavily, Jina Reader

## Local Dev Setup

You'll need `GEMINI_API_KEY` and `TAVILY_API_KEY` in a `.env` file in the `/backend` folder.

1.  **Backend:**
    ```bash
    cd backend
    pip install -r requirements.txt
    uvicorn main:app --reload
    ```
2.  **Frontend:**
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

## Deployment

The app is set up for monolithic deployment. Running `npm run build` in the frontend outputs static files to `dist`, which the FastAPI backend is configured to serve automatically.
