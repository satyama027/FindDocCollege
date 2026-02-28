import { useState, useRef, useEffect } from 'react';

function App() {
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');

  const [flowState, setFlowState] = useState('IDLE'); // IDLE, SEARCHING, CONFIRM, EXTRACTING, DONE

  const [searchForm, setSearchForm] = useState({ name: '', hospital: '', additional_context: '' });
  const [searchResult, setSearchResult] = useState(null);

  const [finalProfile, setFinalProfile] = useState(null);
  const ws = useRef(null);

  const connectWebSocket = () => {
    if (ws.current) ws.current.close();

    // Connect to FastAPI backend dynamically based on environment
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = import.meta.env.DEV ? '127.0.0.1:8000' : window.location.host;
    ws.current = new WebSocket(`${wsProtocol}//${wsHost}/ws/extract`);

    ws.current.onopen = () => {
      // Send initial payload
      ws.current.send(JSON.stringify({
        name: searchForm.name,
        hospital: searchForm.hospital,
        additional_context: searchForm.additional_context
      }));
    };

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'status':
          setStatus(data.message);
          break;
        case 'search_failed':
          setFlowState('CONFIRM'); // Still need to prompt for refine
          setSearchResult({ failed: true, reasoning: data.reasoning });
          setStatus('');
          break;
        case 'search_result':
          setFlowState('CONFIRM');
          setSearchResult(data.data);
          setStatus('');
          break;
        case 'final_result':
          setFlowState('DONE');
          setFinalProfile(data.data);
          setStatus('');
          ws.current.close();
          break;
        case 'error':
          setError(data.message);
          setStatus('');
          setFlowState('IDLE');
          ws.current.close();
          break;
        default:
          break;
      }
    };

    ws.current.onerror = (e) => {
      setError("WebSocket connection failed. Make sure the FastAPI backend is running on port 8000.");
      setFlowState('IDLE');
    };

    ws.current.onclose = () => {
      if (flowState !== 'DONE' && flowState !== 'IDLE') {
        // Backend closed unexpectedly without final result or error
      }
    };
  };

  const handleStartSearch = (e) => {
    e.preventDefault();
    if (!searchForm.name || !searchForm.hospital) {
      setError("Name and hospital are required");
      return;
    }
    setError('');
    setSearchResult(null);
    setFinalProfile(null);
    setFlowState('SEARCHING');
    connectWebSocket();
  };

  const handleConfirm = () => {
    setFlowState('EXTRACTING');
    ws.current.send(JSON.stringify({ action: 'confirm' }));
  };

  const handleRefine = (e) => {
    e.preventDefault();
    setFlowState('SEARCHING');
    ws.current.send(JSON.stringify({
      action: 'refine',
      additional_context: searchForm.additional_context
    }));
  };

  const resetFlow = () => {
    setFlowState('IDLE');
    setSearchResult(null);
    setFinalProfile(null);
    setError('');
    setStatus('');
    setSearchForm({ name: '', hospital: '', additional_context: '' });
    if (ws.current) ws.current.close();
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>FindDocCollege</h1>
        <p>AI-Powered Doctor Extraction Agent</p>
      </div>

      {error && (
        <div className="error-box">
          <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {error}
        </div>
      )}

      {flowState === 'IDLE' && (
        <div className="card">
          <h2 className="card-title">Search Doctor Profile</h2>
          <form onSubmit={handleStartSearch}>
            <div className="form-group">
              <label>Doctor's Name</label>
              <input
                className="form-input"
                type="text"
                placeholder="e.g. Sushanth Shivaswamy"
                value={searchForm.name}
                onChange={e => setSearchForm({ ...searchForm, name: e.target.value })}
              />
            </div>
            <div className="form-group">
              <label>Hospital or Location</label>
              <input
                className="form-input"
                type="text"
                placeholder="e.g. Kauveri Hospital Bangalore"
                value={searchForm.hospital}
                onChange={e => setSearchForm({ ...searchForm, hospital: e.target.value })}
              />
            </div>
            <button type="submit" className="btn btn-primary" disabled={!searchForm.name || !searchForm.hospital}>
              Search Web
            </button>
          </form>
        </div>
      )}

      {(flowState === 'SEARCHING' || flowState === 'EXTRACTING') && (
        <div className="status-box">
          <div className="spinner"></div>
          <div>{status || 'Processing request...'}</div>
        </div>
      )}

      {flowState === 'CONFIRM' && searchResult && (
        <div className="card">
          <h2 className="card-title">
            {searchResult.failed ? "Doctor Not Found" : "Profile Found"}
          </h2>

          <div className="result-item">
            <div className="result-label">Agent Reasoning</div>
            <div className="reasoning">{searchResult.reasoning || searchResult.confidence_reasoning}</div>
          </div>

          {!searchResult.failed && (
            <>
              <div className="result-item">
                <div className="result-label">Doctor Name</div>
                <div className="result-value">{searchResult.found_name}</div>
              </div>
              <div className="result-item">
                <div className="result-label">Specialty</div>
                <div className="result-value">{searchResult.found_specialty}</div>
              </div>
              <div className="result-item">
                <div className="result-label">Profile URL</div>
                <div className="result-value">
                  <a href={searchResult.profile_url} target="_blank" rel="noreferrer">
                    {searchResult.profile_url}
                  </a>
                  {searchResult.source_platform && (
                    <span className="badge badge-private" style={{ marginLeft: '10px' }}>
                      Source: {searchResult.source_platform}
                    </span>
                  )}
                </div>
              </div>
            </>
          )}

          <div className="action-buttons">
            {!searchResult.failed && (
              <button className="btn btn-primary" onClick={handleConfirm} style={{ marginBottom: '1rem' }}>
                Yes, Extract Data
              </button>
            )}

            <form onSubmit={handleRefine}>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label>Not the right person?</label>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <input
                    className="form-input"
                    type="text"
                    placeholder="Provide more context (e.g. City)"
                    value={searchForm.additional_context}
                    onChange={e => setSearchForm({ ...searchForm, additional_context: e.target.value })}
                  />
                  <button type="submit" className="btn btn-secondary" style={{ width: 'auto', marginTop: 0 }}>
                    Refine
                  </button>
                </div>
              </div>
            </form>
          </div>
        </div>
      )}

      {flowState === 'DONE' && finalProfile && (
        <div className="card" style={{ borderColor: '#10B981' }}>
          <h2 className="card-title" style={{ color: '#047857' }}>Extraction Complete</h2>

          <div style={{ marginBottom: '1.5rem' }}>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.25rem' }}>{finalProfile.name}</h3>
            <p style={{ color: '#64748B' }}>{finalProfile.hospital}</p>
          </div>

          <div className="result-item">
            <div className="result-label">Educational Background</div>
            {finalProfile.colleges.length === 0 ? (
              <div className="result-value" style={{ color: '#94A3B8', fontStyle: 'italic' }}>No colleges found.</div>
            ) : (
              <ul className="college-list">
                {finalProfile.colleges.map((col, idx) => (
                  <li key={idx} className="college-item">
                    <div className="result-value">{col.name}</div>
                    <span className={`badge ${col.is_government ? 'badge-gov' : 'badge-private'}`}>
                      {col.is_government ? 'Government Institution' : 'Private Institution'}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="result-item" style={{ marginTop: '1.5rem' }}>
            <div className="result-label">Medical Registrations</div>
            {finalProfile.registrations.length === 0 ? (
              <div className="result-value" style={{ color: '#94A3B8', fontStyle: 'italic' }}>No registrations found.</div>
            ) : (
              <ul className="registration-list">
                {finalProfile.registrations.map((reg, idx) => (
                  <li key={idx} className="registration-item">
                    <div className="result-value">• {reg}</div>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <button className="btn btn-secondary" onClick={resetFlow} style={{ marginTop: '2rem' }}>
            Start New Search
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
