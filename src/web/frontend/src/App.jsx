import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Search, Play, Pause, Database, UploadCloud, CloudRain, Disc, Home, Music, User } from 'lucide-react';
import './index.css';

const API_BASE = 'http://localhost:8000/api';

function App() {
  const [activeTab, setActiveTab] = useState('search'); // 'search' | 'database'
  
  // Database State
  const [records, setRecords] = useState([]);
  const [totalRecords, setTotalRecords] = useState(0);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterGender, setFilterGender] = useState('');
  
  // Search State
  const [uploading, setUploading] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [uploadedFileUrl, setUploadedFileUrl] = useState(null);
  const fileInputRef = useRef(null);

  // Global Audio Player State
  const [currentTrack, setCurrentTrack] = useState(null); // { url, title, subtitle }
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef(null);

  useEffect(() => {
    if (activeTab === 'database') {
      fetchRecords();
    }
  }, [activeTab, filterGender, searchQuery]);

  const fetchRecords = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ limit: 20, offset: 0, search: searchQuery, gender: filterGender });
      const response = await axios.get(`${API_BASE}/records?${params}`);
      setRecords(response.data.records);
      setTotalRecords(response.data.total);
    } catch (error) {
      console.error(error);
    }
    setLoading(false);
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (uploadedFileUrl) {
      URL.revokeObjectURL(uploadedFileUrl);
    }

    setUploadedFileName(file.name);
    setUploading(true);
    
    // Auto play uploaded file
    const objectUrl = URL.createObjectURL(file);
    setUploadedFileUrl(objectUrl);
    playTrack({ url: objectUrl, title: file.name, subtitle: 'Uploaded Sample' });

    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await axios.post(`${API_BASE}/search`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setSearchResults(response.data);
    } catch (error) {
      console.error(error);
      alert('Error searching voice database');
    }
    setUploading(false);
  };

  const playTrack = (track) => {
    if (currentTrack?.url === track.url) {
      // Toggle play/pause if same track
      if (isPlaying) {
        audioRef.current.pause();
        setIsPlaying(false);
      } else {
        audioRef.current.play();
        setIsPlaying(true);
      }
    } else {
      // Play new track
      setCurrentTrack(track);
      setIsPlaying(true);
      setTimeout(() => {
        if (audioRef.current) {
          audioRef.current.play();
        }
      }, 50);
    }
  };

  const formatAudioUrl = (filePath) => {
    return `http://localhost:8000/data/${filePath.replace(/\\/g, '/')}`;
  };

  return (
    <div className="app-container">
      {/* Hidden Global Audio Element */}
      <audio 
        ref={audioRef} 
        src={currentTrack?.url} 
        className="hidden-audio"
        onEnded={() => setIsPlaying(false)}
        onPause={() => setIsPlaying(false)}
        onPlay={() => setIsPlaying(true)}
      />

      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="logo-section">
          <div className="logo-icon">
            <CloudRain size={24} strokeWidth={2.5} />
          </div>
          <div className="logo-text">VoiceCloud</div>
        </div>

        <div className="profile-section">
          <div className="profile-greeting">Hi,</div>
          <div className="profile-name">Researcher</div>
        </div>

        <div className="nav-menu">
          <div className="nav-menu-title">Menu</div>
          <div 
            className={`nav-item ${activeTab === 'search' ? 'active' : ''}`}
            onClick={() => setActiveTab('search')}
          >
            <Home size={20} /> Semantic Search
          </div>
          <div 
            className={`nav-item ${activeTab === 'database' ? 'active' : ''}`}
            onClick={() => setActiveTab('database')}
          >
            <Database size={20} /> Voice Database
          </div>
        </div>

        {/* Global Player Widget placed at bottom left */}
        <div className="global-player-widget" style={{ opacity: currentTrack ? 1 : 0.5, pointerEvents: currentTrack ? 'auto' : 'none' }}>
          <div className="player-info">
            <div className="player-avatar">
              <Disc size={24} className={isPlaying ? 'spin-anim' : ''} style={{ animation: isPlaying ? 'spin 4s linear infinite' : 'none' }} />
            </div>
            <div className="player-details">
              <h4>{currentTrack ? currentTrack.title : 'No track'}</h4>
              <p>{currentTrack ? currentTrack.subtitle : 'Select audio to play'}</p>
            </div>
          </div>
          <div className="player-controls">
            <button className="play-btn-main" onClick={() => currentTrack && playTrack(currentTrack)}>
              {isPlaying ? <Pause fill="currentColor" size={20} /> : <Play fill="currentColor" size={20} style={{ marginLeft: '4px' }}/>}
            </button>
          </div>
        </div>
      </aside>

      {/* Main View Area */}
      <main className="main-content">
        {activeTab === 'search' ? (
          <div className="fade-in">
            <div className="hero-banner">
              <h1>Listen to the most similar voices in the database</h1>
              <p>Upload a voice sample to extract acoustic features (MFCC, Spectral Contrast, Chroma) and find exact biometric matches instantly.</p>
              
              <input 
                type="file" 
                ref={fileInputRef} 
                style={{ display: 'none' }} 
                accept=".wav"
                onChange={handleFileUpload}
              />
              <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
                <button 
                  className="btn-primary" 
                  onClick={() => fileInputRef.current.click()}
                  disabled={uploading}
                >
                  {uploading ? <div className="loader"></div> : <UploadCloud size={20} />}
                  {uploading ? 'Processing Audio...' : (uploadedFileName ? 'Upload Another' : 'Upload Sample')}
                </button>
                
                {uploadedFileName && !uploading && uploadedFileUrl && (
                  <div className="fade-in" style={{ 
                    display: 'flex', alignItems: 'center', gap: '1rem', 
                    background: 'rgba(255,255,255,0.15)', padding: '0.5rem 1rem', 
                    borderRadius: '99px', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.3)'
                  }}>
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                      <span style={{ fontSize: '0.75rem', opacity: 0.9 }}>Uploaded Source</span>
                      <span style={{ fontSize: '0.875rem', fontWeight: 600 }}>{uploadedFileName}</span>
                    </div>
                    <button 
                      className="play-btn-small" 
                      style={{ background: 'white', color: 'var(--primary-orange)' }}
                      onClick={() => playTrack({ url: uploadedFileUrl, title: uploadedFileName, subtitle: 'Uploaded Sample' })}
                    >
                      {currentTrack?.url === uploadedFileUrl && isPlaying ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" style={{ marginLeft: '2px' }} />}
                    </button>
                  </div>
                )}
              </div>
            </div>

            {searchResults.length > 0 && (
              <>
                <div className="section-header">
                  <h2 className="section-title">Top Trending Matches</h2>
                  <span className="section-link">Explore more</span>
                </div>
                
                <div className="trending-list">
                  {searchResults.map((res, index) => {
                    const isCurrentPlaying = currentTrack?.url === formatAudioUrl(res.file_path);
                    return (
                      <div className="track-item fade-in" style={{ animationDelay: `${index * 0.1}s` }} key={res.file_id}>
                        <div className="track-index">{(index + 1).toString().padStart(2, '0')}</div>
                        <div className="track-img">
                          <User size={24} />
                        </div>
                        <div className="track-info">
                          <div className="track-title">{res.speaker}</div>
                          <div className="track-subtitle">
                            <span className="similarity-badge">{(res.similarity * 100).toFixed(2)}% Match</span>
                            <span>• {res.gender} • {res.accent || 'Unknown'} Accent</span>
                          </div>
                        </div>
                        <button 
                          className="play-btn-small"
                          onClick={() => playTrack({
                            url: formatAudioUrl(res.file_path),
                            title: res.speaker,
                            subtitle: `${(res.similarity * 100).toFixed(1)}% Match`
                          })}
                        >
                          {isCurrentPlaying && isPlaying ? <Pause size={16} fill="white" /> : <Play size={16} fill="white" style={{ marginLeft: '2px' }} />}
                        </button>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        ) : (
          <div className="fade-in">
            <div className="section-header">
              <h1 className="section-title" style={{ fontSize: '2rem' }}>Voice Database</h1>
              <span className="section-link">{totalRecords} Records</span>
            </div>

            <div className="filters-bar fade-in">
              <input 
                type="text" 
                className="filter-input" 
                placeholder="Search by Speaker ID..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              <select 
                className="filter-input" 
                value={filterGender} 
                onChange={(e) => setFilterGender(e.target.value)}
                style={{ flex: '0 0 200px' }}
              >
                <option value="">All Genders</option>
                <option value="female">Female</option>
                <option value="male">Male</option>
              </select>
            </div>
            
            {loading ? (
              <div className="loader orange"></div>
            ) : (
              <div className="artist-grid">
                {records.map((record) => {
                  const isCurrentPlaying = currentTrack?.url === formatAudioUrl(record.file_path);
                  return (
                    <div 
                      className="artist-card fade-in" 
                      key={record.file_id}
                      onClick={() => playTrack({
                        url: formatAudioUrl(record.file_path),
                        title: record.speaker,
                        subtitle: `${record.gender} Voice`
                      })}
                    >
                      <div className="artist-avatar">
                        {record.speaker.substring(0, 2).toUpperCase()}
                      </div>
                      <div className="artist-info">
                        <h4>{record.speaker}</h4>
                        <p>{record.gender} • {record.age ? `${record.age} yrs` : 'N/A'}</p>
                        {isCurrentPlaying && isPlaying && (
                          <p style={{ color: 'var(--primary-orange)', fontWeight: 'bold', marginTop: '4px' }}>Playing...</p>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
