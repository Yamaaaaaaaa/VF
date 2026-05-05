import React, { useState, useEffect, useRef } from 'react';
import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { CloudRain, Home, Database, Disc, Play, Pause } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

function Layout() {
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
  const [currentTrack, setCurrentTrack] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef(null);

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

    if (uploadedFileUrl) URL.revokeObjectURL(uploadedFileUrl);

    setUploadedFileName(file.name);
    setUploading(true);

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
      if (isPlaying) {
        audioRef.current.pause();
        setIsPlaying(false);
      } else {
        audioRef.current.play();
        setIsPlaying(true);
      }
    } else {
      setCurrentTrack(track);
      setIsPlaying(true);
      setTimeout(() => {
        if (audioRef.current) audioRef.current.play();
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

        <nav className="nav-menu">
          <div className="nav-menu-title">Menu</div>
          <NavLink
            to="/"
            end
            className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
          >
            <Home size={20} /> Semantic Search
          </NavLink>
          <NavLink
            to="/database"
            className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
          >
            <Database size={20} /> Voice Database
          </NavLink>
        </nav>

        {/* Global Player Widget */}
        <div
          className="global-player-widget"
          style={{ opacity: currentTrack ? 1 : 0.5, pointerEvents: currentTrack ? 'auto' : 'none' }}
        >
          <div className="player-info">
            <div className="player-avatar">
              <Disc
                size={24}
                style={{ animation: isPlaying ? 'spin 4s linear infinite' : 'none' }}
              />
            </div>
            <div className="player-details">
              <h4>{currentTrack ? currentTrack.title : 'No track'}</h4>
              <p>{currentTrack ? currentTrack.subtitle : 'Select audio to play'}</p>
            </div>
          </div>
          <div className="player-controls">
            <button className="play-btn-main" onClick={() => currentTrack && playTrack(currentTrack)}>
              {isPlaying
                ? <Pause fill="currentColor" size={20} />
                : <Play fill="currentColor" size={20} style={{ marginLeft: '4px' }} />}
            </button>
          </div>
        </div>
      </aside>

      {/* Main View — renders the matched route's page component */}
      <main className="main-content">
        <Outlet context={{
          // Search props
          uploading,
          uploadedFileName,
          uploadedFileUrl,
          searchResults,
          fileInputRef,
          onFileUpload: handleFileUpload,
          // Database props
          records,
          totalRecords,
          loading,
          searchQuery,
          filterGender,
          onSearchChange: setSearchQuery,
          onFilterChange: setFilterGender,
          fetchRecords,
          // Shared
          currentTrack,
          isPlaying,
          onPlayTrack: playTrack,
          formatAudioUrl,
        }} />
      </main>
    </div>
  );
}

export default Layout;
