import React from 'react';
import { useOutletContext } from 'react-router-dom';
import { UploadCloud, User, Play, Pause } from 'lucide-react';

function SearchPage() {
  const { 
    uploading, uploadedFileName, uploadedFileUrl, searchResults,
    fileInputRef, currentTrack, isPlaying,
    onFileUpload, onPlayTrack, formatAudioUrl
  } = useOutletContext();
  return (
    <div className="fade-in">
      <div className="hero-banner">
        <h1>Listen to the most similar voices in the database</h1>
        <p>Upload a voice sample to extract acoustic features (MFCC, Spectral Contrast, Chroma) and find exact biometric matches instantly.</p>
        
        <input 
          type="file" 
          ref={fileInputRef} 
          style={{ display: 'none' }} 
          accept=".wav"
          onChange={onFileUpload}
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
                onClick={() => onPlayTrack({ url: uploadedFileUrl, title: uploadedFileName, subtitle: 'Uploaded Sample' })}
              >
                {currentTrack?.url === uploadedFileUrl && isPlaying 
                  ? <Pause size={16} fill="currentColor" /> 
                  : <Play size={16} fill="currentColor" style={{ marginLeft: '2px' }} />}
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
              const audioUrl = formatAudioUrl(res.file_path);
              const isCurrentPlaying = currentTrack?.url === audioUrl;
              return (
                <div className="track-item fade-in" style={{ animationDelay: `${index * 0.1}s` }} key={res.file_id}>
                  <div className="track-index">{(index + 1).toString().padStart(2, '0')}</div>
                  <div className="track-img">
                    <User size={24} />
                  </div>
                  <div className="track-info">
                    <div className="track-title">{res.file_id}</div>
                    <div className="track-subtitle">
                      <span className="similarity-badge">{(res.similarity * 100).toFixed(2)}% Match</span>
                      <span>• {res.gender} • {res.accent || 'Unknown'} Accent</span>
                    </div>
                  </div>
                  <button 
                    className="play-btn-small"
                    onClick={() => onPlayTrack({
                      url: audioUrl,
                      title: `File #${res.file_id}`,
                      subtitle: `${(res.similarity * 100).toFixed(1)}% Match`
                    })}
                  >
                    {isCurrentPlaying && isPlaying 
                      ? <Pause size={16} fill="white" /> 
                      : <Play size={16} fill="white" style={{ marginLeft: '2px' }} />}
                  </button>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

export default SearchPage;
