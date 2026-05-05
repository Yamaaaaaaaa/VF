import React, { useEffect } from 'react';
import { useOutletContext } from 'react-router-dom';
import { Play, Pause } from 'lucide-react';

function DatabasePage() {
  const {
    records, totalRecords, loading,
    searchQuery, filterGender,
    currentTrack, isPlaying,
    onSearchChange, onFilterChange,
    fetchRecords, onPlayTrack, formatAudioUrl
  } = useOutletContext();

  useEffect(() => {
    fetchRecords();
  }, [searchQuery, filterGender]);
  return (
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
          onChange={(e) => onSearchChange(e.target.value)}
        />
        <select 
          className="filter-input" 
          value={filterGender} 
          onChange={(e) => onFilterChange(e.target.value)}
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
            const audioUrl = formatAudioUrl(record.file_path);
            const isCurrentPlaying = currentTrack?.url === audioUrl;
            return (
              <div 
                className="artist-card fade-in" 
                key={record.file_id}
                onClick={() => onPlayTrack({
                  url: audioUrl,
                  title: `File #${record.file_id}`,
                  subtitle: `${record.gender} Voice`
                })}
              >
                <div className="artist-avatar">
                  #{record.file_id}
                </div>
                <div className="artist-info">
                  <h4>File #{record.file_id}</h4>
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
  );
}

export default DatabasePage;
