import React, { useEffect, useState, useCallback } from 'react';
import { useOutletContext } from 'react-router-dom';
import axios from 'axios';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import AnalysisPanel from '../components/AnalysisPanel';

const API_BASE = 'http://localhost:8000/api';
const PAGE_SIZE = 8;

function DatabasePage() {
  const { currentTrack, isPlaying, onPlayTrack, formatAudioUrl } = useOutletContext();

  // Records state
  const [records, setRecords]       = useState([]);
  const [total, setTotal]           = useState(0);
  const [loading, setLoading]       = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterGender, setFilterGender] = useState('');
  const [currentPage, setCurrentPage] = useState(1);

  // Selected record for analysis
  const [selectedRecord, setSelectedRecord] = useState(null);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  const fetchRecords = useCallback(async (page, query, gender) => {
    setLoading(true);
    try {
      const offset = (page - 1) * PAGE_SIZE;
      const params = new URLSearchParams({
        limit: PAGE_SIZE, offset, search: query, gender
      });
      const res = await axios.get(`${API_BASE}/records?${params}`);
      setRecords(res.data.records);
      setTotal(res.data.total);
    } catch (e) { console.error(e); }
    setLoading(false);
  }, []);

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1);
    setSelectedRecord(null);
    fetchRecords(1, searchQuery, filterGender);
  }, [searchQuery, filterGender]);

  // Fetch when page changes
  useEffect(() => {
    fetchRecords(currentPage, searchQuery, filterGender);
  }, [currentPage]);

  const handleCardClick = (record) => {
    // Toggle: click same card → deselect
    if (selectedRecord?.file_id === record.file_id) {
      setSelectedRecord(null);
      return;
    }
    setSelectedRecord(record);
    onPlayTrack({
      url: formatAudioUrl(record.file_path),
      title: `File #${record.file_id}`,
      subtitle: `${record.gender} Voice`,
    });
  };

  return (
    <div className="fade-in">
      {/* ── Header ── */}
      <div className="section-header">
        <h1 className="section-title" style={{ fontSize: '2rem' }}>Voice Database</h1>
        <span className="section-link">{total} Records</span>
      </div>

      {/* ── Filters ── */}
      <div className="filters-bar fade-in">
        <input
          type="text"
          className="filter-input"
          placeholder="Search by Speaker ID..."
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
        />
        <select
          className="filter-input"
          value={filterGender}
          onChange={e => setFilterGender(e.target.value)}
          style={{ flex: '0 0 180px' }}
        >
          <option value="">All Genders</option>
          <option value="female">Female</option>
          <option value="male">Male</option>
        </select>
      </div>

      {/* ── Record Grid: 4 columns × 2 rows ── */}
      {loading ? (
        <div className="loader orange" />
      ) : (
        <div className="record-grid">
          {records.map(record => {
            const audioUrl = formatAudioUrl(record.file_path);
            const isPlaying_ = currentTrack?.url === audioUrl && isPlaying;
            const isSelected = selectedRecord?.file_id === record.file_id;
            return (
              <div
                key={record.file_id}
                className={`record-card${isSelected ? ' selected' : ''}`}
                onClick={() => handleCardClick(record)}
                title="Click to view analysis"
              >
                <div className="record-card-avatar">#{record.file_id}</div>
                <div className="record-card-info">
                  <div className="record-card-id">File #{record.file_id}</div>
                  <div className="record-card-meta">
                    {record.gender} · {record.age ? `${record.age} yrs` : 'N/A'}
                  </div>
                  {record.accent && (
                    <div className="record-card-accent">{record.accent}</div>
                  )}
                  {isPlaying_ && (
                    <div className="record-card-playing">▶ Playing</div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Pagination ── */}
      {!loading && totalPages > 1 && (
        <div className="pagination">
          <button
            className="page-btn"
            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
            disabled={currentPage === 1}
          >
            <ChevronLeft size={18} />
          </button>
          <div className="page-info">
            Page <strong>{currentPage}</strong> / {totalPages}
            <span className="page-count"> · {total} records</span>
          </div>
          <button
            className="page-btn"
            onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
          >
            <ChevronRight size={18} />
          </button>
        </div>
      )}

      {/* ── Inline Analysis Panel (only when a record is selected) ── */}
      {selectedRecord && (
        <AnalysisPanel key={selectedRecord.file_id} record={selectedRecord} />
      )}
    </div>
  );
}

export default DatabasePage;
