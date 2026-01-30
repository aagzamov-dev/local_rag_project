import React, { useState } from 'react';
import axios from 'axios';
import { Upload, FileText, CheckCircle, AlertCircle } from 'lucide-react';

const API_URL = 'http://localhost:8000';

export default function FileUploader() {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState<{type: 'success' | 'error', msg: string} | null>(null);
  const [activeTab, setActiveTab] = useState<'upload' | 'json'>('upload');

  const [fileName, setFileName] = useState('');
  const [jsonContent, setJsonContent] = useState('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
    }
  };

  const handleUpload = async () => {
    if (!files.length) return;
    setUploading(true);
    setStatus(null);

    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    try {
      await axios.post(`${API_URL}/upload`, formData);
      setStatus({ type: 'success', msg: 'Files uploaded! Run ingestion to index.' });
      setFiles([]);
    } catch (err) {
      setStatus({ type: 'error', msg: 'Upload failed.' });
    } finally {
      setUploading(false);
    }
  };

  const handleCreateJson = async () => {
    if (!fileName || !jsonContent) return;
    setUploading(true);
    setStatus(null);

    try {
      let parsedData;
      try {
        parsedData = JSON.parse(jsonContent);
      } catch (e) {
        setStatus({ type: 'error', msg: 'Invalid JSON format.' });
        return;
      }

      await axios.post(`${API_URL}/create-json`, {
        filename: fileName,
        data: parsedData
      });
      setStatus({ type: 'success', msg: 'JSON created! Run ingestion to index.' });
      setFileName('');
      setJsonContent('');
    } catch (err) {
      setStatus({ type: 'error', msg: 'Creation failed.' });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="uploader-container">
      <div className="uploader-card">
        <h2 style={{ textAlign: 'center', marginBottom: '30px', fontSize: '1.8rem', background: '-webkit-linear-gradient(45deg, #818cf8, #c084fc)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Data Management
        </h2>

        <div className="tab-switcher">
          <div 
            className={`tab ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            File Upload
          </div>
          <div 
            className={`tab ${activeTab === 'json' ? 'active' : ''}`}
            onClick={() => setActiveTab('json')}
          >
            Create JSON
          </div>
        </div>

        {activeTab === 'upload' ? (
          <div>
            <div className="dropzone">
              <input 
                type="file" 
                multiple 
                onChange={handleFileChange} 
                className="dropzone-input" 
              />
              <Upload size={48} color="#818cf8" style={{ marginBottom: 15 }} />
              <p className="main-text">Drag files here or click to upload</p>
              <p>PDF, TXT, MD, JSON</p>
            </div>

            {files.length > 0 && (
              <div className="file-list">
                {files.map((f, i) => (
                  <div key={i} className="file-item">
                    <FileText size={16} /> {f.name}
                  </div>
                ))}
              </div>
            )}

            <button 
              onClick={handleUpload}
              disabled={uploading || files.length === 0}
              className="action-btn"
            >
              {uploading ? 'Uploading...' : 'Upload Files'}
            </button>
          </div>
        ) : (
          <div>
            <div className="field-group">
              <label className="field-label">Filename</label>
              <input 
                placeholder="my_data"
                value={fileName}
                onChange={(e) => setFileName(e.target.value)}
                className="data-input"
              />
            </div>
            <div className="field-group">
              <label className="field-label">JSON Content</label>
              <textarea 
                rows={8}
                placeholder='{ "key": "value" }'
                value={jsonContent}
                onChange={(e) => setJsonContent(e.target.value)}
                className="data-input"
                style={{ fontFamily: 'monospace' }}
              />
            </div>
            <button 
              onClick={handleCreateJson}
              disabled={uploading}
              className="action-btn"
            >
              {uploading ? 'Creating...' : 'Create JSON File'}
            </button>
          </div>
        )}

        {status && (
          <div className={`status-msg ${status.type}`}>
            {status.type === 'success' ? <CheckCircle size={20} /> : <AlertCircle size={20} />}
            {status.msg}
          </div>
        )}
      </div>
    </div>
  );
}
