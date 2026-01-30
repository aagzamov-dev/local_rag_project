import { useState } from 'react';
import './App.css';
import { MessageSquare, Database, Settings, HelpCircle } from 'lucide-react';
import ChatInterface from './components/ChatInterface';
import FileUploader from './components/FileUploader';

function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'data'>('chat');

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <nav className="sidebar">
        <div className="sidebar-logo">
          R
        </div>
        
        <div 
          className={`nav-item ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
          title="Chat"
        >
          <MessageSquare size={24} />
        </div>
        
        <div 
          className={`nav-item ${activeTab === 'data' ? 'active' : ''}`}
          onClick={() => setActiveTab('data')}
          title="Data Management"
        >
          <Database size={24} />
        </div>

        <div className="sidebar-footer">
           <div className="nav-item" title="Settings">
            <Settings size={24} />
          </div>
           <div className="nav-item" title="Help">
            <HelpCircle size={24} />
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="main-content">
        <div className="bg-effects">
          <div className="effect-blob blob-1"></div>
          <div className="effect-blob blob-2"></div>
        </div>

        <div className="content-area">
          {activeTab === 'chat' ? <ChatInterface /> : <FileUploader />}
        </div>
      </main>
    </div>
  );
}

export default App;
