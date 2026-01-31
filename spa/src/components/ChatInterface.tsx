import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, StopCircle, RefreshCw, Cpu, Globe } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const API_URL = 'http://localhost:8000';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState<'local' | 'openai'>('local');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const adjustHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg = input.trim();
    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setLoading(true);

    try {
      // Add simplified logic for optimistic UI or wait for stream
      // Create a placeholder for the assistant
      setMessages(prev => [...prev, { role: 'assistant', content: '' }]);
      
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMsg, model }),
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let currentResponse = '';

      while (!done) {
        const { value, done: DONE } = await reader.read();
        done = DONE;
        if (value) {
          const chunk = decoder.decode(value);
          currentResponse += chunk;
          
          setMessages(prev => {
            const newArr = [...prev];
            newArr[newArr.length - 1] = { role: 'assistant', content: currentResponse };
            return newArr;
          });
        }
      }
    } catch (error) {
      console.error(error);
      setMessages(prev => {
         const newArr = [...prev];
         newArr[newArr.length - 1] = { role: 'assistant', content: '**Error**: Connection failed. Please check the backend.' };
         return newArr;
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      {/* Header with Model Selector */}
      <div className="chat-header">
        <h3>AI Assistant</h3>
        <div className="model-selector">
          <button 
            className={`model-btn ${model === 'local' ? 'active' : ''}`}
            onClick={() => setModel('local')}
            title="Local LLM (Private & Free)"
          >
            <Cpu size={16} className="icon" />
            <span>Local</span>
          </button>
          <button 
            className={`model-btn ${model === 'openai' ? 'active' : ''}`}
            onClick={() => setModel('openai')}
            title="OpenAI GPT-3.5 (Requires Key, Smarter)"
          >
            <Globe size={16} className="icon" />
            <span>OpenAI</span>
          </button>
        </div>
      </div>

      <div className="messages-list">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon-wrapper">
              <Bot size={48} className="text-primary" />
            </div>
            <h2>RAG Assistant</h2>
            <p>Documents loaded. Ask me anything!</p>
          </div>
        )}
        
        {messages.map((msg, idx) => (
          <div key={idx} className={`message-row ${msg.role === 'assistant' ? 'assistant' : 'user'}`}>
            <div className="avatar">
              {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
            </div>
            
            <div className={`message-bubble`}>
               {msg.role === 'assistant' && msg.content === '' ? (
                 <div className="typing-indicator">
                   <span></span><span></span><span></span>
                 </div>
               ) : (
                 <ReactMarkdown>{msg.content}</ReactMarkdown>
               )}
            </div>
          </div>
        ))}
        {/* Helper to keep auto-scroll valid even if last message is incomplete */}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area glass-panel">
        <form onSubmit={handleSubmit} className="input-form">
          <textarea
            ref={textareaRef}
            className="chat-textarea"
            placeholder="Type your question..."
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              adjustHeight();
            }}
            onKeyDown={handleKeyDown}
            disabled={loading}
            rows={1}
          />
          <button 
            type="submit" 
            disabled={loading || !input.trim()}
            className={`send-btn ${loading ? 'loading' : ''}`}
          >
            {loading ? <RefreshCw size={20} className="spin" /> : <Send size={20} />}
          </button>
        </form>
      </div>
    </div>
  );
}
