import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, StopCircle, RefreshCw, Cpu, Globe } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const API_URL = 'http://localhost:8000';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  model?: 'local' | 'openai';
}

const TOOL_TRACE_LABEL = 'Tool Trace:';
const SOURCES_LABEL = 'Sources:';

function splitAssistantContent(content: string) {
  let main = content || '';
  let toolTrace = '';
  let sources = '';

  const toolIndex = main.indexOf(TOOL_TRACE_LABEL);
  if (toolIndex !== -1) {
    toolTrace = main.slice(toolIndex + TOOL_TRACE_LABEL.length).trim();
    main = main.slice(0, toolIndex).trim();
  }

  const sourcesIndex = main.indexOf(SOURCES_LABEL);
  if (sourcesIndex !== -1) {
    sources = main.slice(sourcesIndex + SOURCES_LABEL.length).trim();
    main = main.slice(0, sourcesIndex).trim();
  }

  return { main, sources, toolTrace };
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
    // Capture current model at start of request
    const currentModel = model;
    
    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setLoading(true);

    try {
      // Create placeholder with correct model
      setMessages(prev => [...prev, { role: 'assistant', content: '', model: currentModel }]);
      
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMsg, model: currentModel }),
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
            // Ensure we preserve the model field
            newArr[newArr.length - 1] = { 
                role: 'assistant', 
                content: currentResponse,
                model: currentModel
            };
            return newArr;
          });
        }
      }
    } catch (error) {
      console.error(error);
      setMessages(prev => {
         const newArr = [...prev];
         newArr[newArr.length - 1] = { 
             role: 'assistant', 
             content: '**Error**: Connection failed. Please check the backend.',
             model: currentModel
         };
         return newArr;
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      {/* ... (Header unchanged) ... */}
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
            title="OpenAI GPT-4o (Requires Key, Smarter)"
          >
            <Globe size={16} className="icon" />
            <span>OpenAI GPT-4o</span>
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
                 <>
                   {msg.role === 'assistant' && (
                       <div className="model-chip">
                         {(msg.model || 'local').toUpperCase()} USED
                       </div>
                   )}
                   {(() => {
                     if (msg.role !== 'assistant') {
                       return <ReactMarkdown>{msg.content}</ReactMarkdown>;
                     }
                     const { main, sources, toolTrace } = splitAssistantContent(msg.content);
                     return (
                       <>
                         <ReactMarkdown>{main}</ReactMarkdown>
                         {sources && (
                           <div className="meta-block sources-block">
                             <div className="meta-label">Sources</div>
                             <div className="meta-text">{sources}</div>
                           </div>
                         )}
                         {toolTrace && (
                           <div className="meta-block tool-block">
                             <div className="meta-label">Tool Trace</div>
                             <ReactMarkdown>{toolTrace}</ReactMarkdown>
                           </div>
                         )}
                       </>
                     );
                   })()}
                 </>
               )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area glass-panel">
         {/* ... (Input form unchanged) ... */}
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
