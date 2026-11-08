import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import ChatInput from './components/ChatInput';
import SourceViewer from './components/SourceViewer';
import { useTheme } from './hooks/useTheme';
import {
  checkBackendHealth,
  getCorpusInfo,
  submitRAGQuery,
} from './services/api';

const STORAGE_KEY_SESSIONS = 'hindi_rag_chat_sessions_v1';
const STORAGE_KEY_ACTIVE_ID = 'hindi_rag_active_session_id';
const STORAGE_KEY_COLLAPSED = 'hindi_rag_sidebar_collapsed';

export default function App() {
  const { theme, toggleTheme } = useTheme();

  // Sessions State
  const [sessions, setSessions] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY_SESSIONS);
      if (saved) return JSON.parse(saved);
    } catch (e) {
      console.warn('Could not load chat sessions from localStorage:', e);
    }
    return [
      {
        id: 'session-default',
        title: 'New Conversation',
        createdAt: new Date().toISOString(),
        messages: [],
      },
    ];
  });

  const [activeSessionId, setActiveSessionId] = useState(() => {
    return localStorage.getItem(STORAGE_KEY_ACTIVE_ID) || 'session-default';
  });

  // UI States
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [retrievalMode, setRetrievalMode] = useState('hybrid'); // 'hybrid' or 'dense'
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [desktopSidebarCollapsed, setDesktopSidebarCollapsed] = useState(() => {
    return localStorage.getItem(STORAGE_KEY_COLLAPSED) === 'true';
  });

  // Backend state
  const [backendOnline, setBackendOnline] = useState(false);
  const [corpusInfo, setCorpusInfo] = useState(null);

  // Modal inspection state
  const [inspectingChunk, setInspectingChunk] = useState(null);

  // Active session helper
  const currentSession =
    sessions.find((s) => s.id === activeSessionId) || sessions[0] || {
      id: 'session-default',
      title: 'New Conversation',
      messages: [],
    };

  // Sync sessions to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY_SESSIONS, JSON.stringify(sessions));
      localStorage.setItem(STORAGE_KEY_ACTIVE_ID, activeSessionId);
    } catch (e) {
      console.warn('Could not save sessions to localStorage:', e);
    }
  }, [sessions, activeSessionId]);

  // Sync desktop collapse state
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY_COLLAPSED, String(desktopSidebarCollapsed));
  }, [desktopSidebarCollapsed]);

  // Backend Health check & Corpus info on mount and periodic interval
  useEffect(() => {
    const checkStatus = async () => {
      const health = await checkBackendHealth();
      setBackendOnline(health.online);

      if (health.online && !corpusInfo) {
        const info = await getCorpusInfo();
        setCorpusInfo(info);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 20000); // Poll every 20s
    return () => clearInterval(interval);
  }, [corpusInfo]);

  // Handlers for Session Management
  const handleNewChat = () => {
    const newSession = {
      id: `session-${Date.now()}`,
      title: 'New Conversation',
      createdAt: new Date().toISOString(),
      messages: [],
    };
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
    setErrorMessage(null);
    setMobileSidebarOpen(false);
  };

  const handleDeleteSession = (sessionId) => {
    setSessions((prev) => {
      const updated = prev.filter((s) => s.id !== sessionId);
      if (updated.length === 0) {
        const fresh = {
          id: `session-${Date.now()}`,
          title: 'New Conversation',
          createdAt: new Date().toISOString(),
          messages: [],
        };
        setActiveSessionId(fresh.id);
        return [fresh];
      }
      if (activeSessionId === sessionId) {
        setActiveSessionId(updated[0].id);
      }
      return updated;
    });
  };

  const handleClearCurrentSession = () => {
    setSessions((prev) =>
      prev.map((s) => (s.id === activeSessionId ? { ...s, messages: [] } : s))
    );
    setErrorMessage(null);
  };

  // Send question to RAG backend
  const handleSendMessage = async (queryText) => {
    if (!queryText.trim() || isLoading) return;

    setErrorMessage(null);
    const userMessage = {
      role: 'user',
      content: queryText.trim(),
      timestamp: new Date().toISOString(),
    };

    // Update active session with user message immediately
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id === activeSessionId) {
          const isFirstMessage = s.messages.length === 0;
          const newTitle = isFirstMessage
            ? queryText.trim().slice(0, 36) + (queryText.trim().length > 36 ? '...' : '')
            : s.title;

          return {
            ...s,
            title: newTitle,
            messages: [...s.messages, userMessage],
          };
        }
        return s;
      })
    );

    setIsLoading(true);
    const startTime = performance.now();

    try {
      const response = await submitRAGQuery({
        query: queryText,
        topK: 5,
        useHybrid: retrievalMode === 'hybrid',
        retrievalMode,
      });

      const latencyMs = Math.round(performance.now() - startTime);

      const assistantMessage = {
        role: 'assistant',
        content: response.answer,
        citations: response.citations || [],
        latency: response.latency_ms || latencyMs,
        is_grounded: response.is_grounded,
        is_refusal: response.is_refusal,
        unverified_draft: response.unverified_draft,
        retrievalMode,
        timestamp: new Date().toISOString(),
      };

      setSessions((prev) =>
        prev.map((s) =>
          s.id === activeSessionId
            ? { ...s, messages: [...s.messages, assistantMessage] }
            : s
        )
      );
    } catch (err) {
      console.error('Query execution failed:', err);
      const friendlyError = err.message || 'Error communicating with backend service.';
      setErrorMessage(friendlyError);

      const errorAssistantMessage = {
        role: 'assistant',
        content: `**Error:** ${friendlyError}\n\n*Please ensure the backend FastAPI service is running on port 8000.*`,
        citations: [],
        is_grounded: false,
        is_refusal: true,
        latency: 0,
        retrievalMode,
        timestamp: new Date().toISOString(),
      };

      setSessions((prev) =>
        prev.map((s) =>
          s.id === activeSessionId
            ? { ...s, messages: [...s.messages, errorAssistantMessage] }
            : s
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-white dark:bg-[#121214] text-gray-900 dark:text-[#ececf1] font-sans transition-colors">
      {/* Sidebar Navigation */}
      <Sidebar
        isOpen={mobileSidebarOpen}
        onClose={() => setMobileSidebarOpen(false)}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={(id) => {
          setActiveSessionId(id);
          setMobileSidebarOpen(false);
          setErrorMessage(null);
        }}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
        corpusInfo={corpusInfo}
        isDesktopCollapsed={desktopSidebarCollapsed}
        onToggleDesktopCollapse={() =>
          setDesktopSidebarCollapsed((prev) => !prev)
        }
      />

      {/* Main Chat Workspace */}
      <div className="flex-1 flex flex-col h-full min-w-0 overflow-hidden relative">
        {/* Top Header */}
        <Header
          theme={theme}
          toggleTheme={toggleTheme}
          onClearConversation={handleClearCurrentSession}
          backendOnline={backendOnline}
          retrievalMode={retrievalMode}
          onChangeRetrievalMode={setRetrievalMode}
          onToggleSidebar={() => setMobileSidebarOpen((prev) => !prev)}
        />

        {/* Message Feed / Welcome Screen */}
        <ChatWindow
          messages={currentSession.messages}
          isLoading={isLoading}
          error={errorMessage}
          onSelectExample={handleSendMessage}
          onRetryQuestion={handleSendMessage}
          onInspectChunk={setInspectingChunk}
          corpusInfo={corpusInfo}
        />

        {/* Floating Input Area */}
        <ChatInput
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          retrievalMode={retrievalMode}
          onChangeRetrievalMode={setRetrievalMode}
        />
      </div>

      {/* Full Chunk Inspection Modal */}
      <SourceViewer
        isOpen={!!inspectingChunk}
        onClose={() => setInspectingChunk(null)}
        chunk={inspectingChunk}
      />
    </div>
  );
}
