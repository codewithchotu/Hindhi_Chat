import React, { useState } from 'react';
import {
  Plus,
  MessageSquare,
  Trash2,
  X,
  BookOpen,
  Database,
  Search,
  ExternalLink,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';

export default function Sidebar({
  isOpen,
  onClose,
  sessions = [],
  activeSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  onClearAllSessions,
  corpusInfo,
  isDesktopCollapsed,
  onToggleDesktopCollapse,
}) {
  const [searchQuery, setSearchQuery] = useState('');

  const filteredSessions = sessions.filter((s) =>
    (s.title || 'New Chat').toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <>
      {/* Mobile Backdrop Overlay */}
      {isOpen && (
        <div
          onClick={onClose}
          className="fixed inset-0 z-40 bg-black/50 md:hidden backdrop-blur-sm"
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-50 flex flex-col bg-gray-50 dark:bg-[#141418] border-r border-gray-200 dark:border-[#22222a] transition-all duration-200 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        } ${
          isDesktopCollapsed ? 'md:w-16' : 'w-72 md:w-64 lg:w-72'
        }`}
      >
        {/* Sidebar Header */}
        <div className="p-3 border-b border-gray-200/70 dark:border-[#22222a] flex items-center justify-between">
          {!isDesktopCollapsed && (
            <button
              type="button"
              onClick={onNewChat}
              className="flex-1 flex items-center justify-center space-x-2 px-3 py-2 rounded-xl bg-white dark:bg-[#1f1f26] border border-gray-200 dark:border-[#2e2e38] text-xs font-medium text-gray-800 dark:text-gray-200 hover:border-blue-500 dark:hover:border-blue-500 hover:text-blue-600 dark:hover:text-blue-400 shadow-sm transition-all"
            >
              <Plus className="w-4 h-4" />
              <span>New Conversation</span>
            </button>
          )}

          {isDesktopCollapsed && (
            <button
              type="button"
              onClick={onNewChat}
              className="mx-auto p-2 rounded-xl bg-white dark:bg-[#1f1f26] border border-gray-200 dark:border-[#2e2e38] text-gray-700 dark:text-gray-300 hover:text-blue-600"
              title="New Chat"
            >
              <Plus className="w-4 h-4" />
            </button>
          )}

          {/* Mobile close button */}
          <button
            type="button"
            onClick={onClose}
            className="md:hidden p-1.5 ml-2 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Search Chats Input (when expanded) */}
        {!isDesktopCollapsed && sessions.length > 2 && (
          <div className="px-3 pt-2">
            <div className="relative flex items-center">
              <Search className="w-3.5 h-3.5 absolute left-2.5 text-gray-400" />
              <input
                type="text"
                placeholder="Search conversations..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-8 pr-3 py-1.5 rounded-lg text-xs bg-white dark:bg-[#1b1b22] border border-gray-200 dark:border-[#2c2c36] text-gray-800 dark:text-gray-200 placeholder-gray-400 focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>
        )}

        {/* Sessions List */}
        <div className="flex-1 overflow-y-auto px-2 py-3 space-y-1">
          {!isDesktopCollapsed && (
            <div className="px-2 pb-1 text-[10px] font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500">
              Conversations ({sessions.length})
            </div>
          )}

          {sessions.length === 0 && !isDesktopCollapsed && (
            <div className="px-3 py-6 text-center text-xs text-gray-400 dark:text-gray-500">
              No conversations yet. Ask a question to begin.
            </div>
          )}

          {filteredSessions.map((session) => {
            const isActive = session.id === activeSessionId;

            if (isDesktopCollapsed) {
              return (
                <button
                  key={session.id}
                  type="button"
                  onClick={() => onSelectSession(session.id)}
                  title={session.title || 'Conversation'}
                  className={`w-full p-2.5 rounded-lg flex items-center justify-center transition-colors ${
                    isActive
                      ? 'bg-blue-100 dark:bg-blue-950/60 text-blue-600 dark:text-blue-300'
                      : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-[#1d1d24]'
                  }`}
                >
                  <MessageSquare className="w-4 h-4" />
                </button>
              );
            }

            return (
              <div
                key={session.id}
                className={`group flex items-center justify-between px-2.5 py-2 rounded-xl text-xs transition-colors cursor-pointer ${
                  isActive
                    ? 'bg-white dark:bg-[#202028] text-blue-600 dark:text-blue-400 font-medium shadow-sm border border-gray-200/80 dark:border-[#2e2e3a]'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100/80 dark:hover:bg-[#1a1a22]'
                }`}
                onClick={() => onSelectSession(session.id)}
              >
                <div className="flex items-center space-x-2 min-w-0 pr-1">
                  <MessageSquare
                    className={`w-3.5 h-3.5 flex-shrink-0 ${
                      isActive ? 'text-blue-600 dark:text-blue-400' : 'text-gray-400'
                    }`}
                  />
                  <span className="truncate">{session.title || 'Untitled Query'}</span>
                </div>

                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteSession(session.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-50 dark:hover:bg-red-950/40 text-gray-400 hover:text-red-500 transition-all flex-shrink-0"
                  title="Delete chat"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            );
          })}
        </div>

        {/* Sidebar Footer — Corpus Metadata Badge */}
        {!isDesktopCollapsed ? (
          <div className="p-3 border-t border-gray-200/70 dark:border-[#22222a] bg-white/60 dark:bg-[#16161c]/80 text-[11px]">
            <div className="flex items-center space-x-2 mb-1.5">
              <div className="w-6 h-6 rounded-md bg-blue-100 dark:bg-blue-950/80 text-blue-600 dark:text-blue-400 flex items-center justify-center flex-shrink-0">
                <BookOpen className="w-3.5 h-3.5" />
              </div>
              <div className="min-w-0">
                <p className="font-semibold text-gray-900 dark:text-gray-200 truncate">
                  {corpusInfo?.title || 'Agni Ki Udaan'}
                </p>
                <p className="text-[10px] text-gray-500 dark:text-gray-400 truncate">
                  22 Pages • 78 Chunks Indexed
                </p>
              </div>
            </div>

            <div className="pt-2 border-t border-gray-100 dark:border-[#242430] flex items-center justify-between text-[10px] text-gray-400 dark:text-gray-500">
              <span className="flex items-center space-x-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block" />
                <span>BGE-M3 + ChromaDB</span>
              </span>
              <button
                type="button"
                onClick={onToggleDesktopCollapse}
                className="hidden md:block p-1 rounded hover:bg-gray-100 dark:hover:bg-[#202028] text-gray-400"
                title="Collapse sidebar"
              >
                <ChevronLeft className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        ) : (
          <div className="p-2 border-t border-gray-200 dark:border-[#22222a] text-center">
            <button
              type="button"
              onClick={onToggleDesktopCollapse}
              className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-[#202028] text-gray-400 mx-auto block"
              title="Expand sidebar"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}
      </aside>
    </>
  );
}
