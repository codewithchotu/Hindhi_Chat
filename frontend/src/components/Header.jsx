import React from 'react';
import {
  Menu,
  RotateCcw,
  BookOpen,
  SlidersHorizontal,
  Activity,
  Layers,
} from 'lucide-react';
import ThemeToggle from './ThemeToggle';

export default function Header({
  theme,
  toggleTheme,
  onClearConversation,
  backendOnline,
  retrievalMode = 'hybrid',
  onChangeRetrievalMode,
  onToggleSidebar,
}) {
  return (
    <header className="sticky top-0 z-30 w-full border-b border-gray-200 dark:border-[#22222a] bg-white/90 dark:bg-[#121214]/90 backdrop-blur-md px-3 sm:px-4 py-2.5">
      <div className="flex items-center justify-between">
        {/* Left: Sidebar toggle + App Identity */}
        <div className="flex items-center space-x-2.5">
          <button
            type="button"
            onClick={onToggleSidebar}
            className="p-1.5 rounded-lg border border-gray-200 dark:border-[#2b2b36] hover:bg-gray-100 dark:hover:bg-[#1e1e26] text-gray-600 dark:text-gray-300 transition-colors"
            title="Toggle sidebar"
          >
            <Menu className="w-4 h-4" />
          </button>

          <div className="flex items-center space-x-2">
            <div className="w-7 h-7 rounded-lg bg-blue-600/10 dark:bg-blue-500/20 text-blue-600 dark:text-blue-400 flex items-center justify-center font-bold text-xs">
              <BookOpen className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="flex items-center space-x-1.5">
                <h1 className="font-semibold text-xs sm:text-sm text-gray-900 dark:text-[#ececf1]">
                  Hindi RAG Engine
                </h1>
                {/* Status Indicator */}
                <span
                  className="flex items-center space-x-1 px-1.5 py-0.5 rounded-full text-[10px] font-medium"
                  title={backendOnline ? 'Backend service online' : 'Backend service unreachable'}
                >
                  <span
                    className={`w-2 h-2 rounded-full ${
                      backendOnline ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'
                    }`}
                  />
                  <span className="hidden sm:inline text-gray-500 dark:text-gray-400">
                    {backendOnline ? 'Online' : 'Offline'}
                  </span>
                </span>
              </div>
              <p className="hidden md:block text-[11px] text-gray-400 dark:text-gray-500">
                Agni Ki Udaan • Multilingual BGE-M3 + Gemini
              </p>
            </div>
          </div>
        </div>

        {/* Right: Retrieval Strategy Selector + Reset + Theme Toggle */}
        <div className="flex items-center space-x-2">
          {/* Mode Selector Pill */}
          <div className="flex items-center space-x-1 bg-gray-100 dark:bg-[#1c1c22] p-0.5 rounded-xl border border-gray-200/80 dark:border-[#292934] text-[11px]">
            <button
              type="button"
              onClick={() => onChangeRetrievalMode('hybrid')}
              className={`px-2 py-1 rounded-lg font-medium transition-colors ${
                retrievalMode === 'hybrid'
                  ? 'bg-white dark:bg-[#282832] text-blue-600 dark:text-blue-400 shadow-sm'
                  : 'text-gray-500 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200'
              }`}
              title="Dense + BM25 combined with Reciprocal Rank Fusion"
            >
              Hybrid RRF
            </button>
            <button
              type="button"
              onClick={() => onChangeRetrievalMode('dense')}
              className={`px-2 py-1 rounded-lg font-medium transition-colors ${
                retrievalMode === 'dense'
                  ? 'bg-white dark:bg-[#282832] text-blue-600 dark:text-blue-400 shadow-sm'
                  : 'text-gray-500 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200'
              }`}
              title="Dense semantic search with BGE-M3"
            >
              Dense
            </button>
          </div>

          {/* Reset Current Chat */}
          <button
            type="button"
            onClick={onClearConversation}
            className="flex items-center space-x-1 px-2.5 py-1.5 text-xs rounded-xl border border-gray-200 dark:border-[#2b2b36] hover:bg-gray-100 dark:hover:bg-[#1e1e26] text-gray-600 dark:text-gray-300 transition-colors"
            title="Reset current conversation"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Reset</span>
          </button>

          {/* Theme Toggle */}
          <ThemeToggle theme={theme} toggleTheme={toggleTheme} />
        </div>
      </div>
    </header>
  );
}
