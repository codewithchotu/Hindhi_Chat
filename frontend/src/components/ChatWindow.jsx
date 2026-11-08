import React, { useRef, useEffect } from 'react';
import ChatMessage from './ChatMessage';
import ExampleQuestions from './ExampleQuestions';
import {
  Sparkles,
  Loader2,
  BookOpen,
  ShieldCheck,
  Languages,
  Database,
  AlertCircle,
  RotateCcw,
} from 'lucide-react';

export default function ChatWindow({
  messages = [],
  isLoading = false,
  error = null,
  onSelectExample,
  onRetryQuestion,
  onInspectChunk,
  corpusInfo,
}) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Empty state when no conversation has started
  if (messages.length === 0) {
    return (
      <div className="flex-1 overflow-y-auto px-4 py-8 flex flex-col items-center justify-center">
        <div className="max-w-xl w-full text-center mx-auto mb-6">
          {/* Logo Badge */}
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-700 text-white mx-auto flex items-center justify-center shadow-lg shadow-blue-500/10 mb-4">
            <BookOpen className="w-7 h-7" />
          </div>

          <h2 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white tracking-tight">
            Cross-Lingual Hindi RAG Engine
          </h2>

          <p className="text-xs sm:text-sm text-gray-600 dark:text-gray-400 mt-2 leading-relaxed">
            Semantic Question-Answering over Dr. A.P.J. Abdul Kalam's Hindi biography{' '}
            <strong className="text-gray-900 dark:text-gray-200">
              <em>अग्नि की उड़ान (Agni Ki Udaan Adhyayan Sahayika)</em>
            </strong>
            . Inquire in either English or Hindi with strictly verifiable vector citations.
          </p>

          {/* Feature Badges */}
          <div className="flex flex-wrap items-center justify-center gap-2 mt-4 text-[11px]">
            <span className="flex items-center space-x-1 px-2.5 py-1 rounded-full bg-blue-50 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300 border border-blue-200/60 dark:border-blue-900/40">
              <Languages className="w-3 h-3" />
              <span>Cross-Lingual (EN ⇄ HI)</span>
            </span>

            <span className="flex items-center space-x-1 px-2.5 py-1 rounded-full bg-indigo-50 dark:bg-indigo-950/40 text-indigo-700 dark:text-indigo-300 border border-indigo-200/60 dark:border-indigo-900/40">
              <Database className="w-3 h-3" />
              <span>78 Chunks • BGE-M3 (1024-d)</span>
            </span>

            <span className="flex items-center space-x-1 px-2.5 py-1 rounded-full bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 border border-emerald-200/60 dark:border-emerald-900/40">
              <ShieldCheck className="w-3 h-3" />
              <span>Strict Grounding & Citations</span>
            </span>
          </div>
        </div>

        {/* Quick Example Questions Grid */}
        <ExampleQuestions onSelectQuestion={onSelectExample} />
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      {/* Messages Feed */}
      {messages.map((msg, idx) => (
        <ChatMessage
          key={idx}
          message={msg}
          onRetry={onRetryQuestion}
          onInspectChunk={onInspectChunk}
        />
      ))}

      {/* Loading Skeleton */}
      {isLoading && (
        <div className="py-5 bg-gray-50/70 dark:bg-[#16161a]/80 border-t border-gray-100 dark:border-[#22222a]/50">
          <div className="max-w-3xl mx-auto px-4 flex space-x-3 sm:space-x-4">
            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-white flex-shrink-0 shadow-sm">
              <Loader2 className="w-4 h-4 animate-spin" />
            </div>
            <div className="flex-1 space-y-2">
              <div className="flex items-center space-x-2">
                <span className="text-xs font-semibold text-gray-900 dark:text-[#ececf1]">
                  Hindi RAG Assistant
                </span>
                <span className="text-[10px] text-blue-600 dark:text-blue-400 font-mono animate-pulse">
                  retrieving context...
                </span>
              </div>

              <div className="h-4 bg-gray-200 dark:bg-[#252530] rounded-md w-3/4 animate-pulse" />
              <div className="h-4 bg-gray-200 dark:bg-[#252530] rounded-md w-1/2 animate-pulse" />
            </div>
          </div>
        </div>
      )}

      {/* Error Banner */}
      {error && (
        <div className="max-w-3xl mx-auto px-4 my-4">
          <div className="p-4 rounded-xl bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900/50 flex items-start space-x-3 text-xs text-red-700 dark:text-red-300">
            <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="font-semibold mb-0.5">Communication Error</p>
              <p className="leading-relaxed">{error}</p>
            </div>
            {onRetryQuestion && (
              <button
                type="button"
                onClick={() => onRetryQuestion(messages[messages.length - 1]?.content)}
                className="px-2.5 py-1 rounded-lg bg-red-100 dark:bg-red-900/60 hover:bg-red-200 text-red-800 dark:text-red-200 flex items-center space-x-1"
              >
                <RotateCcw className="w-3 h-3" />
                <span>Retry</span>
              </button>
            )}
          </div>
        </div>
      )}

      <div ref={bottomRef} className="h-4" />
    </div>
  );
}
