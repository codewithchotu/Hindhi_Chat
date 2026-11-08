import React, { useState } from 'react';
import {
  User,
  Sparkles,
  Copy,
  Check,
  Clock,
  RotateCcw,
  ShieldCheck,
  AlertTriangle,
  Info,
  SlidersHorizontal,
} from 'lucide-react';
import MarkdownRenderer from './MarkdownRenderer';
import SourceCitation from './SourceCitation';

export default function ChatMessage({
  message,
  onRetry,
  onInspectChunk,
}) {
  const isUser = message.role === 'user';
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content || '');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const renderStatusBadge = () => {
    if (isUser) return null;

    if (message.is_refusal) {
      return (
        <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-amber-50 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300 border border-amber-200 dark:border-amber-800/60">
          <Info className="w-3 h-3 flex-shrink-0" />
          <span>Document Refusal</span>
        </span>
      );
    }

    if (message.is_grounded === false) {
      return (
        <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-rose-50 text-rose-800 dark:bg-rose-950/60 dark:text-rose-300 border border-rose-200 dark:border-rose-800/60">
          <AlertTriangle className="w-3 h-3 flex-shrink-0" />
          <span>Grounding Warning</span>
        </span>
      );
    }

    if (message.is_grounded === true) {
      return (
        <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-50 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800/60">
          <ShieldCheck className="w-3 h-3 flex-shrink-0" />
          <span>Verified Grounded</span>
        </span>
      );
    }

    return null;
  };

  return (
    <div
      className={`py-5 transition-colors ${
        isUser
          ? 'bg-transparent'
          : 'bg-gray-50/70 dark:bg-[#16161a]/80 border-y border-gray-100 dark:border-[#22222a]/50'
      }`}
    >
      <div className="max-w-3xl mx-auto px-4 flex space-x-3 sm:space-x-4">
        {/* Avatar */}
        <div className="flex-shrink-0 pt-0.5">
          {isUser ? (
            <div className="w-7 h-7 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center text-gray-700 dark:text-gray-300">
              <User className="w-4 h-4" />
            </div>
          ) : (
            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-white shadow-sm">
              <Sparkles className="w-4 h-4" />
            </div>
          )}
        </div>

        {/* Content Body */}
        <div className="flex-1 min-w-0">
          {/* Header Row */}
          <div className="flex items-center justify-between mb-1.5 flex-wrap gap-2">
            <div className="flex items-center space-x-2">
              <span className="text-xs font-semibold text-gray-900 dark:text-[#ececf1]">
                {isUser ? 'You' : 'Hindi RAG Assistant'}
              </span>
              {renderStatusBadge()}
            </div>

            <div className="flex items-center space-x-2.5 text-[11px] text-gray-400 dark:text-gray-500">
              {message.latency && (
                <span className="flex items-center space-x-1" title="End-to-End Processing Latency">
                  <Clock className="w-3 h-3" />
                  <span>{message.latency} ms</span>
                </span>
              )}

              {message.retrievalMode && (
                <span className="hidden sm:flex items-center space-x-1" title="Retrieval Strategy">
                  <SlidersHorizontal className="w-3 h-3" />
                  <span className="capitalize">{message.retrievalMode}</span>
                </span>
              )}

              {!isUser && message.content && (
                <button
                  type="button"
                  onClick={handleCopy}
                  className="p-1 rounded hover:bg-gray-200/60 dark:hover:bg-[#25252e] text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors"
                  title="Copy response"
                >
                  {copied ? (
                    <Check className="w-3.5 h-3.5 text-emerald-500" />
                  ) : (
                    <Copy className="w-3.5 h-3.5" />
                  )}
                </button>
              )}

              {isUser && onRetry && (
                <button
                  type="button"
                  onClick={() => onRetry(message.content)}
                  className="p-1 rounded hover:bg-gray-200/60 dark:hover:bg-[#25252e] text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors"
                  title="Rerun this question"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>

          {/* Answer / Question Body */}
          {isUser ? (
            <p className="text-sm text-gray-900 dark:text-[#ececf1] leading-relaxed font-normal whitespace-pre-wrap">
              {message.content}
            </p>
          ) : (
            <div className="text-sm text-gray-800 dark:text-[#e4e4ea] leading-relaxed">
              <MarkdownRenderer content={message.content} />
            </div>
          )}

          {/* Citations block for assistant */}
          {!isUser && message.citations && message.citations.length > 0 && (
            <SourceCitation
              citations={message.citations}
              onInspectChunk={onInspectChunk}
            />
          )}
        </div>
      </div>
    </div>
  );
}
