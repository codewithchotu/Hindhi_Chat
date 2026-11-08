import React, { useState } from 'react';
import {
  ChevronDown,
  ChevronUp,
  FileText,
  Copy,
  Check,
  ExternalLink,
  Info,
} from 'lucide-react';

export default function SourceCitation({ citations = [], onInspectChunk }) {
  const [expandedIndex, setExpandedIndex] = useState(null);
  const [copiedIndex, setCopiedIndex] = useState(null);

  if (!citations || citations.length === 0) return null;

  const handleCopySnippet = (e, text, idx) => {
    e.stopPropagation();
    navigator.clipboard.writeText(text);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const formatScoreBadge = (score, scoreType) => {
    const type = scoreType || 'dense_similarity';

    if (type === 'rrf_score') {
      return {
        label: `RRF Score: ${score ? score.toFixed(4) : 'N/A'}`,
        tooltip: 'Reciprocal Rank Fusion score (ranking metric, not probability)',
        colorClass: 'bg-indigo-50 text-indigo-700 dark:bg-indigo-950/50 dark:text-indigo-300 border-indigo-200 dark:border-indigo-800',
      };
    }
    if (type === 'bm25_score') {
      return {
        label: `BM25: ${score ? score.toFixed(2) : 'N/A'}`,
        tooltip: 'BM25Okapi lexical term match score',
        colorClass: 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800',
      };
    }
    // Default: dense similarity
    const pct = score ? (score * 100).toFixed(1) : '0';
    return {
      label: `Similarity: ${pct}%`,
      tooltip: 'Cosine similarity in BGE-M3 1024-d multilingual embedding space',
      colorClass: 'bg-blue-50 text-blue-700 dark:bg-blue-950/50 dark:text-blue-300 border-blue-200 dark:border-blue-800',
    };
  };

  return (
    <div className="mt-4 pt-3 border-t border-gray-100 dark:border-[#2e2e38]/70">
      <div className="flex items-center justify-between mb-2.5">
        <div className="flex items-center space-x-1.5 text-xs font-semibold text-gray-700 dark:text-gray-300">
          <FileText className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
          <span>Verifiable Source Citations ({citations.length})</span>
        </div>
        <span className="text-[10px] text-gray-400 dark:text-gray-500">
          From Agni Ki Udaan PDF
        </span>
      </div>

      <div className="space-y-2">
        {citations.map((cite, idx) => {
          const isExpanded = expandedIndex === idx;
          const scoreInfo = formatScoreBadge(cite.score, cite.score_type);
          const snippetText = cite.snippet || cite.text || '';

          return (
            <div
              key={cite.chunk_id || idx}
              className={`rounded-lg border transition-all text-xs overflow-hidden ${
                isExpanded
                  ? 'border-blue-300 dark:border-blue-700/60 bg-blue-50/20 dark:bg-[#1a1a24]'
                  : 'border-gray-200 dark:border-[#2b2b34] bg-white dark:bg-[#17171c] hover:border-gray-300 dark:hover:border-[#383844]'
              }`}
            >
              {/* Citation Header Row */}
              <button
                type="button"
                onClick={() => setExpandedIndex(isExpanded ? null : idx)}
                className="w-full px-3 py-2 flex items-center justify-between text-left hover:bg-gray-50/60 dark:hover:bg-[#1f1f26] transition-colors"
              >
                <div className="flex items-center space-x-2 min-w-0 pr-2">
                  <span className="flex-shrink-0 px-2 py-0.5 rounded font-semibold text-[11px] bg-blue-100 text-blue-800 dark:bg-blue-900/60 dark:text-blue-200">
                    Page {cite.page}
                  </span>
                  <span className="font-medium text-gray-800 dark:text-gray-200 truncate">
                    {cite.section || 'General Section'}
                  </span>
                </div>

                <div className="flex items-center space-x-2 flex-shrink-0">
                  <span className="hidden sm:inline font-mono text-[10px] text-gray-400 dark:text-gray-500">
                    {cite.chunk_id}
                  </span>

                  <span
                    title={scoreInfo.tooltip}
                    className={`px-1.5 py-0.5 rounded text-[10px] font-mono border font-medium ${scoreInfo.colorClass}`}
                  >
                    {scoreInfo.label}
                  </span>

                  {isExpanded ? (
                    <ChevronUp className="w-3.5 h-3.5 text-gray-400" />
                  ) : (
                    <ChevronDown className="w-3.5 h-3.5 text-gray-400" />
                  )}
                </div>
              </button>

              {/* Collapsible Snippet Area */}
              {isExpanded && (
                <div className="px-3.5 py-3 bg-gray-50/70 dark:bg-[#141418] border-t border-gray-100 dark:border-[#282832]">
                  <div className="flex items-center justify-between pb-2 mb-2 border-b border-gray-200/60 dark:border-[#262630] text-[11px] text-gray-500 dark:text-gray-400">
                    <span className="font-mono">
                      Chunk ID: <strong className="text-gray-700 dark:text-gray-300">{cite.chunk_id}</strong>
                    </span>
                    <div className="flex items-center space-x-2">
                      <button
                        type="button"
                        onClick={(e) => handleCopySnippet(e, snippetText, idx)}
                        className="flex items-center space-x-1 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                        title="Copy Devanagari snippet"
                      >
                        {copiedIndex === idx ? (
                          <>
                            <Check className="w-3 h-3 text-emerald-500" />
                            <span className="text-emerald-500 text-[10px]">Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3 h-3" />
                            <span className="text-[10px]">Copy</span>
                          </>
                        )}
                      </button>

                      {onInspectChunk && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onInspectChunk(cite);
                          }}
                          className="flex items-center space-x-1 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                          title="View full passage modal"
                        >
                          <ExternalLink className="w-3 h-3" />
                          <span className="text-[10px]">Inspect</span>
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Devanagari Snippet Text */}
                  <div className="text-xs leading-relaxed text-gray-800 dark:text-gray-200 font-normal whitespace-pre-wrap font-sans">
                    {snippetText || '(No text snippet available in metadata)'}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
