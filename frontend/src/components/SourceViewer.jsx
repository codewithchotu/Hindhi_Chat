import React from 'react';
import { X, Copy, Check, FileText } from 'lucide-react';

export default function SourceViewer({ isOpen, onClose, chunk }) {
  const [copied, setCopied] = React.useState(false);

  if (!isOpen || !chunk) return null;

  const handleCopy = () => {
    const textToCopy = chunk.snippet || chunk.text || '';
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-white dark:bg-[#18181d] border border-gray-200 dark:border-[#2e2e38] rounded-2xl max-w-2xl w-full p-6 shadow-2xl relative flex flex-col max-h-[85vh]">
        {/* Modal Header */}
        <div className="flex items-center justify-between pb-4 border-b border-gray-100 dark:border-[#282834]">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-lg bg-blue-100 dark:bg-blue-950/60 text-blue-700 dark:text-blue-300 flex items-center justify-center font-bold text-xs">
              P{chunk.page}
            </div>
            <div>
              <h3 className="font-semibold text-sm text-gray-900 dark:text-white">
                {chunk.section || 'Corpus Source Passage'}
              </h3>
              <p className="text-[11px] font-mono text-gray-400">
                Chunk ID: {chunk.chunk_id} • Source: {chunk.source_doc || 'Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf'}
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-[#25252e] text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors"
            title="Close viewer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Content - Devanagari Passage */}
        <div className="my-4 flex-1 overflow-y-auto pr-1">
          <div className="p-4 rounded-xl bg-gray-50 dark:bg-[#121215] border border-gray-200/60 dark:border-[#262630] text-sm text-gray-800 dark:text-gray-200 leading-relaxed font-sans whitespace-pre-wrap">
            {chunk.snippet || chunk.text || '(No text available)'}
          </div>
        </div>

        {/* Modal Footer */}
        <div className="pt-3 border-t border-gray-100 dark:border-[#282834] flex items-center justify-between text-xs">
          <div className="flex items-center space-x-2">
            <span className="font-mono text-gray-500 dark:text-gray-400 text-[11px]">
              Retrieval Metric: {chunk.score_type || 'dense_similarity'} ({typeof chunk.score === 'number' ? chunk.score.toFixed(4) : chunk.score})
            </span>
          </div>

          <div className="flex items-center space-x-2">
            <button
              type="button"
              onClick={handleCopy}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border border-gray-200 dark:border-[#2e2e38] hover:bg-gray-100 dark:hover:bg-[#25252e] text-gray-700 dark:text-gray-300 transition-colors text-xs"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-500" />
                  <span className="text-emerald-500">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy Passage</span>
                </>
              )}
            </button>
            <button
              type="button"
              onClick={onClose}
              className="px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium transition-colors text-xs"
            >
              Done
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
