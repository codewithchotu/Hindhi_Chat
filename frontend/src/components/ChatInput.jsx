import React, { useState, useRef, useEffect } from 'react';
import {
  ArrowUp,
  SlidersHorizontal,
  Loader2,
  X,
  Languages,
} from 'lucide-react';

export default function ChatInput({
  onSendMessage,
  isLoading,
  retrievalMode = 'hybrid',
  onChangeRetrievalMode,
}) {
  const [input, setInput] = useState('');
  const textareaRef = useRef(null);

  // Auto-resize textarea to fit content up to 140px
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        140
      )}px`;
    }
  }, [input]);

  // Detect script (Devanagari vs Latin)
  const isDevanagari = /[\u0900-\u097F]/.test(input);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    onSendMessage(input.trim());
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="sticky bottom-0 w-full bg-gradient-to-t from-white via-white/95 to-transparent dark:from-[#121214] dark:via-[#121214]/95 dark:to-transparent pt-3 pb-4 px-3 sm:px-4 z-20">
      <div className="max-w-3xl mx-auto">
        <form
          onSubmit={handleSubmit}
          className="relative flex flex-col rounded-2xl border border-gray-300/90 dark:border-[#2e2e38] bg-white dark:bg-[#1a1a1f] shadow-sm focus-within:border-blue-500 dark:focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500/20 transition-all p-2"
        >
          {/* Textarea */}
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question in English or हिंदी (e.g., कलाम को भारत रत्न किस वर्ष मिला?)..."
            disabled={isLoading}
            className="w-full max-h-36 resize-none bg-transparent px-3 py-1.5 text-sm text-gray-900 dark:text-[#ececf1] placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none leading-relaxed"
          />

          {/* Bottom Bar inside input: Language hint, Mode chip, Send button */}
          <div className="flex items-center justify-between pt-1 px-1">
            {/* Left: Language indicator & clear */}
            <div className="flex items-center space-x-2">
              {input.trim().length > 0 ? (
                <div className="flex items-center space-x-1.5">
                  <span
                    className={`inline-flex items-center space-x-1 px-1.5 py-0.5 rounded text-[10px] font-medium ${
                      isDevanagari
                        ? 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300'
                        : 'bg-blue-100 text-blue-800 dark:bg-blue-950/60 dark:text-blue-300'
                    }`}
                  >
                    <Languages className="w-3 h-3" />
                    <span>{isDevanagari ? 'Devanagari Hindi' : 'English Query'}</span>
                  </span>

                  <button
                    type="button"
                    onClick={() => setInput('')}
                    className="p-1 rounded-md text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                    title="Clear input"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ) : (
                <span className="hidden sm:inline text-[11px] text-gray-400 dark:text-gray-500">
                  Cross-Lingual Devanagari Search
                </span>
              )}
            </div>

            {/* Right: Retrieval Mode + Send button */}
            <div className="flex items-center space-x-2">
              <button
                type="button"
                onClick={() =>
                  onChangeRetrievalMode &&
                  onChangeRetrievalMode(retrievalMode === 'hybrid' ? 'dense' : 'hybrid')
                }
                className={`flex items-center space-x-1 px-2 py-1 rounded-lg text-xs font-medium transition-colors ${
                  retrievalMode === 'hybrid'
                    ? 'bg-blue-50 text-blue-600 dark:bg-blue-950/50 dark:text-blue-400'
                    : 'bg-gray-100 text-gray-600 dark:bg-[#25252e] dark:text-gray-400'
                }`}
                title={
                  retrievalMode === 'hybrid'
                    ? 'Hybrid mode active (Dense BGE-M3 + BM25Okapi with RRF)'
                    : 'Dense mode active (BGE-M3 only)'
                }
              >
                <SlidersHorizontal className="w-3 h-3" />
                <span className="text-[10px]">
                  {retrievalMode === 'hybrid' ? 'Hybrid RRF' : 'Dense Only'}
                </span>
              </button>

              <button
                type="submit"
                disabled={!input.trim() || isLoading}
                aria-label="Send question"
                className={`p-2 rounded-xl transition-all ${
                  input.trim() && !isLoading
                    ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-sm'
                    : 'bg-gray-100 dark:bg-[#25252d] text-gray-400 dark:text-gray-600 cursor-not-allowed'
                }`}
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
                ) : (
                  <ArrowUp className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>
        </form>

        <p className="mt-2 text-center text-[11px] text-gray-400 dark:text-gray-500">
          Source document: <em>Agni Ki Udaan Adhyayan Sahayika</em> (22 Pages). Dynamic citations generated from vector DB metadata.
        </p>
      </div>
    </div>
  );
}
