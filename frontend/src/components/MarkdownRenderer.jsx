import React from 'react';

/**
 * Lightweight Markdown renderer tailored for RAG assistant answers.
 * Parses headers, bold, italics, inline code, code blocks, lists, blockquotes,
 * and highlights citation markers like [doc_p8_c3] or [Page 8] cleanly.
 */
export default function MarkdownRenderer({ content, onCitationClick }) {
  if (!content) return null;

  const lines = content.split('\n');
  const elements = [];
  let inCodeBlock = false;
  let codeBlockContent = [];
  let codeBlockLang = '';

  const renderInline = (text) => {
    // Match inline formatting: bold, italic, inline code, and citation tags [doc_pX_cY]
    const parts = [];
    let remaining = text;
    let keyIdx = 0;

    // Pattern to match citations [doc_pX_cY] or [Page X], bold **text**, italic *text*, code `text`
    const pattern = /(\[doc_p\d+_c\d+\]|\[(?:Page|पृष्ठ)\s*\d+\]|\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
    let match;
    let lastIndex = 0;

    while ((match = pattern.exec(text)) !== null) {
      // Text before match
      if (match.index > lastIndex) {
        parts.push(
          <span key={keyIdx++}>{text.substring(lastIndex, match.index)}</span>
        );
      }

      const matchStr = match[0];

      if (matchStr.startsWith('[doc_p') || matchStr.startsWith('[Page') || matchStr.startsWith('[पृष्ठ')) {
        // Citation badge
        const cleanTag = matchStr.replace(/[\[\]]/g, '');
        parts.push(
          <span
            key={keyIdx++}
            onClick={() => onCitationClick && onCitationClick(cleanTag)}
            className="inline-flex items-center px-1.5 py-0.2 mx-0.5 rounded text-[11px] font-mono font-medium bg-blue-100 text-blue-800 dark:bg-blue-950/80 dark:text-blue-300 border border-blue-200 dark:border-blue-800/60 cursor-default"
            title={`Citation Reference: ${cleanTag}`}
          >
            {matchStr}
          </span>
        );
      } else if (matchStr.startsWith('**') && matchStr.endsWith('**')) {
        // Bold
        parts.push(
          <strong key={keyIdx++} className="font-semibold text-gray-900 dark:text-white">
            {matchStr.slice(2, -2)}
          </strong>
        );
      } else if (matchStr.startsWith('*') && matchStr.endsWith('*')) {
        // Italic
        parts.push(
          <em key={keyIdx++} className="italic text-gray-800 dark:text-gray-200">
            {matchStr.slice(1, -1)}
          </em>
        );
      } else if (matchStr.startsWith('`') && matchStr.endsWith('`')) {
        // Inline code
        parts.push(
          <code
            key={keyIdx++}
            className="px-1.5 py-0.5 rounded text-xs font-mono bg-gray-100 dark:bg-[#25252d] text-blue-600 dark:text-blue-400 border border-gray-200 dark:border-[#353540]"
          >
            {matchStr.slice(1, -1)}
          </code>
        );
      }

      lastIndex = pattern.lastIndex;
    }

    if (lastIndex < text.length) {
      parts.push(<span key={keyIdx++}>{text.substring(lastIndex)}</span>);
    }

    return parts.length > 0 ? parts : text;
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Code block toggle
    if (line.trim().startsWith('```')) {
      if (inCodeBlock) {
        elements.push(
          <div
            key={`code-${i}`}
            className="my-3 rounded-lg overflow-hidden border border-gray-200 dark:border-[#2e2e38] bg-gray-900 text-gray-100 font-mono text-xs"
          >
            {codeBlockLang && (
              <div className="px-3 py-1 bg-gray-800 text-[10px] text-gray-400 uppercase font-semibold border-b border-gray-700">
                {codeBlockLang}
              </div>
            )}
            <pre className="p-3 overflow-x-auto leading-relaxed">
              <code>{codeBlockContent.join('\n')}</code>
            </pre>
          </div>
        );
        inCodeBlock = false;
        codeBlockContent = [];
        codeBlockLang = '';
      } else {
        inCodeBlock = true;
        codeBlockLang = line.trim().slice(3).trim();
      }
      continue;
    }

    if (inCodeBlock) {
      codeBlockContent.push(line);
      continue;
    }

    // Headers
    if (line.startsWith('### ')) {
      elements.push(
        <h3 key={i} className="text-sm font-semibold text-gray-900 dark:text-white mt-3 mb-1">
          {renderInline(line.slice(4))}
        </h3>
      );
    } else if (line.startsWith('## ')) {
      elements.push(
        <h2 key={i} className="text-base font-semibold text-gray-900 dark:text-white mt-4 mb-2">
          {renderInline(line.slice(3))}
        </h2>
      );
    } else if (line.startsWith('# ')) {
      elements.push(
        <h1 key={i} className="text-lg font-bold text-gray-900 dark:text-white mt-4 mb-2">
          {renderInline(line.slice(2))}
        </h1>
      );
    }
    // Blockquote
    else if (line.startsWith('> ')) {
      elements.push(
        <blockquote
          key={i}
          className="my-2 pl-3 border-l-2 border-blue-500 text-xs text-gray-600 dark:text-gray-400 italic"
        >
          {renderInline(line.slice(2))}
        </blockquote>
      );
    }
    // Unordered list
    else if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
      const listText = line.trim().slice(2);
      elements.push(
        <li key={i} className="ml-4 list-disc text-sm text-gray-800 dark:text-gray-200 my-1">
          {renderInline(listText)}
        </li>
      );
    }
    // Ordered list
    else if (/^\d+\.\s/.test(line.trim())) {
      const listText = line.trim().replace(/^\d+\.\s/, '');
      elements.push(
        <li key={i} className="ml-4 list-decimal text-sm text-gray-800 dark:text-gray-200 my-1">
          {renderInline(listText)}
        </li>
      );
    }
    // Empty line (paragraph break)
    else if (!line.trim()) {
      elements.push(<div key={i} className="h-2" />);
    }
    // Standard paragraph line
    else {
      elements.push(
        <p key={i} className="text-sm text-gray-800 dark:text-gray-200 leading-relaxed my-1">
          {renderInline(line)}
        </p>
      );
    }
  }

  return <div className="space-y-1 font-normal">{elements}</div>;
}
