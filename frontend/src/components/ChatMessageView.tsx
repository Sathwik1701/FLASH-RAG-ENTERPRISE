import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { CitationBadge } from '../types';
import { BookOpen, Copy, Check } from 'lucide-react';

interface ChatMessageViewProps {
  role: 'user' | 'assistant';
  content: string;
  citations?: CitationBadge[];
  isStreaming?: boolean;
  onCitationClick?: (docId: string, page: number, citation?: CitationBadge) => void;
}

export const ChatMessageView: React.FC<ChatMessageViewProps> = ({
  role,
  content,
  citations = [],
  isStreaming = false,
  onCitationClick,
}) => {
  const [copied, setCopied] = React.useState(false);

  // Citation matching regex: [Doc: <doc_name_or_id>, Page: <page_num>] or variations
  const citationRegex = /\[Doc:\s*([^,\]]+),\s*Page:\s*(\d+)\]/gi;

  // Build lookup maps for rapid resolution
  const citationMap = new Map<string, CitationBadge>();
  citations.forEach((badge) => {
    citationMap.set(`${badge.document_id}:${badge.page_number}`, badge);
    citationMap.set(`${badge.filename}:${badge.page_number}`, badge);
    citationMap.set(`p:${badge.page_number}`, badge);
  });

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Pre-process text to convert citation bracket strings into custom interactive tokens or clean badges
  const formatTextCitations = (raw: string) => {
    return raw.replace(citationRegex, (_match, docName, pageNum) => {
      const cleanDoc = docName.trim();
      const cleanPage = pageNum.trim();
      return ` ||CITE:${cleanDoc}:${cleanPage}|| `;
    });
  };

  const processedContent = formatTextCitations(content);

  // Custom text renderer to turn ||CITE:doc:page|| tokens into clickable interactive pills
  const renderTextWithCitationPills = (text: string) => {
    const tokenRegex = /\|\|CITE:([^:]+):(\d+)\|\|/g;
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = tokenRegex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parts.push(text.slice(lastIndex, match.index));
      }

      const docName = match[1];
      const pageNum = parseInt(match[2], 10);
      const matchedBadge =
        citationMap.get(`${docName}:${pageNum}`) ||
        citationMap.get(`p:${pageNum}`);

      const displayName = matchedBadge?.filename || docName;
      const shortName = displayName.length > 20 ? displayName.slice(0, 18) + '…' : displayName;

      parts.push(
        <button
          key={`badge-${match.index}`}
          onClick={() => onCitationClick?.(docName, pageNum, matchedBadge)}
          title={
            matchedBadge
              ? `Document: ${matchedBadge.filename}\nPage: ${matchedBadge.page_number}\nSection: ${matchedBadge.section_header || 'N/A'}`
              : `Open ${displayName} (Page ${pageNum})`
          }
          className="inline-flex items-center gap-1 mx-1 px-2 py-0.5 text-[11px] font-medium rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-slate-100 border border-slate-700 transition-all cursor-pointer shadow-2xs align-baseline"
        >
          <BookOpen className="w-3 h-3 text-slate-400" />
          <span>{shortName} • p.{pageNum}</span>
        </button>
      );

      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < text.length) {
      parts.push(text.slice(lastIndex));
    }

    return parts.length > 0 ? parts : text;
  };

  return (
    <div className={`flex w-full mb-6 ${role === 'user' ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`relative transition-all duration-200 ${
          role === 'user'
            ? 'max-w-[80%] bg-slate-800 border border-slate-700/80 text-slate-100 rounded-2xl rounded-tr-xs px-5 py-3 shadow-xs font-normal text-xs'
            : 'max-w-[92%] w-full bg-slate-900 border border-slate-800/90 rounded-2xl rounded-tl-xs p-6 shadow-sm text-slate-200'
        }`}
      >
        {/* Assistant Header with Model Attribution & Copy Action */}
        {role === 'assistant' && (
          <div className="flex items-center justify-between pb-3.5 mb-4 border-b border-slate-800/80 text-xs text-slate-400">
            <div className="flex items-center gap-2.5">
              <span className="w-5 h-5 rounded-md bg-slate-800 border border-slate-700/80 flex items-center justify-center text-slate-300 font-semibold text-[10px]">
                AI
              </span>
              <span className="font-medium text-slate-300 uppercase tracking-wider text-[10px]">
                Synthesized Attribution
              </span>
              <span className="text-slate-600">•</span>
              <span className="text-slate-400 font-normal text-[11px]">
                Grounding Verified
              </span>
            </div>

            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 text-slate-400 hover:text-slate-200 px-2 py-1 rounded-md hover:bg-slate-800 transition-colors cursor-pointer text-[11px]"
              title="Copy answer"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-emerald-400 font-medium">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>
        )}

        {/* Content Body */}
        {role === 'user' ? (
          <div className="whitespace-pre-wrap leading-relaxed">{content}</div>
        ) : (
          <div className="prose prose-invert max-w-none text-xs leading-relaxed text-slate-300">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                h1: ({ children }) => (
                  <h1 className="text-sm font-semibold text-slate-100 border-b border-slate-800 pb-2 mb-3 mt-4 first:mt-0 tracking-tight">
                    {children}
                  </h1>
                ),
                h2: ({ children }) => (
                  <h2 className="text-xs font-semibold text-slate-100 border-b border-slate-800 pb-1.5 mb-2.5 mt-4 tracking-tight">
                    {children}
                  </h2>
                ),
                h3: ({ children }) => (
                  <h3 className="text-xs font-semibold text-slate-200 bg-slate-800/80 px-3 py-1.5 rounded-lg border-l-2 border-slate-500 mb-2 mt-4 first:mt-0">
                    {children}
                  </h3>
                ),
                h4: ({ children }) => (
                  <h4 className="text-xs font-medium text-slate-300 mb-1.5 mt-3">
                    {children}
                  </h4>
                ),
                p: ({ children }) => {
                  if (typeof children === 'string') {
                    return <p className="mb-3 leading-relaxed text-slate-300">{renderTextWithCitationPills(children)}</p>;
                  }
                  if (Array.isArray(children)) {
                    return (
                      <p className="mb-3 leading-relaxed text-slate-300">
                        {children.map((child, i) =>
                          typeof child === 'string' ? (
                            <React.Fragment key={i}>{renderTextWithCitationPills(child)}</React.Fragment>
                          ) : (
                            child
                          )
                        )}
                      </p>
                    );
                  }
                  return <p className="mb-3 leading-relaxed text-slate-300">{children}</p>;
                },
                ul: ({ children }) => <ul className="list-disc list-outside pl-5 mb-3 space-y-1.5 text-slate-300">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal list-outside pl-5 mb-3 space-y-1.5 text-slate-300">{children}</ol>,
                li: ({ children }) => {
                  if (typeof children === 'string') {
                    return <li className="pl-1 leading-relaxed">{renderTextWithCitationPills(children)}</li>;
                  }
                  if (Array.isArray(children)) {
                    return (
                      <li className="pl-1 leading-relaxed">
                        {children.map((child, i) =>
                          typeof child === 'string' ? (
                            <React.Fragment key={i}>{renderTextWithCitationPills(child)}</React.Fragment>
                          ) : (
                            child
                          )
                        )}
                      </li>
                    );
                  }
                  return <li className="pl-1 leading-relaxed">{children}</li>;
                },
                table: ({ children }) => (
                  <div className="overflow-x-auto my-4 rounded-xl border border-slate-800 bg-slate-900/50">
                    <table className="w-full text-left text-xs divide-y divide-slate-800">
                      {children}
                    </table>
                  </div>
                ),
                thead: ({ children }) => <thead className="bg-slate-800/80 font-medium text-slate-200">{children}</thead>,
                tbody: ({ children }) => <tbody className="divide-y divide-slate-800/60 bg-slate-900/30">{children}</tbody>,
                tr: ({ children }) => <tr className="hover:bg-slate-800/40 transition-colors">{children}</tr>,
                th: ({ children }) => <th className="px-3.5 py-2.5 font-medium text-slate-200 text-[11px]">{children}</th>,
                td: ({ children }) => {
                  if (typeof children === 'string') {
                    return <td className="px-3.5 py-2 text-slate-400 align-top text-[11px]">{renderTextWithCitationPills(children)}</td>;
                  }
                  return <td className="px-3.5 py-2 text-slate-400 align-top text-[11px]">{children}</td>;
                },
                code: ({ children }) => (
                  <code className="bg-slate-800 text-slate-200 font-mono text-[11px] px-1.5 py-0.5 rounded border border-slate-700">
                    {children}
                  </code>
                ),
                strong: ({ children }) => <strong className="font-semibold text-slate-100">{children}</strong>,
              }}
            >
              {processedContent}
            </ReactMarkdown>

            {isStreaming && (
              <span className="inline-block w-1.5 h-3.5 ml-1 bg-slate-400 animate-pulse align-middle" />
            )}
          </div>
        )}

        {/* Sources Shelf Footer */}
        {role === 'assistant' && citations.length > 0 && !isStreaming && (
          <div className="mt-5 pt-3.5 border-t border-slate-800/80">
            <div className="flex items-center justify-between mb-2.5">
              <span className="text-[10px] uppercase tracking-wider font-semibold text-slate-400 flex items-center gap-1.5">
                <BookOpen className="w-3.5 h-3.5 text-slate-400" />
                Referenced Context Passages ({citations.length})
              </span>
              <span className="text-[10px] text-slate-500">Click to view highlighted segment</span>
            </div>

            <div className="flex flex-wrap gap-2">
              {citations.map((c) => (
                <button
                  key={c.chunk_id}
                  onClick={() => onCitationClick?.(c.document_id, c.page_number, c)}
                  className="group inline-flex items-center gap-2 text-xs bg-slate-800/70 hover:bg-slate-800 text-slate-300 hover:text-slate-100 px-3 py-1.5 rounded-lg border border-slate-700/80 hover:border-slate-600 transition-all cursor-pointer"
                >
                  <span className="font-normal truncate max-w-[200px] text-[11px]">{c.filename}</span>
                  <span className="bg-slate-900 group-hover:bg-slate-700 text-slate-400 group-hover:text-slate-200 text-[10px] font-medium px-1.5 py-0.5 rounded border border-slate-700/60">
                    p.{c.page_number}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

