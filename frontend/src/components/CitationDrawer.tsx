import React from 'react';
import { X, BookOpen } from 'lucide-react';
import type { CitationBadge } from '../types';

interface CitationDrawerProps {
  citation: CitationBadge | null;
  onClose: () => void;
}

export const CitationDrawer: React.FC<CitationDrawerProps> = ({ citation, onClose }) => {
  if (!citation) return null;

  return (
    <div className="fixed inset-y-0 right-0 w-96 bg-slate-900 shadow-2xl border-l border-slate-800 z-50 flex flex-col p-6 animate-slideLeft text-slate-200">
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-slate-400" />
          <h3 className="font-semibold text-slate-100 text-xs tracking-tight">Source Attribution</h3>
        </div>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-slate-200 p-1 rounded-md hover:bg-slate-800 transition-colors cursor-pointer"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="mt-4 space-y-4 flex-1 overflow-y-auto pr-1">
        <div className="bg-slate-800/60 p-3 rounded-lg border border-slate-700/80 text-xs space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-slate-400 text-[11px]">Document:</span>
            <span className="font-medium text-slate-200 truncate max-w-[200px] text-[11px]" title={citation.filename}>
              {citation.filename}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-400 text-[11px]">Page:</span>
            <span className="font-medium text-slate-200 text-[11px]">Page {citation.page_number}</span>
          </div>
          {citation.section_header && (
            <div className="flex items-center justify-between">
              <span className="text-slate-400 text-[11px]">Section:</span>
              <span className="font-medium text-slate-200 text-[11px]">{citation.section_header}</span>
            </div>
          )}
          <div className="flex items-center justify-between">
            <span className="text-slate-400 text-[11px]">Reranker Match:</span>
            <span className="font-medium text-emerald-400 text-[11px]">
              {(citation.relevance_score * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        <div>
          <h4 className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Retrieved Text Passage:
          </h4>
          <div className="p-3.5 bg-slate-950 border border-slate-800 rounded-xl text-[11px] text-slate-300 leading-relaxed font-mono whitespace-pre-wrap">
            {citation.preview_text}
          </div>
        </div>
      </div>

      <div className="pt-4 border-t border-slate-800">
        <button
          onClick={onClose}
          className="w-full py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg transition-colors cursor-pointer border border-slate-700/80"
        >
          Close Drawer
        </button>
      </div>
    </div>
  );
};
