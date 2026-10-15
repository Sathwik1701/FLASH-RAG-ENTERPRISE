import React, { useState, useEffect } from 'react';
import { UploadCloud, CheckCircle2, FileText, RefreshCw } from 'lucide-react';
import type { DocumentInfo, IngestResponse } from '../types';
import { API_BASE_URL } from '../config';

interface DocumentManagerProps {
  selectedDocIds: string[];
  onSelectionChange: (selectedIds: string[]) => void;
}

export const DocumentManager: React.FC<DocumentManagerProps> = ({
  selectedDocIds,
  onSelectionChange,
}) => {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data);
      }
    } catch (e) {
      console.error('Failed to fetch documents:', e);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadStatus(`Processing "${file.name}" through Ingestion Pipeline...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE_URL}/api/documents/upload`, {
        method: 'POST',
        body: formData,
      });

      const data: IngestResponse = await res.json();
      if (res.ok) {
        if (data.is_duplicate) {
          setUploadStatus(`Duplicate Document detected: "${data.filename}" (SHA-256 match).`);
        } else {
          setUploadStatus(`Indexed "${data.filename}": ${data.total_pages} pages, ${data.total_chunks} chunks.`);
        }
        await fetchDocuments();
        // Automatically select the uploaded document
        if (!selectedDocIds.includes(data.document_id)) {
          onSelectionChange([...selectedDocIds, data.document_id]);
        }
      } else {
        setUploadStatus(`Upload failed: ${data.message || 'Unknown error'}`);
      }
    } catch (err) {
      setUploadStatus(`Error: ${err instanceof Error ? err.message : 'Upload failed'}`);
    } finally {
      setIsUploading(false);
      e.target.value = '';
    }
  };

  const toggleDocument = (id: string) => {
    if (selectedDocIds.includes(id)) {
      onSelectionChange(selectedDocIds.filter((d) => d !== id));
    } else {
      onSelectionChange([...selectedDocIds, id]);
    }
  };

  const toggleSelectAll = () => {
    if (selectedDocIds.length === documents.length) {
      onSelectionChange([]);
    } else {
      onSelectionChange(documents.map((d) => d.document_id));
    }
  };

  return (
    <div className="bg-slate-900 border-r border-slate-800/80 w-80 h-full flex flex-col p-4 select-none text-slate-200">
      <div className="flex items-center justify-between pb-3.5 border-b border-slate-800">
        <div>
          <h2 className="font-semibold text-slate-100 text-sm tracking-tight">Document Repository</h2>
          <p className="text-[11px] text-slate-400">PDF Ingestion & Staging</p>
        </div>
        <button
          onClick={fetchDocuments}
          className="text-slate-400 hover:text-slate-200 transition-colors cursor-pointer p-1.5 rounded-lg hover:bg-slate-800"
          title="Refresh document list"
        >
          <RefreshCw className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Upload Drop Area */}
      <div className="mt-4">
        <label className="border border-dashed border-slate-700/80 hover:border-slate-500 rounded-xl p-4 flex flex-col items-center justify-center cursor-pointer transition-all bg-slate-800/40 hover:bg-slate-800/80 group">
          <UploadCloud className="w-5 h-5 text-slate-400 group-hover:text-slate-200 mb-1.5 transition-colors" />
          <span className="text-xs font-medium text-slate-200">Upload PDF Document</span>
          <span className="text-[10px] text-slate-500 mt-0.5">Layout & SPLADE indexed</span>
          <input
            type="file"
            accept=".pdf,application/pdf"
            onChange={handleFileUpload}
            disabled={isUploading}
            className="hidden"
          />
        </label>
      </div>

      {/* Ingestion Pipeline Feedback */}
      {uploadStatus && (
        <div className="mt-3 p-2.5 rounded-lg bg-slate-800/90 border border-slate-700 text-slate-300 text-xs flex items-start gap-2 animate-fadeIn">
          {isUploading ? (
            <RefreshCw className="w-3.5 h-3.5 text-blue-400 animate-spin mt-0.5 shrink-0" />
          ) : (
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 mt-0.5 shrink-0" />
          )}
          <span className="leading-tight text-[11px] text-slate-300">{uploadStatus}</span>
        </div>
      )}

      {/* Document Selection List */}
      <div className="mt-4 flex-1 overflow-y-auto pr-1">
        <div className="flex items-center justify-between mb-2.5 px-0.5">
          <span className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
            Indexed Files ({documents.length})
          </span>
          {documents.length > 0 && (
            <button
              onClick={toggleSelectAll}
              className="text-[11px] text-slate-400 hover:text-slate-200 cursor-pointer font-medium transition-colors"
            >
              {selectedDocIds.length === documents.length ? 'Deselect All' : 'Select All'}
            </button>
          )}
        </div>

        {documents.length === 0 ? (
          <div className="text-center py-10 text-xs text-slate-500">
            No documents indexed yet. Upload a PDF to begin.
          </div>
        ) : (
          <div className="space-y-1.5">
            {documents.map((doc) => {
              const isSelected = selectedDocIds.includes(doc.document_id);
              return (
                <div
                  key={doc.document_id}
                  onClick={() => toggleDocument(doc.document_id)}
                  className={`p-2.5 rounded-lg border text-xs flex items-center gap-2.5 cursor-pointer transition-all ${
                    isSelected
                      ? 'bg-slate-800 border-slate-600 text-slate-100 shadow-xs'
                      : 'bg-slate-900/60 border-slate-800/80 text-slate-400 hover:bg-slate-800/50 hover:text-slate-300'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => {}}
                    className="rounded border-slate-700 bg-slate-800 text-blue-500 focus:ring-0 cursor-pointer"
                  />
                  <FileText className={`w-4 h-4 shrink-0 ${isSelected ? 'text-blue-400' : 'text-slate-500'}`} />
                  <div className="flex-1 min-w-0">
                    <p className="font-medium truncate text-xs">{doc.filename}</p>
                    <p className="text-[10px] text-slate-500">
                      {doc.total_pages} pgs • {doc.total_chunks} chunks
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="pt-3 border-t border-slate-800 text-[11px] text-slate-500 text-center">
        {selectedDocIds.length === 0
          ? 'Searching across entire index'
          : `Filtering to ${selectedDocIds.length} document(s)`}
      </div>
    </div>
  );
};
