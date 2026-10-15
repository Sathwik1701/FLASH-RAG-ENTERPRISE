export interface CitationBadge {
  citation_id: number;
  chunk_id: string;
  document_id: string;
  filename: string;
  page_number: number;
  section_header?: string | null;
  preview_text: string;
  relevance_score: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: CitationBadge[];
  timestamp: Date;
}

export interface DocumentInfo {
  document_id: string;
  filename: string;
  file_hash: string;
  total_pages: number;
  total_chunks: number;
  uploaded_at?: string;
}

export interface IngestResponse {
  status: string;
  document_id: string;
  filename: string;
  file_hash: string;
  total_pages: number;
  total_chunks: number;
  message: string;
  is_duplicate: boolean;
}

export type StreamState = 'idle' | 'loading' | 'streaming' | 'complete' | 'error';
