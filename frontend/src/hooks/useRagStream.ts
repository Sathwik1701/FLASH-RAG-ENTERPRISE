import { useState, useRef, useCallback } from 'react';
import type { CitationBadge, StreamState } from '../types';
import { API_BASE_URL } from '../config';

interface UseRagStreamOptions {
  onToken?: (token: string) => void;
  onCitations?: (citations: CitationBadge[]) => void;
  onDone?: () => void;
  onError?: (error: Error) => void;
}

export function useRagStream(options: UseRagStreamOptions = {}) {
  const [state, setState] = useState<StreamState>('idle');
  const [streamedText, setStreamedText] = useState<string>('');
  const [citations, setCitations] = useState<CitationBadge[]>([]);
  const abortControllerRef = useRef<AbortController | null>(null);

  const startStream = useCallback(async (
    query: string,
    chatHistory: Array<{ role: string; content: string }>,
    selectedDocuments: string[]
  ) => {
    // Cancel any ongoing stream
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    setState('loading');
    setStreamedText('');
    setCitations([]);

    try {
      const response = await fetch(`${API_BASE_URL}/api/rag/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          chat_history: chatHistory,
          selected_documents: selectedDocuments,
        }),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}: ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error('ReadableStream not supported or empty response body');
      }

      setState('streaming');

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        // Keep the last partial line in the buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;

          const jsonStr = trimmed.slice(5).trim();
          if (!jsonStr) continue;

          try {
            const event = JSON.parse(jsonStr);

            if (event.type === 'citations') {
              setCitations(event.payload || []);
              options.onCitations?.(event.payload || []);
            } else if (event.type === 'token') {
              setStreamedText((prev) => {
                const updated = prev + event.content;
                options.onToken?.(event.content);
                return updated;
              });
            } else if (event.type === 'done') {
              setState('complete');
              options.onDone?.();
            } else if (event.type === 'error') {
              throw new Error(event.message || 'Stream generation failed');
            }
          } catch (e) {
            console.warn('Failed to parse SSE line:', jsonStr, e);
          }
        }
      }

      setState('complete');
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') {
        console.info('Generation canceled by user.');
        setState('idle');
      } else {
        const error = err instanceof Error ? err : new Error(String(err));
        console.error('SSE Stream error:', error);
        setState('error');
        options.onError?.(error);
      }
    } finally {
      abortControllerRef.current = null;
    }
  }, [options]);

  const cancelStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setState('idle');
    }
  }, []);

  return {
    state,
    streamedText,
    citations,
    startStream,
    cancelStream,
    isStreaming: state === 'streaming' || state === 'loading',
  };
}
