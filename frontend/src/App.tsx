import React, { useState, useRef, useEffect } from 'react';
import { Send, Square, Sparkles, Layers } from 'lucide-react';
import { useRagStream } from './hooks/useRagStream';
import { ChatMessageView } from './components/ChatMessageView';
import { DocumentManager } from './components/DocumentManager';
import { CitationDrawer } from './components/CitationDrawer';
import type { ChatMessage, CitationBadge } from './types';

export function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [activeCitation, setActiveCitation] = useState<CitationBadge | null>(null);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const {
    state: streamState,
    streamedText,
    citations: activeCitations,
    startStream,
    cancelStream,
    isStreaming,
  } = useRagStream({
    onDone: () => {
      // Stream finalized
    },
    onError: (err) => {
      setMessages((prev) => [
        ...prev,
        {
          id: String(Date.now()),
          role: 'assistant',
          content: `⚠️ Error: ${err.message}`,
          timestamp: new Date(),
        },
      ]);
    },
  });

  // When streamed response finishes, persist it to messages list
  useEffect(() => {
    if (streamState === 'complete' && streamedText) {
      setMessages((prev) => [
        ...prev,
        {
          id: String(Date.now()),
          role: 'assistant',
          content: streamedText,
          citations: activeCitations,
          timestamp: new Date(),
        },
      ]);
    }
  }, [streamState]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || isStreaming) return;

    const userMessage: ChatMessage = {
      id: String(Date.now()),
      role: 'user',
      content: inputText.trim(),
      timestamp: new Date(),
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    const query = inputText.trim();
    setInputText('');

    // Format chat history for LangGraph Intent Analyzer
    const chatHistory = newMessages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    await startStream(query, chatHistory, selectedDocIds);
  };

  const handleCitationClick = (_docId: string, _page: number, citation?: CitationBadge) => {
    if (citation) {
      setActiveCitation(citation);
    }
  };

  return (
    <div className="flex h-screen w-screen bg-slate-950 font-sans text-slate-100 antialiased overflow-hidden">
      {/* Sidebar: Document Staging & Ingestion */}
      <DocumentManager
        selectedDocIds={selectedDocIds}
        onSelectionChange={setSelectedDocIds}
      />

      {/* Main Orchestrated Chat Workspace */}
      <div className="flex-1 flex flex-col h-full bg-slate-950">
        {/* Navigation / Header */}
        <header className="h-14 border-b border-slate-800/80 bg-slate-900/90 backdrop-blur-md px-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg bg-slate-800 border border-slate-700/80 flex items-center justify-center text-slate-200">
              <Layers className="w-3.5 h-3.5" />
            </div>
            <div>
              <h1 className="text-xs font-semibold tracking-wide text-slate-100">Enterprise RAG Engine</h1>
              <p className="text-[10px] text-slate-400">
                Hybrid Dense + SPLADE • FlashRank Cross-Encoder • Grounded Attribution
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className="px-2.5 py-1 rounded-md bg-emerald-950/60 text-emerald-400 border border-emerald-800/50 font-medium text-[11px] flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              Connected
            </span>
          </div>
        </header>

        {/* Chat Stream History */}
        <div className="flex-1 overflow-y-auto px-8 py-6 space-y-6">
          {messages.length === 0 && !isStreaming && (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto text-slate-500 space-y-3">
              <div className="w-10 h-10 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-400">
                <Sparkles className="w-5 h-5 text-slate-400" />
              </div>
              <h3 className="font-medium text-slate-200 text-sm">Document Intelligence Ready</h3>
              <p className="text-xs leading-relaxed text-slate-400 max-w-xs">
                Select or upload PDF documents from the left to inquire, extract specifications, or synthesize comparisons.
              </p>
            </div>
          )}

          {/* Historical turns */}
          {messages.map((msg) => (
            <ChatMessageView
              key={msg.id}
              role={msg.role}
              content={msg.content}
              citations={msg.citations}
              onCitationClick={handleCitationClick}
            />
          ))}

          {/* Active Live Streaming Turn */}
          {isStreaming && (
            <ChatMessageView
              role="assistant"
              content={streamedText || 'Retrieving relevant passages and synthesizing response...'}
              citations={activeCitations}
              isStreaming={true}
              onCitationClick={handleCitationClick}
            />
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Query Input Form */}
        <div className="p-4 bg-slate-900/90 border-t border-slate-800/80">
          <form onSubmit={handleSend} className="max-w-4xl mx-auto flex items-center gap-2.5">
            <div className="flex-1 relative">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder={
                  selectedDocIds.length > 0
                    ? `Inquire across ${selectedDocIds.length} active document(s)...`
                    : 'Inquire across all indexed documents...'
                }
                className="w-full px-4 py-3 text-xs bg-slate-800/90 border border-slate-700/80 rounded-xl focus:outline-none focus:border-slate-500 text-slate-100 placeholder-slate-500 transition-colors"
              />
            </div>

            {isStreaming ? (
              <button
                type="button"
                onClick={cancelStream}
                className="px-4 py-3 bg-red-950/80 hover:bg-red-900 border border-red-800 text-red-300 rounded-xl text-xs font-medium flex items-center gap-1.5 transition-colors cursor-pointer"
              >
                <Square className="w-3.5 h-3.5" />
                Stop
              </button>
            ) : (
              <button
                type="submit"
                disabled={!inputText.trim()}
                className="px-5 py-3 bg-slate-100 hover:bg-white disabled:opacity-30 disabled:hover:bg-slate-100 text-slate-950 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer shadow-xs"
              >
                <Send className="w-3.5 h-3.5 text-slate-900" />
                Send
              </button>
            )}
          </form>
        </div>
      </div>

      {/* Citation Preview Drawer */}
      <CitationDrawer
        citation={activeCitation}
        onClose={() => setActiveCitation(null)}
      />
    </div>
  );
}

export default App;
