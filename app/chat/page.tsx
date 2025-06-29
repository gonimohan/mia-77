"use client";
import React, { useState, useEffect, useRef, FormEvent } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import { Send, MessageSquare, Loader2, Upload, File, Brain, Sparkles } from 'lucide-react';
import { SidebarInset, SidebarTrigger } from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { FileUploadManager } from "@/components/file-upload-manager";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'error';
  text: string;
  timestamp: string;
  context?: any;
}

interface FileContext {
  file_id: string;
  filename: string;
  file_type: string;
  summary?: string;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [fileContext, setFileContext] = useState<FileContext[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
  const { toast } = useToast();
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  useEffect(() => {
    let storedSessionId = localStorage.getItem('chatSessionId');
    if (!storedSessionId) {
      storedSessionId = `session_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
      localStorage.setItem('chatSessionId', storedSessionId);
    }
    setSessionId(storedSessionId);
    
    // Load chat history from localStorage
    const savedMessages = localStorage.getItem(`chat_messages_${storedSessionId}`);
    if (savedMessages) {
      try {
        setMessages(JSON.parse(savedMessages));
      } catch (error) {
        console.error('Failed to load chat history:', error);
      }
    }
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Save messages to localStorage whenever messages change
  useEffect(() => {
    if (sessionId && messages.length > 0) {
      localStorage.setItem(`chat_messages_${sessionId}`, JSON.stringify(messages));
    }
  }, [messages, sessionId]);

  const handleFileAnalyzed = (fileData: any) => {
    const newFileContext: FileContext = {
      file_id: fileData.file_id,
      filename: fileData.filename,
      file_type: fileData.file_type,
      summary: `Uploaded ${fileData.file_type} file with ${fileData.processed_data?.rows || 'unknown'} records`
    };
    
    setFileContext(prev => [...prev, newFileContext]);
    
    // Auto-add to selected files for context
    setSelectedFiles(prev => new Set([...prev, fileData.file_id]));
    
    toast({
      title: "File Available for Chat",
      description: `${fileData.filename} is now available in chat context.`,
    });
  };

  const toggleFileSelection = (fileId: string) => {
    setSelectedFiles(prev => {
      const newSet = new Set(prev);
      if (newSet.has(fileId)) {
        newSet.delete(fileId);
      } else {
        newSet.add(fileId);
      }
      return newSet;
    });
  };

  const handleSendMessage = async (e?: FormEvent) => {
    if (e) e.preventDefault();
    if (!currentMessage.trim() || isLoading || !sessionId) return;

    const userMessageText = currentMessage;
    const timestamp = new Date().toISOString();
    
    // Add user message to UI optimistically
    const userMessage: ChatMessage = {
      role: 'user',
      text: userMessageText,
      id: Date.now().toString(),
      timestamp
    };
    
    setMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');
    setIsLoading(true);

    try {
      // Prepare enhanced context with file information
      const contextFiles = fileContext.filter(file => selectedFiles.has(file.file_id));
      
      const messagesForApi = messages.filter(msg => msg.role === 'user' || msg.role === 'assistant').map(msg => ({
        role: msg.role,
        content: msg.text
      }));

      messagesForApi.push({ role: 'user', content: userMessageText });

      const payload = {
        messages: messagesForApi,
        context: {
          session_id: sessionId,
          files: contextFiles,
          has_file_context: contextFiles.length > 0,
          rag_enabled: true
        }
      };

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const result = await response.json();

      if (!response.ok || result.error || (result.success !== undefined && !result.success)) {
        const errorMsg = result.error || (result.data ? result.data.response : 'Failed to get response from AI.');
        throw new Error(errorMsg);
      }

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        text: result.response,
        id: Date.now().toString() + 'ai',
        timestamp: new Date().toISOString(),
        context: result.context
      };

      setMessages(prev => [...prev, assistantMessage]);

      // Show suggestions if available
      if (result.suggestions && result.suggestions.length > 0) {
        // Could implement suggestion pills here
      }

    } catch (error: any) {
      const errorMessage: ChatMessage = {
        role: 'error',
        text: `Error: ${error.message}`,
        id: Date.now().toString() + 'err',
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, errorMessage]);
      toast({ title: "Chat Error", description: error.message, variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    if (confirm('Are you sure you want to clear the chat history?')) {
      setMessages([]);
      if (sessionId) {
        localStorage.removeItem(`chat_messages_${sessionId}`);
      }
    }
  };

  const formatMessageText = (text: string) => {
    // Basic markdown-like formatting
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code class="bg-gray-800 px-1 rounded">$1</code>');
  };

  return (
    <SidebarInset className="bg-dark-bg flex flex-col h-screen">
      <header className="flex h-16 shrink-0 items-center gap-2 border-b border-dark-border bg-dark-card/50 backdrop-blur-sm px-4">
        <SidebarTrigger className="-ml-1 text-white hover:bg-dark-card" />
        <Separator orientation="vertical" className="mr-2 h-4 bg-dark-border" />
        <div className="flex items-center gap-2">
          <MessageSquare className="w-5 h-5 text-neon-blue" />
          <h1 className="text-lg font-semibold text-white">RAG-Powered Chat</h1>
          <Badge className="bg-neon-purple/20 text-neon-purple border-neon-purple/50">
            <Brain className="w-3 h-3 mr-1" />
            AI Enhanced
          </Badge>
        </div>
        <div className="ml-auto flex items-center gap-2">
          {fileContext.length > 0 && (
            <Badge className="bg-neon-green/20 text-neon-green border-neon-green/50">
              <File className="w-3 h-3 mr-1" />
              {selectedFiles.size} of {fileContext.length} files
            </Badge>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={clearChat}
            className="border-gray-600 text-gray-400 hover:bg-dark-bg"
          >
            Clear Chat
          </Button>
        </div>
      </header>

      <div className="flex-1 flex">
        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* File Context Bar */}
          {fileContext.length > 0 && (
            <div className="p-3 border-b border-dark-border bg-dark-card/30">
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="w-4 h-4 text-neon-blue" />
                <span className="text-sm font-medium text-white">File Context</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {fileContext.map((file) => (
                  <Badge
                    key={file.file_id}
                    variant="outline"
                    className={`cursor-pointer transition-colors ${
                      selectedFiles.has(file.file_id)
                        ? 'bg-neon-blue/20 text-neon-blue border-neon-blue/50'
                        : 'bg-gray-500/20 text-gray-400 border-gray-500/50 hover:bg-gray-500/30'
                    }`}
                    onClick={() => toggleFileSelection(file.file_id)}
                  >
                    <File className="w-3 h-3 mr-1" />
                    {file.filename}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-dark-bg">
            {messages.length === 0 && (
              <div className="text-center py-12">
                <MessageSquare className="w-16 h-16 text-gray-500 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">Welcome to RAG-Powered Chat</h3>
                <p className="text-gray-400 mb-4">
                  Upload files and ask questions about your data for AI-powered insights.
                </p>
                <div className="flex flex-wrap justify-center gap-2 text-sm text-gray-500">
                  <span>• Upload CSV/Excel for data analysis</span>
                  <span>• Ask questions about market trends</span>
                  <span>• Get AI-powered insights</span>
                </div>
              </div>
            )}
            
            {messages.map(msg => (
              <Card 
                key={msg.id} 
                className={`max-w-4xl p-4 rounded-lg ${
                  msg.role === 'user' 
                    ? 'ml-auto bg-neon-blue/20 border-neon-blue/50 text-white' :
                  msg.role === 'assistant' 
                    ? 'mr-auto bg-dark-card border-dark-border text-white' :
                    'mr-auto bg-red-500/20 border-red-500/50 text-red-200'
                }`}
              >
                <div className="flex items-start gap-3">
                  {msg.role === 'assistant' && (
                    <div className="w-8 h-8 rounded-full bg-neon-purple/20 flex items-center justify-center flex-shrink-0">
                      <Brain className="w-4 h-4 text-neon-purple" />
                    </div>
                  )}
                  <div className="flex-1">
                    <div 
                      className="text-white whitespace-pre-wrap"
                      dangerouslySetInnerHTML={{ __html: formatMessageText(msg.text) }}
                    />
                    <div className="text-xs text-gray-400 mt-2 flex items-center gap-2">
                      <span>{new Date(msg.timestamp).toLocaleTimeString()}</span>
                      {msg.context?.confidence && (
                        <Badge variant="outline" className="text-xs">
                          Confidence: {(msg.context.confidence * 100).toFixed(0)}%
                        </Badge>
                      )}
                    </div>
                  </div>
                </div>
              </Card>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Loading Indicator */}
          {isLoading && (
            <div className="p-4 text-center text-gray-400 text-sm flex items-center justify-center border-t border-dark-border">
              <Loader2 className="w-4 h-4 animate-spin mr-2"/>
              <span>AI is analyzing your query{selectedFiles.size > 0 ? ' with file context' : ''}...</span>
            </div>
          )}

          {/* Input Area */}
          <form onSubmit={handleSendMessage} className="p-4 border-t border-dark-border bg-dark-card/50">
            <div className="flex items-center gap-2">
              <Input
                value={currentMessage}
                onChange={e => setCurrentMessage(e.target.value)}
                placeholder={
                  selectedFiles.size > 0 
                    ? `Ask questions about your ${selectedFiles.size} selected file${selectedFiles.size > 1 ? 's' : ''}...`
                    : "Ask me anything about market intelligence or upload files for context..."
                }
                className="flex-1 bg-dark-bg border-dark-border text-white focus:border-neon-blue"
                disabled={isLoading}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
              />
              <Button 
                type="submit" 
                disabled={isLoading || !currentMessage.trim()} 
                className="bg-neon-blue hover:bg-neon-blue/80 text-white"
              >
                <Send className="w-4 h-4" />
              </Button>
            </div>
          </form>
        </div>

        {/* File Upload Sidebar */}
        <div className="w-96 border-l border-dark-border bg-dark-card/30 overflow-y-auto">
          <div className="p-4">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Upload className="w-5 h-5 text-neon-green" />
              File Context
            </h3>
            <FileUploadManager onFileAnalyzed={handleFileAnalyzed} />
          </div>
        </div>
      </div>
    </SidebarInset>
  );
}
