"use client"

import { useState, useEffect } from "react"
import { Download, FileText, Info, List, BarChart3, Loader2, Zap, Database, ImageIcon, Archive } from "lucide-react" // Added missing icons
import { SidebarInset, SidebarTrigger } from "@/components/ui/sidebar"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { useToast } from "@/hooks/use-toast"
import { useAuth } from "@/components/auth-provider" // Import useAuth

// Interfaces matching backend Pydantic models
interface AnalysisStateSummary {
  state_id: string;
  market_domain?: string | null;
  query?: string | null;
  created_at: string; // ISO string
  user_id?: string | null;
  report_filename?: string | null;
  status?: string | null;
}

interface DownloadableFile {
  category: string;
  filename: string;
  description?: string | null;
}

export default function DownloadsPage() {
  const { toast } = useToast();
  const { supabaseClient, user, loading: authLoading, isConfigured } = useAuth(); // Use client from AuthProvider

  const [analysisStates, setAnalysisStates] = useState<AnalysisStateSummary[]>([]);
  const [selectedStateId, setSelectedStateId] = useState<string | null>(null);
  const [selectedStateFiles, setSelectedStateFiles] = useState<DownloadableFile[]>([]);
  const [isLoadingStates, setIsLoadingStates] = useState(true);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);

  const getAuthToken = async () => {
    if (!supabaseClient) return null; // Check if supabaseClient is available
    const { data: { session }, error } = await supabaseClient.auth.getSession(); // Use supabaseClient
    if (error) {
      console.error("Auth Error:", error.message);
      toast({ title: "Authentication Error", description: error.message, variant: "destructive" });
      return null;
    }
    return session?.access_token;
  };

  const fetchAnalysisStates = async () => {
    if (!supabaseClient || !isConfigured) { // Check supabaseClient and configuration
      setIsLoadingStates(false);
      if (!authLoading && !isConfigured) {
         toast({ title: "Configuration Error", description: "Supabase is not configured. Cannot fetch analyses.", variant: "destructive" });
      }
      return;
    }
    setIsLoadingStates(true);
    try {
      const token = await getAuthToken();
      if (!token) {
        // This case should be handled by auth checks before calling, or by getAuthToken itself.
        // If still no token, user might be logged out. AuthProvider should manage redirects.
        toast({ title: "Authentication Error", description: "Session not found. Please log in again.", variant: "destructive" });
        setAnalysisStates([]);
        setIsLoadingStates(false);
        return;
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_PYTHON_AGENT_API_BASE_URL}/analysis-states`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Failed to fetch analysis states."}));
        throw new Error(errorData.detail);
      }
      const data: AnalysisStateSummary[] = await response.json();
      setAnalysisStates(data);
    } catch (error: any) {
      console.error("Fetch Analysis States Error:", error);
      toast({ title: "Error Fetching Analyses", description: error.message, variant: "destructive" });
      setAnalysisStates([]);
    } finally {
      setIsLoadingStates(false);
    }
  };

  const fetchStateDownloadableFiles = async (stateId: string) => {
    if (!stateId || !supabaseClient || !isConfigured) { // Check supabaseClient and configuration
       setIsLoadingFiles(false);
       if (!authLoading && !isConfigured) {
          toast({ title: "Configuration Error", description: "Supabase is not configured. Cannot fetch files.", variant: "destructive" });
       }
      return;
    }
    setSelectedStateId(stateId);
    setIsLoadingFiles(true);
    setSelectedStateFiles([]);
    try {
      const token = await getAuthToken();
      if (!token) {
        toast({ title: "Authentication Error", description: "Session not found. Please log in again.", variant: "destructive" });
        setSelectedStateFiles([]);
        setIsLoadingFiles(false);
        return;
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_PYTHON_AGENT_API_BASE_URL}/analysis-states/${stateId}/downloads`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) {
         const errorData = await response.json().catch(() => ({ detail: "Failed to fetch downloadable files."}));
        throw new Error(errorData.detail);
      }
      const data: { files: DownloadableFile[] } = await response.json();
      setSelectedStateFiles(data.files || []);
    } catch (error: any) {
      console.error("Fetch Downloadable Files Error:", error);
      toast({ title: "Error Fetching Files", description: error.message, variant: "destructive" });
      setSelectedStateFiles([]);
    } finally {
      setIsLoadingFiles(false);
    }
  };

  const handleDownload = async (stateId: string, fileIdentifier: string, filenameToSave: string) => {
    if (!supabaseClient || !isConfigured) { // Check supabaseClient and configuration
      toast({ title: "Configuration Error", description: "Supabase is not configured.", variant: "destructive"});
      return;
    }
    const token = await getAuthToken();
    if (!token) {
      toast({ title: "Authentication Error", description: "Could not get auth token.", variant: "destructive"});
      return;
    }
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_PYTHON_AGENT_API_BASE_URL}/analysis-states/${stateId}/downloads/${encodeURIComponent(fileIdentifier)}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || `Download failed: ${response.statusText}`);
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filenameToSave;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      toast({ title: "Download Started", description: `${filenameToSave} is downloading.`});
    } catch (error: any) {
      toast({ title: "Download Error", description: error.message, variant: "destructive" });
    }
  };

  useEffect(() => {
    // Fetch data only when auth state is settled, Supabase is configured, and client is available
    if (!authLoading && isConfigured && supabaseClient) {
      fetchAnalysisStates();
    } else if (!authLoading && !isConfigured) {
      setIsLoadingStates(false); // Ensure loading stops if not configured
      toast({ title: "Configuration Error", description: "Supabase is not configured. Cannot fetch data.", variant: "destructive" });
    }
    // Not adding fetchAnalysisStates to deps to avoid re-runs if it's stable.
    // If it changes, wrap in useCallback.
  }, [authLoading, isConfigured, supabaseClient, toast]);


  const getFileIcon = (filename: string) => {
    const extension = filename.split('.').pop()?.toLowerCase();
    switch (extension) {
      case "pdf": return <FileText className="w-5 h-5 text-neon-pink" />;
      case "json": return <Database className="w-5 h-5 text-neon-blue" />; // Changed icon for JSON
      case "csv": return <List className="w-5 h-5 text-neon-green" />; // Changed icon for CSV
      case "png": return <ImageIcon className="w-5 h-5 text-neon-purple" />;
      case "md": return <FileText className="w-5 h-5 text-neon-orange" />;
      default: return <Archive className="w-5 h-5 text-gray-400" />;
    }
  };

  // Broader loading state for the page until auth is resolved
  if (authLoading) {
    return (
      <SidebarInset className="bg-dark-bg flex flex-col h-screen">
        <header className="flex h-16 shrink-0 items-center gap-2 border-b border-dark-border bg-dark-card/50 backdrop-blur-sm px-4">
           <SidebarTrigger className="-ml-1 text-white hover:bg-dark-card" />
           <Separator orientation="vertical" className="mr-2 h-4 bg-dark-border" />
           <div className="flex items-center gap-2">
             <Download className="w-5 h-5 text-neon-green" />
             <h1 className="text-lg font-semibold text-white">Downloads</h1>
           </div>
        </header>
        <div className="flex flex-1 items-center justify-center">
          <Loader2 className="w-10 h-10 text-neon-blue animate-spin" />
        </div>
      </SidebarInset>
    );
  }

  if (!isConfigured) {
     return (
      <SidebarInset className="bg-dark-bg flex flex-col h-screen">
        <header className="flex h-16 shrink-0 items-center gap-2 border-b border-dark-border bg-dark-card/50 backdrop-blur-sm px-4">
           <SidebarTrigger className="-ml-1 text-white hover:bg-dark-card" />
           <Separator orientation="vertical" className="mr-2 h-4 bg-dark-border" />
           <div className="flex items-center gap-2">
             <Download className="w-5 h-5 text-neon-green" />
             <h1 className="text-lg font-semibold text-white">Downloads</h1>
           </div>
        </header>
        <div className="flex flex-1 items-center justify-center text-white">
          Supabase is not configured. Please check your environment settings.
        </div>
      </SidebarInset>
    );
  }

  if (!user && !authLoading) { // Should be handled by ProtectedRoute, but as a fallback
    return (
      <SidebarInset className="bg-dark-bg flex flex-col h-screen">
        <header className="flex h-16 shrink-0 items-center gap-2 border-b border-dark-border bg-dark-card/50 backdrop-blur-sm px-4">
           <SidebarTrigger className="-ml-1 text-white hover:bg-dark-card" />
           <Separator orientation="vertical" className="mr-2 h-4 bg-dark-border" />
           <div className="flex items-center gap-2">
             <Download className="w-5 h-5 text-neon-green" />
             <h1 className="text-lg font-semibold text-white">Downloads</h1>
           </div>
        </header>
        <div className="flex flex-1 items-center justify-center text-white">
          Redirecting to login...
        </div>
      </SidebarInset>
    );
  }


  return (
    <SidebarInset className="bg-dark-bg">
      {/* Header */}
      <header className="flex h-16 shrink-0 items-center gap-2 border-b border-dark-border bg-dark-card/50 backdrop-blur-sm px-4">
        <SidebarTrigger className="-ml-1 text-white hover:bg-dark-card" />
        <Separator orientation="vertical" className="mr-2 h-4 bg-dark-border" />
        <div className="flex items-center gap-2">
          <Download className="w-5 h-5 text-neon-green" />
          <h1 className="text-lg font-semibold text-white">Downloads</h1>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex flex-1 flex-col gap-6 p-6">
        <Card className="bg-dark-card border-dark-border">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <List className="w-5 h-5 text-neon-blue" />
              Past Analyses
            </CardTitle>
            <CardDescription className="text-gray-400">
              Select an analysis to view and download its generated files.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoadingStates ? (
              <div className="flex items-center justify-center p-8">
                <Loader2 className="w-8 h-8 text-neon-blue animate-spin" />
                <p className="ml-2 text-white">Loading analyses...</p>
              </div>
            ) : analysisStates.length === 0 ? (
              <p className="text-gray-400 text-center p-8">No past analyses found.</p>
            ) : (
              <ul className="space-y-3 max-h-96 overflow-y-auto">
                {analysisStates.map((state) => (
                  <li key={state.state_id}>
                    <Button
                      variant="ghost"
                      className={`w-full justify-start p-3 text-left h-auto hover:bg-dark-bg/70 ${selectedStateId === state.state_id ? 'bg-neon-blue/20 text-neon-blue' : 'text-white'}`}
                      onClick={() => fetchStateDownloadableFiles(state.state_id)}
                      disabled={!supabaseClient || !isConfigured} // Disable if client not ready
                    >
                      <div className="flex flex-col">
                        <span className="font-medium">Query: {state.query || "N/A"}</span>
                        <span className="text-xs text-gray-400">
                          Domain: {state.market_domain || "N/A"} | Created: {new Date(state.created_at).toLocaleString()}
                        </span>
                        <span className="text-xs text-gray-500">ID: {state.state_id}</span>
                      </div>
                    </Button>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>

        {selectedStateId && (
          <Card className="bg-dark-card border-dark-border mt-6">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <FileText className="w-5 h-5 text-neon-green" />
                Downloadable Files for Analysis <span className="text-neon-green truncate max-w-xs">{selectedStateId}</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {isLoadingFiles ? (
                 <div className="flex items-center justify-center p-8">
                  <Loader2 className="w-8 h-8 text-neon-green animate-spin" />
                  <p className="ml-2 text-white">Loading files...</p>
                </div>
              ) : selectedStateFiles.length === 0 ? (
                <p className="text-gray-400 text-center p-8">No downloadable files found for this analysis.</p>
              ) : (
                <div className="space-y-3">
                  {selectedStateFiles.map((file) => (
                    <div key={file.category + '_' + file.filename} className="flex items-center justify-between p-3 border border-dark-border rounded-lg bg-dark-bg/30 hover:bg-dark-bg/50">
                      <div className="flex items-center gap-3 flex-grow min-w-0">
                        {getFileIcon(file.filename)}
                        <div className="flex-grow min-w-0">
                          <p className="text-white font-medium truncate" title={file.filename}>{file.filename}</p>
                          <p className="text-xs text-gray-400 truncate" title={file.description || undefined}>{file.description || file.category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</p>
                        </div>
                      </div>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleDownload(selectedStateId, file.filename, file.filename)}
                        className="border-neon-green/50 text-neon-green hover:bg-neon-green/10 ml-4"
                        disabled={!supabaseClient || !isConfigured} // Disable if client not ready
                      >
                        <Download className="w-3 h-3 mr-1.5" />
                        Download
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {!selectedStateId && !isLoadingStates && analysisStates.length > 0 && (
            <div className="text-center text-gray-500 p-8">
                <Info className="w-6 h-6 mx-auto mb-2" />
                Select an analysis from the list above to see its downloadable files.
            </div>
        )}

      </div>
    </SidebarInset>
  )
}
