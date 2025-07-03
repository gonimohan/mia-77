"use client"

import { useState, useEffect } from "react"
import { Download, FileText, Info, List, BarChart3, Loader2, Zap, Database, ImageIcon, Archive } from "lucide-react" // Added missing icons
import { SidebarInset, SidebarTrigger } from "@/components/ui/sidebar"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { useToast } from "@/hooks/use-toast"
import { useAuth } from "@/components/auth-provider" // Import useAuth

// Interface for report items listed from /api/reports
interface ReportListItem {
  id: string; // This is the UUID of the report record in Supabase 'reports' table
  title: string;
  market_domain?: string | null;
  status?: string | null;
  created_at: string; // ISO string
  state_id?: string | null; // This is the agent's state_id, crucial for fetching files
  // query_text might also be useful to display, if returned by /api/reports
}

interface DownloadableFile {
  category: string;
  filename: string;
  description?: string | null;
}

export default function DownloadsPage() {
  const { toast } = useToast();
  const { supabaseClient, user, loading: authLoading, isConfigured } = useAuth();

  const [reportsList, setReportsList] = useState<ReportListItem[]>([]); // Changed state variable name
  const [selectedReport, setSelectedReport] = useState<ReportListItem | null>(null); // Store the whole report object
  const [selectedStateFiles, setSelectedStateFiles] = useState<DownloadableFile[]>([]);
  const [isLoadingReports, setIsLoadingReports] = useState(true); // Changed state variable name
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

  const fetchReportsList = async () => { // Renamed function
    if (!isConfigured) { // Simpler check, authProvider handles session
      setIsLoadingReports(false);
      if (!authLoading) { // Avoid toast if auth is still loading
         toast({ title: "Configuration Error", description: "Service is not configured. Cannot fetch reports.", variant: "destructive" });
      }
      return;
    }
    setIsLoadingReports(true);
    try {
      // Calls Next.js API route /api/reports which handles auth
      const response = await fetch(`/api/reports`);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Failed to fetch reports list."}));
        throw new Error(errorData.detail);
      }
      const data: ReportListItem[] = await response.json();
      setReportsList(data);
    } catch (error: any) {
      console.error("Fetch Reports List Error:", error);
      toast({ title: "Error Fetching Reports", description: error.message, variant: "destructive" });
      setReportsList([]);
    } finally {
      setIsLoadingReports(false);
    }
  };

  const fetchStateDownloadableFiles = async (report: ReportListItem) => { // Takes ReportListItem
    if (!report || !report.state_id ||!isConfigured) {
       setIsLoadingFiles(false);
       if (!report || !report.state_id) {
            toast({ title: "Info", description: "This report has no associated analysis state or files.", variant: "default" });
       } else if (!authLoading && !isConfigured) {
          toast({ title: "Configuration Error", description: "Service is not configured. Cannot fetch files.", variant: "destructive" });
       }
      setSelectedReport(report); // Still set selected report to show its details even if no files
      setSelectedStateFiles([]);
      return;
    }
    setSelectedReport(report);
    setIsLoadingFiles(true);
    setSelectedStateFiles([]); // Clear previous files
    try {
      // Calls Next.js API route, auth handled there
      const response = await fetch(`/api/analysis-states/${report.state_id}/downloads-info`);
      if (!response.ok) {
         const errorData = await response.json().catch(() => ({ detail: "Failed to fetch downloadable files for this report."}));
        throw new Error(errorData.detail);
      }
      const data: { files: DownloadableFile[] } = await response.json(); // Assuming Python returns {"files": [...]}
      setSelectedStateFiles(data.files || []);
    } catch (error: any) {
      console.error("Fetch Downloadable Files Error:", error);
      toast({ title: "Error Fetching Report Files", description: error.message, variant: "destructive" });
      setSelectedStateFiles([]);
    } finally {
      setIsLoadingFiles(false);
    }
  };

  const handleDownload = async (stateId: string | undefined, fileIdentifier: string, filenameToSave: string) => {
    if (!stateId) {
      toast({ title: "Error", description: "Analysis state ID is missing for download.", variant: "destructive"});
      return;
    }
    if (!isConfigured) {
      toast({ title: "Configuration Error", description: "Service is not configured.", variant: "destructive"});
      return;
    }
    // Auth is handled by the Next.js API route
    try {
      // Calls Next.js API route
      const response = await fetch(`/api/analysis-states/${stateId}/download-file/${encodeURIComponent(fileIdentifier)}`);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Download request failed." })); // Try to parse error from Next.js route
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
    if (!authLoading && isConfigured) { // Removed supabaseClient from direct check, isConfigured implies it's ready via useAuth
      fetchReportsList(); // Call renamed function
    } else if (!authLoading && !isConfigured) {
      setIsLoadingReports(false);
      toast({ title: "Configuration Error", description: "Service is not configured. Cannot fetch reports.", variant: "destructive" });
    }
  }, [authLoading, isConfigured, toast]); // Removed supabaseClient from deps, fetchReportsList uses isConfigured


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
              Generated Reports
            </CardTitle>
            <CardDescription className="text-gray-400">
              Select a report to view and download its generated files.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoadingReports ? (
              <div className="flex items-center justify-center p-8">
                <Loader2 className="w-8 h-8 text-neon-blue animate-spin" />
                <p className="ml-2 text-white">Loading reports...</p>
              </div>
            ) : reportsList.length === 0 ? (
              <p className="text-gray-400 text-center p-8">No past reports found.</p>
            ) : (
              <ul className="space-y-3 max-h-96 overflow-y-auto">
                {reportsList.map((report) => (
                  <li key={report.id}> {/* Use report.id (from Supabase 'reports' table) as key */}
                    <Button
                      variant="ghost"
                      className={`w-full justify-start p-3 text-left h-auto hover:bg-dark-bg/70 ${selectedReport?.id === report.id ? 'bg-neon-blue/20 text-neon-blue' : 'text-white'}`}
                      onClick={() => fetchStateDownloadableFiles(report)}
                      disabled={!isConfigured}
                    >
                      <div className="flex flex-col">
                        <span className="font-medium truncate" title={report.title}>{report.title}</span>
                        <span className="text-xs text-gray-400">
                          Domain: {report.market_domain || "N/A"} | Status: <span className={report.status === 'completed' ? 'text-neon-green' : 'text-neon-orange'}>{report.status || "Unknown"}</span>
                        </span>
                        <span className="text-xs text-gray-500">
                          Created: {new Date(report.created_at).toLocaleString()} | Report ID: {report.id.substring(0,8)}...
                          {report.state_id && ` (State: ${report.state_id.substring(0,8)}...)`}
                        </span>
                      </div>
                    </Button>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>

        {selectedReport && ( // Check selectedReport instead of selectedStateId
          <Card className="bg-dark-card border-dark-border mt-6">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <FileText className="w-5 h-5 text-neon-green" />
                Downloadable Files for Report: <span className="text-neon-green truncate max-w-xs" title={selectedReport.title}>{selectedReport.title}</span>
              </CardTitle>
               <CardDescription className="text-gray-400">
                 Report ID: {selectedReport.id} | Agent State ID: {selectedReport.state_id || "N/A"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoadingFiles ? (
                 <div className="flex items-center justify-center p-8">
                  <Loader2 className="w-8 h-8 text-neon-green animate-spin" />
                  <p className="ml-2 text-white">Loading files...</p>
                </div>
              ) : selectedStateFiles.length === 0 ? (
                <p className="text-gray-400 text-center p-8">
                  {selectedReport.state_id ? "No downloadable files found for this report's analysis state." : "This report has no associated analysis state to fetch files from."}
                </p>
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
                        // Use selectedReport.state_id for the download call
                        onClick={() => handleDownload(selectedReport.state_id, file.filename, file.filename)}
                        className="border-neon-green/50 text-neon-green hover:bg-neon-green/10 ml-4"
                        disabled={!isConfigured || !selectedReport.state_id}
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

        {!selectedReport && !isLoadingReports && reportsList.length > 0 && ( // Check selectedReport
            <div className="text-center text-gray-500 p-8">
                <Info className="w-6 h-6 mx-auto mb-2" />
                Select a report from the list above to see its downloadable files.
            </div>
        )}

      </div>
    </SidebarInset>
  )
}
