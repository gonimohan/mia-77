"use client"

import { useState, useEffect } from "react"
import { RefreshCw, Zap, CheckCircle, AlertCircle, Clock, Database } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { useToast } from "@/hooks/use-toast"

// Aligned with backend's SyncStatusResponseItem
interface SyncStatusInfo {
  sourceName: string; // Display name (e.g., "News API") - maps to localDataSources.name
  sourceId: string;   // Identifier used for communication & matching (e.g., "NewsAPI") - maps to localDataSources.id
  status: "syncing" | "completed" | "synced" | "error" | "pending" | "unknown";
  progress: number;
  message?: string;
  last_update: string;
}

// This local dataSources array is kept for this subtask as per Option B.
// IDs here should match what's sent to backend and used as keys in sync_statuses_store.
const localDataSources = [
  { id: "NewsAPI", name: "News API", category: "news" },
  { id: "MediaStack", name: "MediaStack", category: "news" },
  { id: "GNews", name: "GNews", category: "news" },
  { id: "Tavily Search", name: "Tavily Search", category: "search" },
  { id: "SerpAPI", name: "SerpAPI", category: "search" },
  { id: "Alpha Vantage", name: "Alpha Vantage", category: "financial" },
  { id: "Financial Modeling Prep", name: "Financial Modeling Prep", category: "financial" },
  { id: "Google Gemini", name: "Google Gemini", category: "ai" }, // Simplified name for consistency
]

export function RealTimeSync() {
  const [syncStatuses, setSyncStatuses] = useState<SyncStatusInfo[]>([])
  const [isGlobalSyncActive, setIsGlobalSyncActive] = useState(false)
  const { toast } = useToast()

  useEffect(() => {
    // Initialize sync statuses from localDataSources
    const initialStatuses = localDataSources.map((source) => ({
      sourceId: source.id,
      sourceName: source.name,
      status: "pending" as SyncStatusInfo["status"],
      progress: 0,
      last_update: "Never",
      message: "Awaiting sync.",
    }));
    setSyncStatuses(initialStatuses);

    // Check for existing sync status from backend
    fetchSyncStatus();
  }, [])

  // Polling logic
  useEffect(() => {
    let intervalId: NodeJS.Timeout | null = null;
    const anySyncing = syncStatuses.some(s => s.status === 'syncing');

    if (isGlobalSyncActive || anySyncing) { // Poll if global sync active OR any source is individually syncing
      intervalId = setInterval(() => {
        fetchSyncStatus();
      }, 5000); // Poll every 5 seconds
    }

    // Check if all syncing is done after a global sync was triggered
    if (isGlobalSyncActive && !anySyncing && syncStatuses.length > 0) {
       const allSourcesAttemptedOrCompleted = localDataSources.every(lds => {
         const ss = syncStatuses.find(s => s.sourceId === lds.id);
         // Consider a source "attempted" if it's not pending or syncing anymore
         return ss && (ss.status === 'completed' || ss.status === 'synced' || ss.status === 'error');
       });

       if (allSourcesAttemptedOrCompleted) { // If all relevant sources are processed
          setIsGlobalSyncActive(false);
          toast({ title: "Sync Cycle Complete", description: "All requested source synchronizations have finished processing." });
       }
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isGlobalSyncActive, syncStatuses, toast]); // Added toast to dependencies


  const fetchSyncStatus = async () => {
    try {
      const response = await fetch("/api/agent/sync"); // GET request
      if (!response.ok) {
        console.error("Failed to fetch sync status:", response.statusText);
        toast({ title: "Status Update Error", description: `Could not fetch latest statuses: ${response.statusText}`, variant: "destructive" });
        return;
      }
      const data = await response.json(); // Expects AllSyncStatusesResponse

      if (data && data.statuses) {
        const backendStatuses = data.statuses as Array<{source: string, status: string, progress: number, message?: string, last_update: string}>;

        setSyncStatuses(prevStatuses => { // Use functional update to ensure we have latest prevStatuses
          return localDataSources.map(lds => {
            // Backend 'source' is expected to match localDataSources 'id' field
            const backendStatus = backendStatuses.find(bs => bs.source === lds.id);
            const currentLocalStatus = prevStatuses.find(ps => ps.sourceId === lds.id);

            if (backendStatus) {
              return {
                sourceId: lds.id,
                sourceName: lds.name,
                status: backendStatus.status as SyncStatusInfo["status"],
                progress: backendStatus.progress,
                message: backendStatus.message,
                last_update: backendStatus.last_update,
              };
            }
            // If no backend status for this source, keep its existing local status
            // or default to pending if somehow it's not in prevStatuses (should not happen with init)
            return currentLocalStatus || {
              sourceId: lds.id,
              sourceName: lds.name,
              status: "pending",
              progress: 0,
              last_update: "Never",
              message: "Awaiting status from backend.",
            };
          });
        });
      }
    } catch (error: any) {
      console.error("Error in fetchSyncStatus:", error);
      toast({ title: "Status Fetch Error", description: error.message || "Could not retrieve sync statuses.", variant: "destructive" });
    }
  };

  const startGlobalSync = async () => {
    setIsGlobalSyncActive(true);
    // Optimistically set UI to "syncing" for sources being synced
    setSyncStatuses(prev =>
      prev.map(s =>
        localDataSources.some(lds => lds.id === s.sourceId) // Only for sources we are about to sync
          ? {...s, status: 'syncing', progress: 5, message: 'Initiating...'}
          : s
      )
    );

    try {
      const response = await fetch("/api/agent/sync", { // POST request
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sources: localDataSources.map((s) => s.id), // Send source IDs (which are names for backend)
          market_domain: "general_technology", // Example, can be dynamic
          sync_type: "full", // Example, can be dynamic
        }),
      });

      const resultData = await response.json(); // Always try to parse JSON

      if (!response.ok) {
        throw new Error(resultData.detail || "Failed to start sync process.");
      }

      toast({
        title: "Global Sync Initiated",
        description: resultData.message || "Synchronization process has been started for all sources.",
      });

      // Fetch status immediately to get the "syncing" state from backend
      await fetchSyncStatus();
      // Polling useEffect will handle subsequent updates.

    } catch (error: any) {
      toast({
        title: "Global Sync Error",
        description: error.message || "Failed to initiate global data synchronization.",
        variant: "destructive",
      });
      setIsGlobalSyncActive(false);
      // Revert status for items that were optimistically set to "syncing"
      setSyncStatuses(prev =>
        prev.map(s =>
          s.status === 'syncing' && localDataSources.some(lds => lds.id === s.sourceId)
            ? {...s, status: 'error', message: 'Failed to start.', progress: 0}
            : s
        )
      );
    }
  };

  // Removed simulateSyncProgress function

  const getStatusIcon = (status: SyncStatusInfo['status']) => {
    switch (status) {
      case "completed":
      case "synced":
        return <CheckCircle className="w-4 h-4 text-neon-green" />
      case "error":
        return <AlertCircle className="w-4 h-4 text-neon-pink" />
      case "syncing":
        return <RefreshCw className="w-4 h-4 text-neon-blue animate-spin" />
      default: // pending, unknown
        return <Clock className="w-4 h-4 text-gray-400" />
    }
  }

  const getStatusBadge = (status: SyncStatusInfo['status']) => { // Use stricter type
    const variants = {
      completed: "bg-neon-green/20 text-neon-green border-neon-green/50",
      synced: "bg-neon-green/20 text-neon-green border-neon-green/50",
      error: "bg-neon-pink/20 text-neon-pink border-neon-pink/50",
      syncing: "bg-neon-blue/20 text-neon-blue border-neon-blue/50",
      pending: "bg-gray-500/20 text-gray-400 border-gray-500/50",
      unknown: "bg-yellow-500/20 text-yellow-400 border-yellow-500/50", // For unknown status
    };
    const style = variants[status] || variants.unknown; // Fallback to unknown style
    return (
      <Badge className={`${style} border text-xs`}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </Badge>
    );
  }

  const completedSourcesCount = syncStatuses.filter((s) => s.status === "completed" || s.status === "synced").length;
  const totalConfiguredSources = localDataSources.length;
  const overallProgress = totalConfiguredSources > 0 ? (completedSourcesCount / totalConfiguredSources) * 100 : 0;

  const lastGlobalActivityTime = syncStatuses.length > 0
  ? syncStatuses
      .filter(s => s.last_update !== "Never")
      .reduce((latest, current) => {
          const currentUpdate = new Date(current.last_update);
          // Initialize latestDate with a very old date if latest is "Never"
          const latestDate = latest === "Never" ? new Date(0) : new Date(latest);
          return currentUpdate > latestDate ? current.last_update : latest;
        }, "Never")
  : "Never";


  return (
    <Card className="bg-dark-card border-dark-border">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <Database className="w-5 h-5 text-neon-blue" />
          Real-Time Data Synchronization
        </CardTitle>
        <CardDescription className="text-gray-400">
          Monitor and control data collection from all configured sources
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Global Sync Controls */}
        <div className="flex items-center justify-between p-4 border border-dark-border rounded-lg bg-dark-bg/50">
          <div className="space-y-1">
            <h4 className="text-white font-medium">Global Synchronization</h4>
            <p className="text-sm text-gray-400">
              {lastGlobalActivityTime !== "Never" ? `Last activity: ${new Date(lastGlobalActivityTime).toLocaleString()}` : "No recent synchronization"}
            </p>
          </div>
          <Button
            onClick={startGlobalSync}
            disabled={isGlobalSyncActive}
            className="bg-neon-blue/20 border border-neon-blue/50 text-neon-blue hover:bg-neon-blue/30"
          >
            {isGlobalSyncActive ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                Syncing...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 mr-2" />
                Start All Sources
              </>
            )}
          </Button>
        </div>

        {/* Overall Progress - show if any sync has started or is active globally */}
        {(isGlobalSyncActive || syncStatuses.some(s => s.status === 'syncing' || s.status === 'completed' || s.status === 'synced' || s.status === 'error')) && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-white">Overall Progress</span>
              <span className="text-sm text-gray-400">
                {completedSourcesCount}/{totalConfiguredSources} sources completed
              </span>
            </div>
            <Progress value={overallProgress} className="h-2 bg-dark-bg" />
          </div>
        )}

        {/* Individual Source Status */}
        <div className="space-y-3">
          <h4 className="text-white font-medium">Data Sources</h4>
          <div className="grid gap-3">
            {syncStatuses.map((syncStatus) => (
                <div // Changed from Card to div for simpler structure if needed, or keep Card
                  key={syncStatus.sourceId}
                  className="flex items-center justify-between p-3 border border-dark-border rounded-lg bg-dark-bg/30"
                >
                  <div className="flex items-center gap-3">
                    {getStatusIcon(syncStatus.status)}
                    <div className="flex-grow">
                      <h5 className="text-white text-sm font-medium">{syncStatus.sourceName}</h5>
                      <p className="text-xs text-gray-400 truncate max-w-xs" title={syncStatus.message || undefined}>
                        {syncStatus.message || (syncStatus.status === 'completed' || syncStatus.status === 'synced' ? 'Sync successful' : 'No message')}
                      </p>
                       <p className="text-xs text-gray-500">Last update: {syncStatus.last_update !== "Never" ? new Date(syncStatus.last_update).toLocaleString() : "Never"}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 min-w-[120px] justify-end">
                    {syncStatus.status === "syncing" && (
                      <div className="w-20">
                        <Progress value={syncStatus.progress} className="h-1 bg-dark-bg" />
                      </div>
                    )}
                    {getStatusBadge(syncStatus.status)}
                  </div>
                </div>
              )
            )}
          </div>
        </div>

        {/* Sync Statistics */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 pt-4 border-t border-dark-border"> {/* Changed to 3 cols */}
          <div className="text-center">
            <p className="text-2xl font-bold text-neon-green">{completedSourcesCount}</p>
            <p className="text-xs text-gray-400">Completed</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-neon-blue">
              {syncStatuses.filter((s) => s.status === "syncing").length}
            </p>
            <p className="text-xs text-gray-400">Syncing</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-neon-pink">
              {syncStatuses.filter((s) => s.status === "error").length}
            </p>
            <p className="text-xs text-gray-400">Errors</p>
          </div>
          {/* Removed Total Records as this info is not directly available from the new status structure */}
        </div>
      </CardContent>
    </Card>
  )
}
