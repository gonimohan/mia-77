"use client"

import { useState, useEffect } from "react" // Added useEffect
import { TrendingUp, Users, Heart, PieChart, RefreshCw, Zap, Activity, Target, Briefcase, DollarSign } from "lucide-react" // Added DollarSign
import { SidebarInset, SidebarTrigger } from "@/components/ui/sidebar"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { KPICard } from "@/components/kpi-card"
import { Separator } from "@/components/ui/separator"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart as RechartsPieChart,
  Cell,
  Pie,
} from "recharts"
import { useColorPalette } from "@/lib/color-context"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogClose,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useToast } from "@/hooks/use-toast"
import { FeatureTracker } from "@/components/feature-tracker"
import { useAuth } from "@/components/auth-provider"
import { OnboardingFlow } from "@/components/onboarding-flow"
import { WidgetSelectorDialog, DashboardWidgetConfig } from "@/components/widget-selector-dialog"
import { HistoricalKpiChartDialog } from "@/components/historical-kpi-chart-dialog"
import { Settings2 } from "lucide-react"

const defaultWidgets: DashboardWidgetConfig[] = [
  { id: "kpiCards", title: "Key Performance Indicators", defaultEnabled: true },
  { id: "trendImpactChart", title: "Identified Trends Impact", defaultEnabled: true },
  { id: "marketShareChart", title: "Market Share", defaultEnabled: true },
  { id: "competitorActivity", title: "Competitor Activity", defaultEnabled: true },
  { id: "developmentProgress", title: "Development Progress (Dev Only)", defaultEnabled: false },
  { id: "quickActions", title: "Quick Actions", defaultEnabled: true },
];

export default function DashboardPage() {
  const { getChartColors } = useColorPalette()
  const chartColors = getChartColors()
  const { user, supabaseClient, loading: authLoading } = useAuth();
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [kpiData, setKpiData] = useState<any[]>([]) // Initialize with empty array
  const [marketShareDataState, setMarketShareDataState] = useState<any[]>([]);
  const [competitorActivityDataState, setCompetitorActivityDataState] = useState<any[]>([]);
  const [trendsChartDataState, setTrendsChartDataState] = useState<any[]>([]);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [isWidgetSelectorOpen, setIsWidgetSelectorOpen] = useState(false);
  const [enabledWidgets, setEnabledWidgets] = useState<string[]>([]);

  // State for Historical KPI Dialog
  const [isHistoricalKpiDialogOpen, setIsHistoricalKpiDialogOpen] = useState(false);
  const [selectedKpiForHistory, setSelectedKpiForHistory] = useState<{ title: string; unit: string } | null>(null);
  const [historicalKpiData, setHistoricalKpiData] = useState<{ date: string; value: number }[]>([]);

  // State for Analysis Dialog
  const [isAnalysisDialogOpen, setIsAnalysisDialogOpen] = useState(false);
  const [analysisQuery, setAnalysisQuery] = useState("");
  const [analysisMarketDomain, setAnalysisMarketDomain] = useState("");
  const [analysisQuestion, setAnalysisQuestion] = useState("");
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);

  // State for AI Insights Dialog
  const [isAiInsightsDialogOpen, setIsAiInsightsDialogOpen] = useState(false);
  const [aiInsightsQuery, setAiInsightsQuery] = useState("");
  const [aiInsightsMarketDomain, setAiInsightsMarketDomain] = useState("");
  const [aiInsightsSpecificQuestion, setAiInsightsSpecificQuestion] = useState("");
  const [isGeneratingAiInsights, setIsGeneratingAiInsights] = useState(false);

  const { toast } = useToast();

  const kpiIconsList = [TrendingUp, Users, Heart, PieChart, DollarSign, Briefcase, Activity, Target];
  const kpiColorsList = ["blue", "green", "pink", "purple", "orange", "blue", "green", "pink"];

  useEffect(() => {
    if (!authLoading && user) {
      const onboardingComplete = user.user_metadata?.onboarding_complete;
      if (onboardingComplete === false || onboardingComplete === undefined || onboardingComplete === null) {
        setShowOnboarding(true);
      }

      const userWidgets = user.user_metadata?.dashboard_widgets;
      if (Array.isArray(userWidgets)) {
        setEnabledWidgets(userWidgets);
      } else {
        // If no custom settings, enable all default widgets
        setEnabledWidgets(defaultWidgets.filter(w => w.defaultEnabled).map(w => w.id));
      }
    }
  }, [user, authLoading]);

  const handleOnboardingComplete = async () => {
    if (user && supabaseClient) {
      try {
        const { data, error } = await supabaseClient.auth.updateUser({
          data: { onboarding_complete: true }
        });

        if (error) {
          console.error("Error updating user metadata:", error);
          toast({
            title: "Onboarding Error",
            description: "Failed to save onboarding status. Please try again.",
            variant: "destructive",
          });
        } else {
          setShowOnboarding(false);
          toast({
            title: "Onboarding Complete",
            description: "Welcome to your dashboard!",
          });
        }
      } catch (err) {
        console.error("Unexpected error during onboarding completion:", err);
        toast({
          title: "Onboarding Error",
          description: "An unexpected error occurred.",
          variant: "destructive",
        });
      }
    }
  };

  const handleSaveWidgets = async (newEnabledWidgets: string[]) => {
    if (user && supabaseClient) {
      try {
        const { data, error } = await supabaseClient.auth.updateUser({
          data: { dashboard_widgets: newEnabledWidgets }
        });

        if (error) {
          console.error("Error updating dashboard widgets:", error);
          toast({
            title: "Save Failed",
            description: "Failed to save widget preferences. Please try again.",
            variant: "destructive",
          });
        } else {
          setEnabledWidgets(newEnabledWidgets);
          toast({
            title: "Preferences Saved",
            description: "Dashboard layout updated successfully.",
          });
        }
      } catch (err) {
        console.error("Unexpected error during widget save:", err);
        toast({
          title: "Save Error",
          description: "An unexpected error occurred while saving preferences.",
          variant: "destructive",
        });
      }
    }
  };

  const fetchHistoricalKpiData = async (kpiTitle: string) => {
    // This is mock data for now. In a real app, you'd fetch this from your backend.
    // The backend would query your database for historical KPI values.
    const mockData = [
      { date: "2023-01-01", value: 100 },
      { date: "2023-02-01", value: 110 },
      { date: "2023-03-01", value: 105 },
      { date: "2023-04-01", value: 120 },
      { date: "2023-05-01", value: 115 },
      { date: "2023-06-01", value: 130 },
      { date: "2023-07-01", value: 125 },
      { date: "2023-08-01", value: 140 },
      { date: "2023-09-01", value: 135 },
      { date: "2023-10-01", value: 150 },
      { date: "2023-11-01", value: 145 },
      { date: "2023-12-01", value: 160 },
    ];
    setHistoricalKpiData(mockData);
  };

  const handleViewKpiHistory = (kpiTitle: string) => {
    const kpi = kpiData.find(k => k.title === kpiTitle);
    if (kpi) {
      setSelectedKpiForHistory({ title: kpi.title, unit: kpi.unit });
      fetchHistoricalKpiData(kpi.title);
      setIsHistoricalKpiDialogOpen(true);
    }
  };

  const fetchKpiData = async () => {
    try {
      const response = await fetch("/api/kpi");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const responseData = await response.json();

      if (responseData && responseData.data) {
        const transformedKpis = responseData.data.map((kpi: any, index: number) => {
          let assignedIcon = kpiIconsList[index % kpiIconsList.length];
          let assignedColor = kpiColorsList[index % kpiColorsList.length];

          const lowerCaseName = String(kpi.metric_name || "").toLowerCase();
          if (lowerCaseName.includes("growth")) { assignedIcon = TrendingUp; assignedColor = "green"; }
          else if (lowerCaseName.includes("competitor") || lowerCaseName.includes("user")) { assignedIcon = Users; assignedColor = "blue"; }
          else if (lowerCaseName.includes("sentiment") || lowerCaseName.includes("satisfaction")) { assignedIcon = Heart; assignedColor = "pink"; }
          else if (lowerCaseName.includes("share")) { assignedIcon = PieChart; assignedColor = "purple"; }
          else if (lowerCaseName.includes("revenue") || lowerCaseName.includes("profit") || lowerCaseName.includes("gmv") || lowerCaseName.includes("sales")) { assignedIcon = DollarSign; assignedColor = "green"; }

          return {
            title: kpi.metric_name || "Untitled KPI",
            value: parseFloat(kpi.metric_value).toLocaleString(undefined, { maximumFractionDigits: 1, minimumFractionDigits: (parseFloat(kpi.metric_value) % 1 !== 0) ? 1:0 }) || "0",
            unit: kpi.metric_unit || "",
            change: parseFloat(kpi.change_percentage) || 0, // This is still 0 from backend for now
            icon: assignedIcon,
            color: assignedColor as "blue" | "green" | "pink" | "purple" | "orange", // Type assertion
          };
        });
        setKpiData(transformedKpis);
      } else {
        console.warn("No data received from /api/kpi or data format is incorrect");
        setKpiData([]);
      }
    } catch (error) {
      console.error("Failed to fetch KPI data:", error);
      setKpiData([]);
      toast({
        title: "KPI Data Error",
        description: "Failed to load KPI data.",
        variant: "destructive",
      });
      throw error;
    }
  };

  const fetchChartData = async () => {
    try {
      // Calls the Next.js API route, which then calls the Python backend with auth
      const response = await fetch(`/api/competitors`);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: `Failed to fetch competitor data: ${response.status} ${response.statusText}` }));
        throw new Error(errorData.detail || `Failed to fetch competitor data: ${response.status} ${response.statusText}`);
      }
      const responseData = await response.json();
      const competitorList = responseData.data || [];

      const transformedMarketShare = competitorList
        .filter((comp: any) => comp.market_share !== undefined && comp.market_share !== null && parseFloat(comp.market_share) > 0)
        .map((comp: any) => ({
          name: comp.company_name || comp.name || comp.title || 'Unnamed Competitor',
          value: parseFloat(comp.market_share),
        }));
      setMarketShareDataState(transformedMarketShare);

      const transformedCompetitorActivity = competitorList.map((comp: any) => ({
        name: comp.company_name || comp.name || comp.title || 'Unnamed Competitor',
        activity: parseFloat(comp.activity_score || comp.activity || comp.engagement_rate || 0),
        growth: parseFloat(comp.growth_rate || comp.growth || comp.user_growth || 0),
      }));
      setCompetitorActivityDataState(transformedCompetitorActivity);

    } catch (error) {
      console.error("Failed to fetch or transform chart data:", error);
      setMarketShareDataState([]);
      setCompetitorActivityDataState([]);
      toast({
        title: "Chart Data Error",
        description: "Failed to load chart data.",
        variant: "destructive",
      });
      throw error;
    }
  };

  const fetchTrendsApiData = async () => {
    try {
      // Calls the Next.js API route, which then calls the Python backend with auth
      const response = await fetch(`/api/trends`);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: `Failed to fetch trends data: ${response.status} ${response.statusText}` }));
        throw new Error(errorData.detail || `Failed to fetch trends data: ${response.status} ${response.statusText}`);
      }
      const responseData = await response.json();
      const trendsList = responseData.data || [];

      const impactToValue = (impact: string | undefined): number => {
        if (!impact) return 0;
        const lowerImpact = impact.toLowerCase();
        if (lowerImpact === 'high') return 3;
        if (lowerImpact === 'medium') return 2;
        if (lowerImpact === 'low') return 1;
        return 0;
      };

      const transformedTrendsData = trendsList.map((trend: any) => ({
        name: trend.trend_name || 'Unnamed Trend',
        impact: impactToValue(trend.estimated_impact),
      })).filter((trend: any) => trend.impact > 0);
      setTrendsChartDataState(transformedTrendsData);

    } catch (error) {
      console.error("Failed to fetch or transform trends data:", error);
      setTrendsChartDataState([]);
      toast({
        title: "Trends Data Error",
        description: "Failed to load trends data.",
        variant: "destructive",
      });
      throw error;
    }
  };

  const loadAllDashboardData = async () => {
    setIsRefreshing(true);
    try {
      await Promise.all([
        fetchKpiData(),
        fetchChartData(),
        fetchTrendsApiData(),
      ]);
    } catch (error) {
      console.error("Failed to load all dashboard data:", error);
      toast({
        title: "Data Load Error",
        description: "Could not fetch all dashboard data. Please try refreshing.",
        variant: "destructive",
      });
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    loadAllDashboardData();
  }, []);

  const handleRefresh = () => {
    loadAllDashboardData();
  };

  const handleGenerateReport = async () => {
    if (!analysisQuery.trim() || !analysisMarketDomain.trim()) {
      toast({
        title: "Validation Error",
        description: "Query and Market Domain are required.",
        variant: "destructive",
      });
      return;
    }
    setIsGeneratingReport(true);
    try {
      const response = await fetch('/api/analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query_str: analysisQuery,
          market_domain_str: analysisMarketDomain,
          question_str: analysisQuestion,
        }),
      });

      const result = await response.json();

      if (!response.ok || result.error || !result.success) {
        throw new Error(result.error || result.message || 'Failed to start analysis');
      }

      toast({
        title: "Analysis Started",
        description: `Report generation for "${analysisQuery}" is in progress. State ID: ${result.data?.state_id || result.state_id}`, // Adjusted to check result.data.state_id
      });
      setIsAnalysisDialogOpen(false);
      // Optionally clear form fields
      setAnalysisQuery("");
      setAnalysisMarketDomain("");
      setAnalysisQuestion("");
    } catch (error: any) {
      toast({
        title: "Error Generating Report",
        description: error.message || "An unexpected error occurred.",
        variant: "destructive",
      });
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const handleGetAiInsights = async () => {
    if (!aiInsightsQuery.trim() || !aiInsightsMarketDomain.trim()) {
      toast({
        title: "Validation Error",
        description: "Query and Market Domain are required for AI Insights.",
        variant: "destructive",
      });
      return;
    }
    setIsGeneratingAiInsights(true);
    try {
      const response = await fetch('/api/analysis', { // Assuming the same endpoint for now
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query_str: aiInsightsQuery,
          market_domain_str: aiInsightsMarketDomain,
          question_str: aiInsightsSpecificQuestion || "", // Pass empty string if not provided
        }),
      });

      const result = await response.json();

      if (!response.ok || result.error || !result.success) {
        throw new Error(result.error || result.message || 'Failed to get AI insights');
      }

      toast({
        title: "AI Insights Request Submitted",
        description: `Your request for "${aiInsightsQuery}" is being processed. State ID: ${result.data?.state_id || result.state_id}`,
      });
      setIsAiInsightsDialogOpen(false);
      // Clear form fields
      setAiInsightsQuery("");
      setAiInsightsMarketDomain("");
      setAiInsightsSpecificQuestion("");
    } catch (error: any) {
      toast({
        title: "Error Generating AI Insights",
        description: error.message || "An unexpected error occurred.",
        variant: "destructive",
      });
    } finally {
      setIsGeneratingAiInsights(false);
    }
  };

  const handleExportKpiData = () => {
    if (!kpiData || kpiData.length === 0) {
      toast({
        title: "No Data",
        description: "No KPI data to export.",
        variant: "default", // Or "destructive" if preferred
      });
      return;
    }

    const cleanedKpiData = kpiData.map(kpi => ({
      title: kpi.title,
      value: kpi.value,
      unit: kpi.unit,
      change: kpi.change,
      color: kpi.color,
      // Omit 'icon' as it's a React component
    }));

    const jsonString = JSON.stringify(cleanedKpiData, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'kpi_data.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      title: "Export Successful",
      description: "KPI data exported to kpi_data.json",
    });
  };

  return (
    <SidebarInset className="bg-dark-bg">
      {showOnboarding && user && (
        <OnboardingFlow onComplete={handleOnboardingComplete} userName={user.user_metadata?.full_name || user.email || ""} />
      )}
      {/* Header */}
      <header className="flex h-16 shrink-0 items-center gap-2 border-b border-dark-border bg-dark-card/50 backdrop-blur-sm px-4">
        <SidebarTrigger className="-ml-1 text-white hover:bg-dark-card" />
        <Separator orientation="vertical" className="mr-2 h-4 bg-dark-border" />
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-neon-blue" />
          <h1 className="text-lg font-semibold text-white">Market Intelligence Dashboard</h1>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <Button
            onClick={() => setIsWidgetSelectorOpen(true)}
            variant="outline"
            className="bg-dark-card/20 border border-dark-border text-white hover:bg-dark-card/50"
          >
            <Settings2 className="w-4 h-4 mr-2" />
            Customize
          </Button>
          <Button
            onClick={handleRefresh}
            disabled={isRefreshing}
            className="bg-neon-blue/20 border border-neon-blue/50 text-neon-blue hover:bg-neon-blue/30 hover:shadow-neon-blue/50 hover:shadow-lg transition-all duration-300"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${isRefreshing ? "animate-spin" : ""}`} />
            {isRefreshing ? "Refreshing..." : "Refresh Data"}
          </Button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex flex-1 flex-col gap-6 p-6">
        {enabledWidgets.includes("kpiCards") && (
          <>
            {/* KPI Cards */}
            {isRefreshing && kpiData.length === 0 ? (
              <div className="md:col-span-2 lg:col-span-4">
                <Card className="bg-dark-card border-dark-border">
                  <CardContent className="p-6 text-center text-gray-400">
                    Loading KPI data...
                  </CardContent>
                </Card>
              </div>
            ) : !isRefreshing && kpiData.length === 0 ? (
              <div className="md:col-span-2 lg:col-span-4">
                <Card className="bg-dark-card border-dark-border">
                  <CardContent className="p-6 text-center text-gray-400">
                    No KPI data available.
                  </CardContent>
                </Card>
              </div>
            ) : (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                {kpiData.map((kpi, index) => (
                  <KPICard key={kpi.title || index} {...kpi} onViewHistory={handleViewKpiHistory} />
                ))}
              </div>
            )}
          </>
        )}

        {enabledWidgets.includes("trendImpactChart") && enabledWidgets.includes("marketShareChart") && (
          <div className="grid gap-6 md:grid-cols-2">
            {/* Trend Impact Chart */}
            {enabledWidgets.includes("trendImpactChart") && (
              <Card className="bg-dark-card border-dark-border">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-neon-blue" />
                    Identified Trends Impact
                  </CardTitle>
                  <CardDescription className="text-gray-400">Estimated impact of key market trends</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={trendsChartDataState}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#404040" />
                      <XAxis dataKey="name" stroke="#9CA3AF" angle={-30} textAnchor="end" height={70} interval={0} />
                      <YAxis
                        stroke="#9CA3AF"
                        domain={[0, 3]}
                        ticks={[0, 1, 2, 3]}
                        tickFormatter={(value) => ['N/A', 'Low', 'Medium', 'High'][value]}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#2C2C2C",
                          border: "1px solid #404040",
                          borderRadius: "8px",
                          color: "#fff",
                        }}
                        formatter={(value: number) => {
                          const level = ['N/A', 'Low', 'Medium', 'High'][value];
                          return [level, "Impact"];
                        }}
                      />
                      <Bar dataKey="impact" name="Impact Level" fill={chartColors[0] || '#00FFFF'} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            )}

            {/* Market Share Chart */}
            {enabledWidgets.includes("marketShareChart") && (
              <Card className="bg-dark-card border-dark-border">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <PieChart className="w-5 h-5 text-neon-pink" />
                    Market Share
                  </CardTitle>
                  <CardDescription className="text-gray-400">Current market distribution</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <RechartsPieChart>
                      <Pie
                        data={marketShareDataState} // Use state variable
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        dataKey="value"
                        label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
                      >
                        {marketShareDataState.map((entry, index) => ( // Use state variable
                          <Cell key={`cell-${index}`} fill={chartColors[index % chartColors.length]} />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#2C2C2C",
                          border: "1px solid #404040",
                          borderRadius: "8px",
                          color: "#fff",
                        }}
                      />
                    </RechartsPieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            )}
          </div>
        )}

        {enabledWidgets.includes("competitorActivity") && (
          <Card className="bg-dark-card border-dark-border">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Users className="w-5 h-5 text-neon-green" />
                Competitor Activity
              </CardTitle>
              <CardDescription className="text-gray-400">Recent competitor performance and growth rates</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={competitorActivityDataState}> {/* Use state variable */}
                  <CartesianGrid strokeDasharray="3 3" stroke="#404040" />
                  <XAxis dataKey="name" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#2C2C2C",
                      border: "1px solid #404040",
                      borderRadius: "8px",
                      color: "#fff",
                    }}
                  />
                  <Bar dataKey="activity" fill={chartColors[1]} name="Activity Score" />
                  <Bar dataKey="growth" fill={chartColors[2]} name="Growth Rate %" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* Development Progress (Show in development) */}
        {enabledWidgets.includes("developmentProgress") && process.env.NODE_ENV === 'development' && (
          <Card className="bg-dark-card border-dark-border">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Target className="w-5 h-5 text-neon-purple" />
                Development Progress
              </CardTitle>
              <CardDescription className="text-gray-400">Track feature completion status</CardDescription>
            </CardHeader>
            <CardContent>
              <FeatureTracker />
            </CardContent>
          </Card>
        )}

        {enabledWidgets.includes("quickActions") && (
          <div className="grid gap-4 md:grid-cols-3">
            <Card
              className="bg-dark-card border-dark-border hover:border-neon-blue/50 transition-all duration-300 cursor-pointer group"
              onClick={() => setIsAnalysisDialogOpen(true)}
            >
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-lg bg-neon-blue/20 group-hover:bg-neon-blue/30 transition-colors">
                    <Target className="w-6 h-6 text-neon-blue" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">Generate Report</h3>
                    <p className="text-sm text-gray-400">Create comprehensive market analysis</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card
              className="bg-dark-card border-dark-border hover:border-neon-green/50 transition-all duration-300 cursor-pointer group"
              onClick={() => setIsAiInsightsDialogOpen(true)}
            >
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-lg bg-neon-green/20 group-hover:bg-neon-green/30 transition-colors">
                    <Zap className="w-6 h-6 text-neon-green" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">AI Insights</h3>
                    <p className="text-sm text-gray-400">Get AI-powered recommendations</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card
              className="bg-dark-card border-dark-border hover:border-neon-pink/50 transition-all duration-300 cursor-pointer group"
              onClick={handleExportKpiData}
            >
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-lg bg-neon-pink/20 group-hover:bg-neon-pink/30 transition-colors">
                    <Briefcase className="w-6 h-6 text-neon-pink" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">Export Data</h3>
                    <p className="text-sm text-gray-400">Download reports and datasets</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>

      <WidgetSelectorDialog
        isOpen={isWidgetSelectorOpen}
        onOpenChange={setIsWidgetSelectorOpen}
        currentEnabledWidgets={enabledWidgets}
        onSave={handleSaveWidgets}
      />

      {selectedKpiForHistory && (
        <HistoricalKpiChartDialog
          isOpen={isHistoricalKpiDialogOpen}
          onOpenChange={setIsHistoricalKpiDialogOpen}
          kpiTitle={selectedKpiForHistory.title}
          historicalData={historicalKpiData}
          unit={selectedKpiForHistory.unit}
        />
      )}

      {/* Generate Report Dialog */}
