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


// Sample data for charts
// const trendData = [ // This will be replaced by fetched data
//   { month: "Jan", value: 65, competitors: 45 },
//   { month: "Feb", value: 72, competitors: 48 },
//   { month: "Mar", value: 68, competitors: 52 },
//   { month: "Apr", value: 85, competitors: 58 },
//   { month: "May", value: 92, competitors: 62 },
//   { month: "Jun", value: 88, competitors: 65 },
// ]

const marketShareData = [
  { name: "Our Company", value: 35, color: "#00FFFF" },
  { name: "Competitor A", value: 25, color: "#39FF14" },
  { name: "Competitor B", value: 20, color: "#FF1493" },
  { name: "Others", value: 20, color: "#BF00FF" },
]

const competitorData = [
  { name: "TechCorp", activity: 85, growth: 12.3 },
  { name: "InnovateLabs", activity: 72, growth: 8.7 },
  { name: "FutureSoft", activity: 91, growth: 15.6 },
  { name: "DataDrive", activity: 68, growth: 6.2 },
]

export default function DashboardPage() {
  const { getChartColors } = useColorPalette()
  const chartColors = getChartColors()
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [kpiData, setKpiData] = useState<any[]>([]) // Initialize with empty array
  const [marketShareDataState, setMarketShareDataState] = useState<any[]>([]);
  const [competitorActivityDataState, setCompetitorActivityDataState] = useState<any[]>([]);
  const [trendsChartDataState, setTrendsChartDataState] = useState<any[]>([]);

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


  const fetchKpiData = async () => {
    setIsRefreshing(true);
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
            value: parseFloat(kpi.metric_value).toLocaleString() || "0",
            unit: kpi.metric_unit || "",
            change: parseFloat(kpi.change_percentage) || 0,
            icon: assignedIcon,
            color: assignedColor as "blue" | "green" | "pink" | "purple" | "orange", // Type assertion
            // description: kpi.description, // If you add a description field
          };
        });
        setKpiData(transformedKpis);
      } else {
        console.warn("No data received from /api/kpi or data format is incorrect");
        setKpiData([]); // Set to empty if no data
      }
    } catch (error) {
      console.error("Failed to fetch KPI data:", error);
      setKpiData([]); // Set to empty on error
    } finally {
      setIsRefreshing(false);
    }
  };

  const fetchChartData = async () => {
    // Assuming setIsRefreshing is handled by the calling context (e.g. fetchKpiData or handleRefresh)
    // If called independently, uncomment:
    // setIsRefreshing(true);
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_PYTHON_AGENT_API_BASE_URL || 'http://localhost:8000'}/competitors`);
      if (!response.ok) {
        console.error(`Failed to fetch competitor data: ${response.status} ${response.statusText}`);
        // Potentially set chart data to empty or error state here
        setMarketShareDataState([]);
        setCompetitorActivityDataState([]);
        return; // Exit if the fetch fails
      }
      const responseData = await response.json();
      const competitorList = responseData.data || [];

      // Transform for Market Share Chart
      const transformedMarketShare = competitorList
        .filter((comp: any) => comp.market_share !== undefined && comp.market_share !== null && parseFloat(comp.market_share) > 0)
        .map((comp: any, index: number) => ({
          name: comp.company_name || comp.name || comp.title || `Competitor ${index + 1}`,
          value: parseFloat(comp.market_share),
        }));
      setMarketShareDataState(transformedMarketShare);

      // Transform for Competitor Activity Chart
      const transformedCompetitorActivity = competitorList.map((comp: any, index: number) => ({
        name: comp.company_name || comp.name || comp.title || `Competitor ${index + 1}`,
        activity: parseFloat(comp.activity_score || comp.activity || comp.engagement_rate || 0),
        growth: parseFloat(comp.growth_rate || comp.growth || comp.user_growth || 0),
      }));
      setCompetitorActivityDataState(transformedCompetitorActivity);

    } catch (error) {
      console.error("Failed to fetch or transform chart data:", error);
      setMarketShareDataState([]);
      setCompetitorActivityDataState([]);
    }
    // If called independently, uncomment:
    // finally { setIsRefreshing(false); }
  };

  const fetchTrendsApiData = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_PYTHON_AGENT_API_BASE_URL || 'http://localhost:8000'}/trends`);
      if (!response.ok) {
        console.error(`Failed to fetch trends data: ${response.status} ${response.statusText}`);
        setTrendsChartDataState([]);
        return;
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

      const transformedTrendsData = trendsList.map((trend: any, index: number) => ({
        name: trend.trend_name || `Trend ${index + 1}`,
        impact: impactToValue(trend.estimated_impact),
      })).filter((trend: any) => trend.impact > 0);
      setTrendsChartDataState(transformedTrendsData);

    } catch (error) {
      console.error("Failed to fetch or transform trends data:", error);
      setTrendsChartDataState([]);
    }
  };

  useEffect(() => {
    fetchKpiData();
    fetchChartData();
    fetchTrendsApiData();
  }, []); // Empty dependency array means this runs once on mount

  const handleRefresh = async () => {
    setIsRefreshing(true); // Set refreshing true for the whole refresh operation
    await Promise.all([fetchKpiData(), fetchChartData(), fetchTrendsApiData()]);
    // setIsRefreshing(false) is handled by fetchKpiData's finally block (or the last promise in Promise.all if they also manage it)
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
              <KPICard key={kpi.title || index} {...kpi} />
            ))}
          </div>
        )}

        {/* Charts Section */}
        <div className="grid gap-6 md:grid-cols-2">
          {/* Trend Impact Chart */}
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

          {/* Market Share Chart */}
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
        </div>

        {/* Competitor Activity */}
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

        {/* Development Progress (Show in development) */}
        {process.env.NODE_ENV === 'development' && (
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

        {/* Quick Actions */}
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
      </div>

      {/* Generate Report Dialog */}
      <Dialog open={isAnalysisDialogOpen} onOpenChange={setIsAnalysisDialogOpen}>
        <DialogContent className="bg-dark-card border-dark-border text-white sm:max-w-[525px]">
          <DialogHeader>
            <DialogTitle className="text-neon-blue">Generate New Market Analysis</DialogTitle>
            <DialogDescription>
              Enter the details for your market analysis report. The agent will begin processing once submitted.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid items-center grid-cols-4 gap-4">
              <Label htmlFor="analysisQueryDialog" className="text-right col-span-1">Query*</Label>
              <Input
                id="analysisQueryDialog"
                value={analysisQuery}
                onChange={(e) => setAnalysisQuery(e.target.value)}
                placeholder="e.g., AI impact on EdTech"
                className="col-span-3 bg-dark-bg border-dark-border placeholder:text-gray-500"
              />
            </div>
            <div className="grid items-center grid-cols-4 gap-4">
              <Label htmlFor="analysisMarketDomainDialog" className="text-right col-span-1">Market Domain*</Label>
              <Input
                id="analysisMarketDomainDialog"
                value={analysisMarketDomain}
                onChange={(e) => setAnalysisMarketDomain(e.target.value)}
                placeholder="e.g., EdTech, FinTech"
                className="col-span-3 bg-dark-bg border-dark-border placeholder:text-gray-500"
              />
            </div>
            <div className="grid items-center grid-cols-4 gap-4">
              <Label htmlFor="analysisQuestionDialog" className="text-right col-span-1">Question (Optional)</Label>
              <Input
                id="analysisQuestionDialog"
                value={analysisQuestion}
                onChange={(e) => setAnalysisQuestion(e.target.value)}
                placeholder="e.g., Key investment areas?"
                className="col-span-3 bg-dark-bg border-dark-border placeholder:text-gray-500"
              />
            </div>
          </div>
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline" className="border-gray-600 text-gray-300 hover:bg-dark-bg hover:text-white">Cancel</Button>
            </DialogClose>
            <Button onClick={handleGenerateReport} disabled={isGeneratingReport} className="bg-neon-blue hover:bg-neon-blue/90 text-white">
              {isGeneratingReport ? "Generating..." : "Generate Analysis"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* AI Insights Dialog */}
      <Dialog open={isAiInsightsDialogOpen} onOpenChange={setIsAiInsightsDialogOpen}>
        <DialogContent className="bg-dark-card border-dark-border text-white sm:max-w-[525px]">
          <DialogHeader>
            <DialogTitle className="text-neon-green">Get AI-Powered Insights</DialogTitle>
            <DialogDescription>
              Describe your area of interest to receive AI-powered recommendations and insights.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid items-center grid-cols-4 gap-4">
              <Label htmlFor="aiInsightsQueryDialog" className="text-right col-span-1">Query*</Label>
              <Input
                id="aiInsightsQueryDialog"
                value={aiInsightsQuery}
                onChange={(e) => setAiInsightsQuery(e.target.value)}
                placeholder="e.g., Future of remote work"
                className="col-span-3 bg-dark-bg border-dark-border placeholder:text-gray-500"
              />
            </div>
            <div className="grid items-center grid-cols-4 gap-4">
              <Label htmlFor="aiInsightsMarketDomainDialog" className="text-right col-span-1">Market Domain*</Label>
              <Input
                id="aiInsightsMarketDomainDialog"
                value={aiInsightsMarketDomain}
                onChange={(e) => setAiInsightsMarketDomain(e.target.value)}
                placeholder="e.g., Human Resources, SaaS"
                className="col-span-3 bg-dark-bg border-dark-border placeholder:text-gray-500"
              />
            </div>
            <div className="grid items-center grid-cols-4 gap-4">
              <Label htmlFor="aiInsightsSpecificQuestionDialog" className="text-right col-span-1">Specific Question (Optional)</Label>
              <Input
                id="aiInsightsSpecificQuestionDialog"
                value={aiInsightsSpecificQuestion}
                onChange={(e) => setAiInsightsSpecificQuestion(e.target.value)}
                placeholder="e.g., What are emerging collaboration tools?"
                className="col-span-3 bg-dark-bg border-dark-border placeholder:text-gray-500"
              />
            </div>
          </div>
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline" className="border-gray-600 text-gray-300 hover:bg-dark-bg hover:text-white">Cancel</Button>
            </DialogClose>
            <Button onClick={handleGetAiInsights} disabled={isGeneratingAiInsights} className="bg-neon-green hover:bg-neon-green/90 text-black">
              {isGeneratingAiInsights ? "Generating Insights..." : "Get Insights"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </SidebarInset>
  )
}
