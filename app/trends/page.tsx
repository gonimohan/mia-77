"use client"

import { useState, useEffect } from "react" // Added useEffect
import { useRouter } from "next/navigation" // Import useRouter
import { TrendingUp, Calendar, Target, Zap, ArrowUp, ArrowDown, Minus, Download } from "lucide-react"
import { SidebarInset, SidebarTrigger } from "@/components/ui/sidebar"
import { Button } from "@/components/ui/button"
import { useToast } from "@/hooks/use-toast" // Import useToast
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface Trend {
  id: string;
  name: string;
  category: string; // Will be defaulted or based on market_domain
  impact: "high" | "medium" | "low";
  direction: "up" | "down" | "stable"; // Default if not from API
  confidence: number; // Default if not from API
  timeframe: "short" | "medium" | "long";
  description: string;
  // data: Array<{ month: string; value: number }>; // Removed mini-chart data
}

export default function TrendsPage() {
  const [trends, setTrends] = useState<Trend[]>([]) // Initialize with empty array
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [selectedCategory, setSelectedCategory] = useState<string>("all")
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>("all")
  const { toast } = useToast(); // Initialize useToast
  const router = useRouter(); // Initialize useRouter

  const fetchTrendsData = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_PYTHON_AGENT_API_BASE_URL || 'http://localhost:8000'}/trends`);
      if (!response.ok) {
        console.error(`Failed to fetch trends data: ${response.statusText}`);
        setTrends([]);
        return;
      }
      const responseData = await response.json();
      const trendsListFromApi = responseData.data || [];

      const transformedData: Trend[] = trendsListFromApi.map((item: any, index: number) => {
        const impactMap = { high: "high", medium: "medium", low: "low" };
        const timeframeMap = { "short-term": "short", "medium-term": "medium", "long-term": "long" };

        return {
          id: item.id || item.trend_name || `trend-${index}`,
          name: item.trend_name || "Unknown Trend",
          category: item.category || item.market_domain || "General", // Default category
          impact: impactMap[item.estimated_impact?.toLowerCase() as keyof typeof impactMap] || "medium",
          direction: item.direction || "stable", // Default direction
          confidence: item.confidence_score || 75, // Default confidence
          timeframe: timeframeMap[item.timeframe?.toLowerCase() as keyof typeof timeframeMap] || "medium",
          description: item.description || item.supporting_evidence || "No description available.",
        };
      });
      setTrends(transformedData);
    } catch (error) {
      console.error("Failed to fetch or transform trends data:", error);
      setTrends([]);
      toast({
        title: "Trends Data Error",
        description: "Failed to load trends data.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchTrendsData();
  }, []);

  const filteredTrends = trends.filter((trend) => {
    const matchesCategory = selectedCategory === "all" || trend.category.toLowerCase() === selectedCategory
    const matchesTimeframe = selectedTimeframe === "all" || trend.timeframe === selectedTimeframe
    return matchesCategory && matchesTimeframe
  })

  const getImpactBadge = (impact: string) => {
    const variants = {
      high: "bg-neon-pink/20 text-neon-pink border-neon-pink/50",
      medium: "bg-neon-orange/20 text-neon-orange border-neon-orange/50",
      low: "bg-neon-green/20 text-neon-green border-neon-green/50",
    }

    return (
      <Badge className={`${variants[impact as keyof typeof variants]} border`}>
        {impact.charAt(0).toUpperCase() + impact.slice(1)} Impact
      </Badge>
    )
  }

  const getDirectionIcon = (direction: string) => {
    switch (direction) {
      case "up":
        return <ArrowUp className="w-4 h-4 text-neon-green" />
      case "down":
        return <ArrowDown className="w-4 h-4 text-neon-pink" />
      default:
        return <Minus className="w-4 h-4 text-gray-400" />
    }
  }

  const getTimeframeBadge = (timeframe: string) => {
    const variants = {
      short: "bg-neon-blue/20 text-neon-blue border-neon-blue/50",
      medium: "bg-neon-purple/20 text-neon-purple border-neon-purple/50",
      long: "bg-neon-orange/20 text-neon-orange border-neon-orange/50",
    }

    return (
      <Badge variant="outline" className={`${variants[timeframe as keyof typeof variants]} border`}>
        {timeframe.charAt(0).toUpperCase() + timeframe.slice(1)}-term
      </Badge>
    )
  }

  const handleExportTrends = () => {
    if (!filteredTrends || filteredTrends.length === 0) {
      toast({
        title: "No Data",
        description: "No trends data to export for the current filters.",
        variant: "default",
      });
      return;
    }

    const jsonString = JSON.stringify(filteredTrends, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'market_trends.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      title: "Export Successful",
      description: "Market trends exported to market_trends.json",
    });
  };

  const handleAnalyzeTrend = (trend: Trend) => {
    localStorage.setItem("analyzeTrendName", trend.name);
    localStorage.setItem("analyzeTrendContext", trend.category); // Using category as context
    toast({
      title: "Preparing Analysis",
      description: `Redirecting to dashboard. Context set for '${trend.name}'.`,
    });
    router.push("/dashboard");
  };

  return (
    <SidebarInset className="bg-dark-bg">
      {/* Header */}
      <header className="flex h-16 shrink-0 items-center gap-2 border-b border-dark-border bg-dark-card/50 backdrop-blur-sm px-4">
        <SidebarTrigger className="-ml-1 text-white hover:bg-dark-card" />
        <Separator orientation="vertical" className="mr-2 h-4 bg-dark-border" />
        <div className="flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-neon-pink" />
          <h1 className="text-lg font-semibold text-white">Market Trends</h1>
        </div>
        <div className="ml-auto flex items-center gap-4">
          <Select value={selectedCategory} onValueChange={setSelectedCategory}>
            <SelectTrigger className="w-40 bg-dark-card border-dark-border text-white">
              <SelectValue placeholder="Category" />
            </SelectTrigger>
            <SelectContent className="bg-dark-card border-dark-border">
              <SelectItem value="all">All Categories</SelectItem>
              <SelectItem value="technology">Technology</SelectItem>
              <SelectItem value="environment">Environment</SelectItem>
              <SelectItem value="saas">SaaS</SelectItem>
              <SelectItem value="finance">Finance</SelectItem>
            </SelectContent>
          </Select>

          <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
            <SelectTrigger className="w-40 bg-dark-card border-dark-border text-white">
              <SelectValue placeholder="Timeframe" />
            </SelectTrigger>
            <SelectContent className="bg-dark-card border-dark-border">
              <SelectItem value="all">All Timeframes</SelectItem>
              <SelectItem value="short">Short-term</SelectItem>
              <SelectItem value="medium">Medium-term</SelectItem>
              <SelectItem value="long">Long-term</SelectItem>
            </SelectContent>
          </Select>

          <Button
            variant="outline"
            className="border-neon-blue/50 text-neon-blue hover:bg-neon-blue/10"
            onClick={handleExportTrends}
          >
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex flex-1 flex-col gap-6 p-6">
        {/* Overview Chart - REMOVED */}

        {/* Trends Grid */}
        {isLoading ? (
          <Card className="bg-dark-card border-dark-border">
            <CardContent className="p-6 text-center text-gray-400">
              Loading trends list...
            </CardContent>
          </Card>
        ) : filteredTrends.length > 0 ? (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3"> {/* Adjusted for potentially more cards */}
            {filteredTrends.map((trend) => (
              <Card
                key={trend.id}
                className="bg-dark-card border-dark-border hover:border-opacity-50 transition-all duration-300"
              >
                <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="space-y-2">
                    <CardTitle className="text-white flex items-center gap-2">
                      {getDirectionIcon(trend.direction)}
                      {trend.name}
                    </CardTitle>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="border-gray-600 text-gray-400">
                        {trend.category}
                      </Badge>
                      {getTimeframeBadge(trend.timeframe)}
                    </div>
                  </div>
                  {getImpactBadge(trend.impact)}
                </div>
                <CardDescription className="text-gray-400">{trend.description}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Confidence Score</span>
                  <div className="flex items-center gap-2">
                    <div className="w-24 h-2 bg-dark-bg rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-neon-pink to-neon-blue rounded-full transition-all duration-300"
                        style={{ width: `${trend.confidence}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium text-white">{trend.confidence}%</span>
                  </div>
                </div>

                {/* Mini-chart removed from here */}

                <div className="flex items-center justify-between pt-4 mt-4 border-t border-dark-border">
                  <div className="flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-gray-400" />
                    <span className="text-xs text-gray-400">
                      {trend.timeframe.charAt(0).toUpperCase() + trend.timeframe.slice(1)}-term outlook
                    </span>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-neon-blue hover:text-neon-blue hover:bg-neon-blue/10"
                    onClick={() => handleAnalyzeTrend(trend)}
                  >
                    <Target className="w-3 h-3 mr-1" />
                    Analyze
                  </Button>
                </div>
              </CardContent>
            </Card>
            ))}
          </div>
        ) : (
          <Card className="bg-dark-card border-dark-border">
            <CardContent className="p-6 text-center text-gray-400">
              No trends found matching your criteria.
            </CardContent>
          </Card>
        )}

        {/* Quick Insights */}
        <div className="grid gap-4 md:grid-cols-3">
          <Card className="bg-dark-card border-dark-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-neon-green/20">
                  <ArrowUp className="w-6 h-6 text-neon-green" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-white">{trends.filter((t) => t.direction === "up").length}</p>
                  <p className="text-sm text-gray-400">Rising Trends</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-dark-card border-dark-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-neon-pink/20">
                  <Zap className="w-6 h-6 text-neon-pink" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-white">{trends.filter((t) => t.impact === "high").length}</p>
                  <p className="text-sm text-gray-400">High Impact</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-dark-card border-dark-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-neon-blue/20">
                  <Target className="w-6 h-6 text-neon-blue" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-white">
                    {trends.length > 0 ? Math.round(trends.reduce((acc, t) => acc + t.confidence, 0) / trends.length) : 0}%
                  </p>
                  <p className="text-sm text-gray-400">Avg Confidence</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </SidebarInset>
  )
}
