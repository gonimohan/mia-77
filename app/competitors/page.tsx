"use client"

import { useState, useEffect } from "react" // Added useEffect
import { Users, TrendingUp, TrendingDown, Building, DollarSign, BarChart3, Eye, Search, Filter } from "lucide-react"
import { SidebarInset, SidebarTrigger } from "@/components/ui/sidebar"
import { Button } from "@/components/ui/button"
import { useToast } from "@/hooks/use-toast" // Import useToast
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
// All Recharts imports removed as no charts are left on this page.

import { useColorPalette } from "@/lib/color-context"

interface Competitor {
  id: string
  name: string
  marketShare: number
  revenue: string
  employees: number
  growthRate: number
  strengths: string[]
  weaknesses: string[]
  recentActivity: string
  threat_level: "low" | "medium" | "high"
  _revenueNumeric?: number; // Added for internal calculation
}

// const mockCompetitors: Competitor[] = [ ... ] // Removed mockCompetitors

// const competitorComparison = [ ... ] // Removed competitorComparison mock data

export default function CompetitorsPage() {
  const { getChartColors } = useColorPalette()
  const chartColors = getChartColors()
  const [competitors, setCompetitors] = useState<Competitor[]>([]) // Initialize with empty array
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedThreatLevel, setSelectedThreatLevel] = useState<string>("all")
  const threatLevels = ["all", "low", "medium", "high"]; // Defined threat levels
  const { toast } = useToast(); // Initialize useToast
  const [combinedMarketShareState, setCombinedMarketShareState] = useState<number>(0);
  const [combinedRevenueState, setCombinedRevenueState] = useState<string>("N/A");
  const [isLoading, setIsLoading] = useState(true); // For loading state

  const fetchCompetitorData = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_PYTHON_AGENT_API_BASE_URL || 'http://localhost:8000'}/competitors`);
      if (!response.ok) {
        console.error(`Failed to fetch competitor data: ${response.status} ${response.statusText}`);
        setCompetitors([]); // Set to empty on error
        return;
      }
      const responseData = await response.json();
      const rawComps = responseData.data || [];

      const transformedData: Competitor[] = rawComps.map((item: any, index: number) => {
        let determinedThreat: "low" | "medium" | "high" = "medium";
        const marketShareNum = parseFloat(item.market_share);
        if (item.threat_level) {
          determinedThreat = item.threat_level;
        } else if (!isNaN(marketShareNum)) {
          if (marketShareNum > 20) determinedThreat = "high";
          else if (marketShareNum < 10) determinedThreat = "low";
        }

        // Attempt to parse revenue string like "$2.5B" or "€100M" to a number for potential sum later
        let revenueNum = 0;
        if (typeof item.revenue === 'string' || typeof item.financial_data?.[0]?.value === 'string') {
            const revenueStr = item.revenue || item.financial_data?.[0]?.value || "";
            const match = revenueStr.match(/([\$€]?)([0-9\.]+)([BMK]?)/i);
            if (match) {
                let numPart = parseFloat(match[2]);
                const multiplier = match[3]?.toUpperCase();
                if (multiplier === 'B') numPart *= 1e9;
                else if (multiplier === 'M') numPart *= 1e6;
                else if (multiplier === 'K') numPart *= 1e3;
                revenueNum = numPart;
            }
        } else if (typeof item.revenue === 'number') {
            revenueNum = item.revenue;
        }


        return {
          id: item.id || item.company_name || item.name || item.title || `comp-${index}`,
          name: item.company_name || item.name || item.title || "Unknown Competitor",
          marketShare: !isNaN(marketShareNum) ? marketShareNum : 0,
          revenue: item.revenue_string || item.revenue || (revenueNum ? `$${(revenueNum/1e9).toFixed(1)}B` : "N/A"), // Display formatted or original
          employees: parseInt(item.employees || item.employee_count || item.number_of_employees) || 0,
          growthRate: parseFloat(item.growth_rate || item.growth_rate_percentage) || 0,
          strengths: Array.isArray(item.strengths) ? item.strengths : (typeof item.strengths === 'string' ? item.strengths.split(',').map(s => s.trim()) : []),
          weaknesses: Array.isArray(item.weaknesses) ? item.weaknesses : (typeof item.weaknesses === 'string' ? item.weaknesses.split(',').map(s => s.trim()) : []),
          recentActivity: item.recent_activity || item.summary || item.latest_news_summary || "N/A",
          threat_level: determinedThreat,
          // Store raw revenue number for sum if needed, or transform revenue to number directly if consistent
          _revenueNumeric: revenueNum
        };
      });
      setCompetitors(transformedData);

      const totalMarketShare = transformedData.reduce((acc, comp) => acc + comp.marketShare, 0);
      setCombinedMarketShareState(totalMarketShare);

      const totalNumericRevenue = transformedData.reduce((acc, comp) => acc + (comp._revenueNumeric || 0), 0);
        if (totalNumericRevenue > 0) {
            if (totalNumericRevenue >= 1e9) {
                setCombinedRevenueState(`$${(totalNumericRevenue / 1e9).toFixed(1)}B`);
            } else if (totalNumericRevenue >= 1e6) {
                setCombinedRevenueState(`$${(totalNumericRevenue / 1e6).toFixed(1)}M`);
            } else {
                setCombinedRevenueState(`$${totalNumericRevenue.toLocaleString()}`);
            }
        } else {
            setCombinedRevenueState("N/A");
        }

    } catch (error) {
      console.error("Failed to fetch or transform competitor data:", error);
      setCompetitors([]);
      toast({
        title: "Competitor Data Error",
        description: "Failed to load competitor data.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchCompetitorData();
  }, []);


  const filteredCompetitors = competitors.filter((competitor) => {
    const matchesSearch = competitor.name.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesThreat = selectedThreatLevel === "all" || competitor.threat_level === selectedThreatLevel
    return matchesSearch && matchesThreat
  })

  const handleFilterCycle = () => {
    const currentIndex = threatLevels.indexOf(selectedThreatLevel);
    const nextIndex = (currentIndex + 1) % threatLevels.length;
    setSelectedThreatLevel(threatLevels[nextIndex]);
  };

  const getThreatBadge = (level: string) => {
    const variants = {
      high: "bg-neon-pink/20 text-neon-pink border-neon-pink/50",
      medium: "bg-neon-orange/20 text-neon-orange border-neon-orange/50",
      low: "bg-neon-green/20 text-neon-green border-neon-green/50",
    }

    return (
      <Badge className={`${variants[level as keyof typeof variants]} border`}>
        {level.charAt(0).toUpperCase() + level.slice(1)} Threat
      </Badge>
    )
  }

  const handleViewCompetitor = (competitorName: string) => {
    toast({
      title: "Feature Coming Soon",
      description: `Detailed view for ${competitorName} is not yet implemented.`,
    });
  };

  return (
    <SidebarInset className="bg-dark-bg">
      {/* Header */}
      <header className="flex h-16 shrink-0 items-center gap-2 border-b border-dark-border bg-dark-card/50 backdrop-blur-sm px-4">
        <SidebarTrigger className="-ml-1 text-white hover:bg-dark-card" />
        <Separator orientation="vertical" className="mr-2 h-4 bg-dark-border" />
        <div className="flex items-center gap-2">
          <Users className="w-5 h-5 text-neon-green" />
          <h1 className="text-lg font-semibold text-white">Competitor Analysis</h1>
        </div>
        <div className="ml-auto flex items-center gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <Input
              placeholder="Search competitors..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 w-64 bg-dark-card border-dark-border text-white placeholder:text-gray-400"
            />
          </div>
          <Button
            variant="outline"
            className="border-neon-green/50 text-neon-green hover:bg-neon-green/10"
            onClick={handleFilterCycle}
          >
            <Filter className="w-4 h-4 mr-2" />
            Filter: {selectedThreatLevel.charAt(0).toUpperCase() + selectedThreatLevel.slice(1)}
          </Button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex flex-1 flex-col gap-6 p-6">
        {/* Overview Cards */}
        <div className="grid gap-4 md:grid-cols-4">
          <Card className="bg-dark-card border-dark-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-neon-green/20">
                  <Users className="w-6 h-6 text-neon-green" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-white">{competitors.length}</p>
                  <p className="text-sm text-gray-400">Active Competitors</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-dark-card border-dark-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-neon-pink/20">
                  <TrendingUp className="w-6 h-6 text-neon-pink" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-white">
                    {competitors.filter((c) => c.threat_level === "high").length}
                  </p>
                  <p className="text-sm text-gray-400">High Threat</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-dark-card border-dark-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-neon-blue/20">
                  <BarChart3 className="w-6 h-6 text-neon-blue" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-white">{combinedMarketShareState.toFixed(1)}%</p>
                  <p className="text-sm text-gray-400">Combined Market Share</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-dark-card border-dark-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-neon-purple/20">
                  <DollarSign className="w-6 h-6 text-neon-purple" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-white">{combinedRevenueState}</p>
                  <p className="text-sm text-gray-400">Combined Revenue</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Competitor Comparison Chart - REMOVED */}

        {/* Competitors List */}
        {isLoading ? (
          <Card className="bg-dark-card border-dark-border">
            <CardContent className="p-6 text-center text-gray-400">
              Loading competitors list...
            </CardContent>
          </Card>
        ) : filteredCompetitors.length > 0 ? (
          <div className="grid gap-4">
            {filteredCompetitors.map((competitor) => (
              <Card
                key={competitor.id}
                // ... rest of competitor card
                className="bg-dark-card border-dark-border hover:border-opacity-50 transition-all duration-300"
              >
                <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="p-3 rounded-lg bg-neon-green/20">
                      <Building className="w-6 h-6 text-neon-green" />
                    </div>
                    <div>
                      <CardTitle className="text-white">{competitor.name}</CardTitle>
                      <CardDescription className="text-gray-400">
                        {competitor.employees.toLocaleString()} employees • {competitor.revenue} revenue
                      </CardDescription>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {getThreatBadge(competitor.threat_level)}
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-gray-400 hover:text-neon-blue hover:bg-neon-blue/10"
                      onClick={() => handleViewCompetitor(competitor.name)}
                    >
                      <Eye className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-400">Market Share</span>
                      <span className="text-sm font-medium text-white">{competitor.marketShare}%</span>
                    </div>
                    <Progress value={competitor.marketShare} className="h-2 bg-dark-bg" />
                  </div>
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-400">Growth Rate</span>
                      <div className="flex items-center gap-1">
                        {competitor.growthRate > 0 ? (
                          <TrendingUp className="w-3 h-3 text-neon-green" />
                        ) : (
                          <TrendingDown className="w-3 h-3 text-neon-pink" />
                        )}
                        <span
                          className={`text-sm font-medium ${
                            competitor.growthRate > 0 ? "text-neon-green" : "text-neon-pink"
                          }`}
                        >
                          {competitor.growthRate}%
                        </span>
                      </div>
                    </div>
                    <Progress value={Math.abs(competitor.growthRate)} className="h-2 bg-dark-bg" />
                  </div>
                </div>

                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-medium text-neon-green mb-2">Strengths</h4>
                    <div className="flex flex-wrap gap-1">
                      {competitor.strengths.map((strength, index) => (
                        <Badge key={index} variant="outline" className="text-xs border-neon-green/30 text-neon-green">
                          {strength}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-neon-pink mb-2">Weaknesses</h4>
                    <div className="flex flex-wrap gap-1">
                      {competitor.weaknesses.map((weakness, index) => (
                        <Badge key={index} variant="outline" className="text-xs border-neon-pink/30 text-neon-pink">
                          {weakness}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="pt-2 border-t border-dark-border">
                  <p className="text-sm text-gray-400">
                    <span className="font-medium text-white">Recent Activity:</span> {competitor.recentActivity}
                  </p>
                </div>
              </CardContent>
            </Card>
            ))}
          </div>
        ) : (
          <Card className="bg-dark-card border-dark-border">
            <CardContent className="p-6 text-center text-gray-400">
              No competitors found matching your criteria.
            </CardContent>
          </Card>
        )}
      </div>
    </SidebarInset>
  )
}
