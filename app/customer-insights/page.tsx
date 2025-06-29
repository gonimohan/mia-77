"use client"

import { useState, useEffect } from "react" // Added useEffect
import { Users, TrendingUp, Target, Heart, UserCheck, Download } from "lucide-react"
import { SidebarInset, SidebarTrigger } from "@/components/ui/sidebar"
import { Button } from "@/components/ui/button"
import { useToast } from "@/hooks/use-toast" // Added useToast import
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import { useColorPalette } from "@/lib/color-context"

interface CustomerSegment {
  segment_name: string
  description: string
  percentage: number
  key_characteristics: string[]
  pain_points: string[]
  growth_potential: string
  satisfaction_score: number
  retention_rate: number
  acquisition_cost: string
  lifetime_value: string
}

export default function CustomerInsightsPage() {
  const { getChartColors } = useColorPalette()
  const chartColors = getChartColors()
  const [insights, setInsights] = useState<CustomerSegment[]>([]) // Initialize with empty array
  const [loading, setLoading] = useState(true) // Start with loading true
  const { toast } = useToast(); // Initialize useToast

  const fetchCustomerInsightsData = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_PYTHON_AGENT_API_BASE_URL || 'http://localhost:8000'}/customer-insights`);
      if (!response.ok) {
        console.error(`Failed to fetch customer insights: ${response.statusText}`);
        setInsights([]);
        return;
      }
      const responseData = await response.json();
      const rawData = responseData.data || [];

      const transformedInsights: CustomerSegment[] = rawData.map((segment: any) => ({
        segment_name: segment.segment_name || "Unnamed Segment",
        description: segment.description || "No description.",
        percentage: parseFloat(segment.percentage) || 0,
        key_characteristics: Array.isArray(segment.key_characteristics) ? segment.key_characteristics : [],
        pain_points: Array.isArray(segment.pain_points) ? segment.pain_points : [],
        growth_potential: segment.growth_potential || "N/A",
        satisfaction_score: parseFloat(segment.satisfaction_score) || 0,
        retention_rate: parseFloat(segment.retention_rate) || 0,
        acquisition_cost: segment.acquisition_cost || "N/A",
        lifetime_value: segment.lifetime_value || "N/A",
      }));
      setInsights(transformedInsights);
    } catch (error) {
      console.error("Failed to fetch or transform customer insights:", error);
      setInsights([]);
      toast({
        title: "Customer Insights Error",
        description: "Failed to load customer insights data.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCustomerInsightsData();
  }, []);

  // Derived data should be memoized or calculated after loading and insights are set
  const satisfactionData = insights.map((segment) => ({
    name: segment.segment_name,
    satisfaction: segment.satisfaction_score,
    retention: segment.retention_rate,
  }))

  const segmentDistribution = insights.map((segment, index) => ({
    name: segment.segment_name,
    value: segment.percentage,
    color: chartColors[index % chartColors.length], // Ensure chartColors is available
  }))

  const getGrowthPotentialColor = (potential: string) => {
    switch (potential.toLowerCase()) {
      case "very high":
        return "text-neon-green"
      case "high":
        return "text-neon-blue"
      case "medium":
        return "text-neon-orange"
      case "low":
        return "text-neon-pink"
      default:
        return "text-gray-400"
    }
  }

  const getGrowthPotentialBadge = (potential: string) => {
    const variants = {
      "very high": "bg-neon-green/20 text-neon-green border-neon-green/50",
      high: "bg-neon-blue/20 text-neon-blue border-neon-blue/50",
      medium: "bg-neon-orange/20 text-neon-orange border-neon-orange/50",
      low: "bg-neon-pink/20 text-neon-pink border-neon-pink/50",
    }

    return (
      <Badge
        className={`${variants[potential.toLowerCase() as keyof typeof variants] || "bg-gray-500/20 text-gray-400"} border`}
      >
        {potential}
      </Badge>
    )
  }

  const handleExportInsights = () => {
    if (!insights || insights.length === 0) {
      toast({
        title: "No Data",
        description: "No customer insights data to export.",
        variant: "default",
      });
      return;
    }

    const jsonString = JSON.stringify(insights, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'customer_insights.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      title: "Export Successful",
      description: "Customer insights exported to customer_insights.json",
    });
  };

  if (loading) {
    return (
      <SidebarInset className="bg-dark-bg">
        <div className="flex h-16 shrink-0 items-center gap-2 border-b border-dark-border bg-dark-card/50 backdrop-blur-sm px-4">
           <SidebarTrigger className="-ml-1 text-white hover:bg-dark-card" />
           <Separator orientation="vertical" className="mr-2 h-4 bg-dark-border" />
            <div className="flex items-center gap-2">
                <Users className="w-5 h-5 text-neon-purple" />
                <h1 className="text-lg font-semibold text-white">Customer Insights</h1>
            </div>
        </div>
        <div className="flex flex-1 items-center justify-center p-6">
          <p className="text-white text-xl">Loading insights...</p>
        </div>
      </SidebarInset>
    )
  }

  // Robust average calculation
  const avgSatisfaction = insights.length > 0 ? (insights.reduce((acc, s) => acc + s.satisfaction_score, 0) / insights.length).toFixed(1) : "0.0";
  const avgRetention = insights.length > 0 ? Math.round(insights.reduce((acc, s) => acc + s.retention_rate, 0) / insights.length) : 0;


  return (
    <SidebarInset className="bg-dark-bg">
      {/* Header */}
      <header className="flex h-16 shrink-0 items-center gap-2 border-b border-dark-border bg-dark-card/50 backdrop-blur-sm px-4">
        <SidebarTrigger className="-ml-1 text-white hover:bg-dark-card" />
        <Separator orientation="vertical" className="mr-2 h-4 bg-dark-border" />
        <div className="flex items-center gap-2">
          <Users className="w-5 h-5 text-neon-purple" />
          <h1 className="text-lg font-semibold text-white">Customer Insights</h1>
        </div>
        <div className="ml-auto">
          <Button
            variant="outline"
            className="border-neon-blue/50 text-neon-blue hover:bg-neon-blue/10"
            onClick={handleExportInsights}
          >
            <Download className="w-4 h-4 mr-2" />
            Export Insights
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
                <div className="p-3 rounded-lg bg-neon-purple/20">
                  <Users className="w-6 h-6 text-neon-purple" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-white">{insights.length > 0 ? insights.length : "0"}</p>
                  <p className="text-sm text-gray-400">Customer Segments</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-dark-card border-dark-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-neon-green/20">
                  <Heart className="w-6 h-6 text-neon-green" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-white">
                    {avgSatisfaction}
                  </p>
                  <p className="text-sm text-gray-400">Avg Satisfaction</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-dark-card border-dark-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-neon-blue/20">
                  <UserCheck className="w-6 h-6 text-neon-blue" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-white">
                    {avgRetention}%
                  </p>
                  <p className="text-sm text-gray-400">Avg Retention</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-dark-card border-dark-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <div className="p-3 rounded-lg bg-neon-orange/20">
                  <TrendingUp className="w-6 h-6 text-neon-orange" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-white">
                    {insights.filter((s) => s.growth_potential === "High" || s.growth_potential === "Very High").length}
                  </p>
                  <p className="text-sm text-gray-400">High Growth Segments</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Charts Section */}
        <div className="grid gap-6 md:grid-cols-2">
          {/* Segment Distribution */}
          <Card className="bg-dark-card border-dark-border">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Target className="w-5 h-5 text-neon-purple" />
                Customer Segment Distribution
              </CardTitle>
              <CardDescription className="text-gray-400">Market share by customer segment</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={segmentDistribution}
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}%`}
                  >
                    {segmentDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
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
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Satisfaction & Retention */}
          <Card className="bg-dark-card border-dark-border">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Heart className="w-5 h-5 text-neon-green" />
                Satisfaction vs Retention
              </CardTitle>
              <CardDescription className="text-gray-400">
                Customer satisfaction and retention rates by segment
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={satisfactionData}>
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
                  <Bar dataKey="satisfaction" fill={chartColors[0]} name="Satisfaction Score" />
                  <Bar dataKey="retention" fill={chartColors[1]} name="Retention Rate %" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* Detailed Segments */}
        {!loading && insights.length === 0 ? (
          <Card className="bg-dark-card border-dark-border">
            <CardContent className="p-6 text-center text-gray-400">
              No customer segments data available.
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-6">
            {insights.map((segment, index) => (
              <Card key={segment.segment_name} className="bg-dark-card border-dark-border">
                <CardHeader>
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-white flex items-center gap-2">
                      <div
                        className="w-4 h-4 rounded-full"
                        style={{ backgroundColor: chartColors[index % chartColors.length] }}
                      />
                      {segment.segment_name}
                    </CardTitle>
                    <CardDescription className="text-gray-400 mt-2">{segment.description}</CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="border-gray-600 text-gray-400">
                      {segment.percentage}% of market
                    </Badge>
                    {getGrowthPotentialBadge(segment.growth_potential)}
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Metrics */}
                <div className="grid gap-4 md:grid-cols-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-400">Satisfaction</span>
                      <span className="text-sm font-medium text-white">{segment.satisfaction_score}/10</span>
                    </div>
                    <Progress value={segment.satisfaction_score * 10} className="h-2" />
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-400">Retention</span>
                      <span className="text-sm font-medium text-white">{segment.retention_rate}%</span>
                    </div>
                    <Progress value={segment.retention_rate} className="h-2" />
                  </div>

                  <div className="space-y-1">
                    <span className="text-sm text-gray-400">Acquisition Cost</span>
                    <p className="text-sm font-medium text-white">{segment.acquisition_cost}</p>
                  </div>

                  <div className="space-y-1">
                    <span className="text-sm text-gray-400">Lifetime Value</span>
                    <p className="text-sm font-medium text-white">{segment.lifetime_value}</p>
                  </div>
                </div>

                {/* Characteristics & Pain Points */}
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <h4 className="text-white font-medium mb-3">Key Characteristics</h4>
                    <div className="flex flex-wrap gap-2">
                      {segment.key_characteristics.map((characteristic, idx) => (
                        <Badge key={idx} variant="outline" className="border-neon-blue/50 text-neon-blue">
                          {characteristic}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="text-white font-medium mb-3">Pain Points</h4>
                    <div className="flex flex-wrap gap-2">
                      {segment.pain_points.map((painPoint, idx) => (
                        <Badge key={idx} variant="outline" className="border-neon-pink/50 text-neon-pink">
                          {painPoint}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
            ))}
          </div>
        )}
      </div>
    </SidebarInset>
  )
}
