
"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { CheckCircle, Circle, AlertCircle, Clock } from "lucide-react"

interface Feature {
  id: string
  name: string
  description: string
  status: 'completed' | 'in-progress' | 'pending' | 'error'
  priority: 'high' | 'medium' | 'low'
  dependencies?: string[]
}

const features: Feature[] = [
  {
    id: 'auth',
    name: 'Authentication System',
    description: 'Complete login/register with Supabase',
    status: 'completed',
    priority: 'high'
  },
  {
    id: 'dashboard',
    name: 'Dashboard',
    description: 'Real-time KPIs and analytics',
    status: 'completed',
    priority: 'high'
  },
  {
    id: 'competitors',
    name: 'Competitor Analysis',
    description: 'Comprehensive competitor tracking',
    status: 'in-progress',
    priority: 'high'
  },
  {
    id: 'trends',
    name: 'Market Trends',
    description: 'AI-powered trend analysis',
    status: 'in-progress',
    priority: 'high'
  },
  {
    id: 'insights',
    name: 'Customer Insights',
    description: 'Advanced customer segmentation',
    status: 'pending',
    priority: 'medium'
  },
  {
    id: 'data-integration',
    name: 'Data Integration',
    description: 'Multiple data source management',
    status: 'in-progress',
    priority: 'high'
  },
  {
    id: 'downloads',
    name: 'Downloads & Reports',
    description: 'Export reports and data',
    status: 'pending',
    priority: 'medium'
  },
  {
    id: 'chat',
    name: 'AI Chat Interface',
    description: 'RAG-powered market intelligence chat',
    status: 'in-progress',
    priority: 'high'
  },
  {
    id: 'file-upload',
    name: 'File Upload',
    description: 'Document processing and analysis',
    status: 'pending',
    priority: 'medium'
  },
  {
    id: 'real-time-sync',
    name: 'Real-time Data Sync',
    description: 'Live data updates and synchronization',
    status: 'pending',
    priority: 'high'
  }
]

export function FeatureTracker() {
  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null)

  const getStatusIcon = (status: Feature['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'in-progress':
        return <Clock className="w-4 h-4 text-yellow-500" />
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />
      default:
        return <Circle className="w-4 h-4 text-gray-400" />
    }
  }

  const getStatusColor = (status: Feature['status']) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500/20 text-green-300 border-green-500/30'
      case 'in-progress':
        return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30'
      case 'error':
        return 'bg-red-500/20 text-red-300 border-red-500/30'
      default:
        return 'bg-gray-500/20 text-gray-300 border-gray-500/30'
    }
  }

  const getPriorityColor = (priority: Feature['priority']) => {
    switch (priority) {
      case 'high':
        return 'bg-red-500/20 text-red-300 border-red-500/30'
      case 'medium':
        return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30'
      default:
        return 'bg-green-500/20 text-green-300 border-green-500/30'
    }
  }

  const completedCount = features.filter(f => f.status === 'completed').length
  const totalCount = features.length
  const progressPercentage = (completedCount / totalCount) * 100

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <Card className="bg-dark-card border-dark-border">
        <CardHeader>
          <CardTitle className="text-white">Development Progress</CardTitle>
          <CardDescription>
            {completedCount} of {totalCount} features completed ({progressPercentage.toFixed(1)}%)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {features.map((feature) => (
              <div
                key={feature.id}
                className="flex items-center justify-between p-3 rounded-lg bg-dark-bg border border-dark-border hover:border-purple-500/30 cursor-pointer transition-colors"
                onClick={() => setSelectedFeature(feature)}
              >
                <div className="flex items-center gap-3">
                  {getStatusIcon(feature.status)}
                  <span className="text-white font-medium">{feature.name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge className={getPriorityColor(feature.priority)}>
                    {feature.priority}
                  </Badge>
                  <Badge className={getStatusColor(feature.status)}>
                    {feature.status}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {selectedFeature && (
        <Card className="bg-dark-card border-dark-border">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              {getStatusIcon(selectedFeature.status)}
              {selectedFeature.name}
            </CardTitle>
            <CardDescription>{selectedFeature.description}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <Badge className={getStatusColor(selectedFeature.status)}>
                  {selectedFeature.status}
                </Badge>
                <Badge className={getPriorityColor(selectedFeature.priority)}>
                  {selectedFeature.priority} priority
                </Badge>
              </div>
              {selectedFeature.dependencies && (
                <div>
                  <h4 className="text-white font-medium mb-2">Dependencies:</h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedFeature.dependencies.map((dep) => (
                      <Badge key={dep} variant="outline" className="border-gray-600 text-gray-300">
                        {dep}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
