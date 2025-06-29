"use client"

import React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { TrendingUp, TrendingDown, Minus, History } from "lucide-react"

interface KPICardProps {
  title: string;
  value: string;
  unit: string;
  change: number;
  icon: React.ElementType;
  color: "blue" | "green" | "pink" | "purple" | "orange";
  onViewHistory?: (kpiTitle: string) => void; // New prop for history button
}

export const KPICard: React.FC<KPICardProps> = ({
  title,
  value,
  unit,
  change,
  icon: Icon,
  color,
  onViewHistory,
}) => {
  const changeType = change > 0 ? "increase" : change < 0 ? "decrease" : "no-change"
  const changeIcon = change > 0 ? TrendingUp : change < 0 ? TrendingDown : Minus
  const changeColor = change > 0 ? "text-neon-green" : change < 0 ? "text-neon-pink" : "text-gray-400"

  const bgColorClass = `bg-${color}-500/20`
  const textColorClass = `text-neon-${color}`
  const borderColorClass = `border-${color}-500/50`

  return (
    <Card className={`bg-dark-card border ${borderColorClass}`}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-gray-400">{title}</CardTitle>
        <Icon className={`h-4 w-4 ${textColorClass}`} />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold text-white">
          {value} {unit}
        </div>
        <p className={`text-xs ${changeColor} flex items-center`}>
          {React.createElement(changeIcon, { className: "h-3 w-3 mr-1" })}
          {change}% from last period
        </p>
        {onViewHistory && (
          <Button
            variant="ghost"
            size="sm"
            className="mt-2 text-gray-400 hover:text-white hover:bg-dark-card/50"
            onClick={() => onViewHistory(title)}
          >
            <History className="w-3 h-3 mr-1" />
            View History
          </Button>
        )}
      </CardContent>
    </Card>
  )
}
