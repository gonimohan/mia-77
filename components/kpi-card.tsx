"use client"

import type { LucideIcon } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface KPICardProps {
  title: string
  value: string | number
  unit?: string
  change: number
  icon: LucideIcon
  color: string
  description?: string
}

export function KPICard({ title, value, unit, change, icon: Icon, color, description }: KPICardProps) {
  const isPositive = change >= 0
  const colorClasses = {
    blue: "text-neon-blue border-neon-blue/20 bg-neon-blue/5",
    green: "text-neon-green border-neon-green/20 bg-neon-green/5",
    pink: "text-neon-pink border-neon-pink/20 bg-neon-pink/5",
    purple: "text-neon-purple border-neon-purple/20 bg-neon-purple/5",
    orange: "text-neon-orange border-neon-orange/20 bg-neon-orange/5",
  }

  return (
    <Card
      className={`
      bg-dark-card border-dark-border hover:border-opacity-50 transition-all duration-300 
      hover:shadow-lg hover:scale-105 group relative overflow-hidden
      ${colorClasses[color as keyof typeof colorClasses] || colorClasses.blue}
    `}
    >
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-gray-300 group-hover:text-white transition-colors">
          {title}
        </CardTitle>
        <Icon
          className={`h-5 w-5 transition-all duration-300 group-hover:scale-110 group-hover:drop-shadow-lg ${
            color === "blue"
              ? "text-neon-blue"
              : color === "green"
                ? "text-neon-green"
                : color === "pink"
                  ? "text-neon-pink"
                  : color === "purple"
                    ? "text-neon-purple"
                    : "text-neon-orange"
          }`}
        />
      </CardHeader>
      <CardContent>
        <div className="flex items-baseline space-x-2">
          <div className="text-2xl font-bold text-white group-hover:text-opacity-90 transition-colors">
            {value}
            {unit && <span className="text-lg text-gray-400 ml-1">{unit}</span>}
          </div>
        </div>
        <div className="flex items-center space-x-2 mt-2">
          <div className={`text-sm font-medium ${isPositive ? "text-neon-green" : "text-neon-pink"}`}>
            {isPositive ? "+" : ""}
            {change}%
          </div>
          <div className="text-xs text-gray-400">vs last period</div>
        </div>
        {description && (
          <p className="text-xs text-gray-500 mt-2 group-hover:text-gray-400 transition-colors">{description}</p>
        )}

        {/* Subtle glow effect on hover */}
        <div
          className={`
          absolute inset-0 opacity-0 group-hover:opacity-10 transition-opacity duration-300 pointer-events-none
          ${
            color === "blue"
              ? "bg-neon-blue"
              : color === "green"
                ? "bg-neon-green"
                : color === "pink"
                  ? "bg-neon-pink"
                  : color === "purple"
                    ? "bg-neon-purple"
                    : "bg-neon-orange"
          }
        `}
        />
      </CardContent>
    </Card>
  )
}
