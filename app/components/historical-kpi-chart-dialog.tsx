"use client"

import React from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts"
import { useColorPalette } from "@/lib/color-context"

interface HistoricalKpiChartDialogProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  kpiTitle: string;
  historicalData: { date: string; value: number }[];
  unit: string;
}

export const HistoricalKpiChartDialog: React.FC<HistoricalKpiChartDialogProps> = ({
  isOpen,
  onOpenChange,
  kpiTitle,
  historicalData,
  unit,
}) => {
  const { getChartColors } = useColorPalette();
  const chartColors = getChartColors();

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="bg-dark-card border-dark-border text-white sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle className="text-neon-blue">Historical Data: {kpiTitle}</DialogTitle>
          <DialogDescription className="text-gray-400">
            Trend of {kpiTitle} over time.
          </DialogDescription>
        </DialogHeader>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={historicalData}
              margin={{
                top: 10,
                right: 30,
                left: 0,
                bottom: 0,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#404040" />
              <XAxis dataKey="date" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" label={{ value: unit, angle: -90, position: 'insideLeft', fill: '#9CA3AF' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#2C2C2C",
                  border: "1px solid #404040",
                  borderRadius: "8px",
                  color: "#fff",
                }}
                formatter={(value: number) => [`${value.toLocaleString()} ${unit}`, "Value"]}
              />
              <Line type="monotone" dataKey="value" stroke={chartColors[0]} strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </DialogContent>
    </Dialog>
  );
};
