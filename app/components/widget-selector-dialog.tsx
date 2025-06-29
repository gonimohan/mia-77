"use client"

import React, { useState, useEffect } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogClose,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { Settings2 } from "lucide-react"

export interface DashboardWidgetConfig {
  id: string;
  title: string;
  defaultEnabled: boolean;
}

interface WidgetSelectorDialogProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  currentEnabledWidgets: string[];
  onSave: (enabledWidgets: string[]) => void;
}

const allWidgets: DashboardWidgetConfig[] = [
  { id: "kpiCards", title: "Key Performance Indicators", defaultEnabled: true },
  { id: "trendImpactChart", title: "Identified Trends Impact", defaultEnabled: true },
  { id: "marketShareChart", title: "Market Share", defaultEnabled: true },
  { id: "competitorActivity", title: "Competitor Activity", defaultEnabled: true },
  { id: "developmentProgress", title: "Development Progress (Dev Only)", defaultEnabled: false },
  { id: "quickActions", title: "Quick Actions", defaultEnabled: true },
];

export const WidgetSelectorDialog: React.FC<WidgetSelectorDialogProps> = ({
  isOpen,
  onOpenChange,
  currentEnabledWidgets,
  onSave,
}) => {
  const [selectedWidgets, setSelectedWidgets] = useState<Set<string>>(new Set(currentEnabledWidgets));

  useEffect(() => {
    setSelectedWidgets(new Set(currentEnabledWidgets));
  }, [currentEnabledWidgets, isOpen]);

  const handleCheckboxChange = (widgetId: string, checked: boolean) => {
    setSelectedWidgets(prev => {
      const newSet = new Set(prev);
      if (checked) {
        newSet.add(widgetId);
      } else {
        newSet.delete(widgetId);
      }
      return newSet;
    });
  };

  const handleSave = () => {
    onSave(Array.from(selectedWidgets));
    onOpenChange(false);
  };

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="bg-dark-card border-dark-border text-white sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle className="text-neon-blue flex items-center gap-2">
            <Settings2 className="w-5 h-5" /> Customize Dashboard
          </DialogTitle>
          <DialogDescription className="text-gray-400">
            Select which sections you want to see on your dashboard.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          {allWidgets.map((widget) => (
            <div key={widget.id} className="flex items-center space-x-2">
              <Checkbox
                id={widget.id}
                checked={selectedWidgets.has(widget.id)}
                onCheckedChange={(checked) => handleCheckboxChange(widget.id, checked as boolean)}
                className="border-gray-500 data-[state=checked]:bg-neon-blue data-[state=checked]:text-white"
              />
              <Label htmlFor={widget.id} className="text-white cursor-pointer">
                {widget.title}
              </Label>
            </div>
          ))}
        </div>
        <DialogFooter>
          <DialogClose asChild>
            <Button variant="outline" className="border-gray-600 text-gray-300 hover:bg-dark-bg hover:text-white">
              Cancel
            </Button>
          </DialogClose>
          <Button onClick={handleSave} className="bg-neon-blue hover:bg-neon-blue/90 text-white">
            Save Changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
