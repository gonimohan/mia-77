"use client"

import { useState } from "react"
import { Check, Palette, Sparkles } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useColorPalette } from "@/lib/color-context"
import { colorPalettes, type ColorPalette } from "@/lib/color-palettes"

interface ColorPreviewProps {
  palette: ColorPalette
  isSelected: boolean
  isPreview: boolean
  onSelect: () => void
  onPreview: () => void
  onPreviewEnd: () => void
}

function ColorPreview({ palette, isSelected, isPreview, onSelect, onPreview, onPreviewEnd }: ColorPreviewProps) {
  return (
    <Card
      className={`
        bg-dark-card border-dark-border cursor-pointer transition-all duration-300 group
        ${isSelected ? "border-neon-blue border-2 shadow-neon-blue/50 shadow-lg" : "hover:border-gray-500"}
        ${isPreview ? "ring-2 ring-neon-pink/50" : ""}
      `}
      onMouseEnter={onPreview}
      onMouseLeave={onPreviewEnd}
      onClick={onSelect}
    >
      <CardContent className="p-4">
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-white text-sm">{palette.name}</h3>
            {isSelected && <Check className="w-4 h-4 text-neon-blue" />}
          </div>

          <div className="flex gap-1">
            {palette.preview.map((color, index) => (
              <div
                key={index}
                className="w-8 h-8 rounded-lg border border-gray-600 transition-transform group-hover:scale-110"
                style={{ backgroundColor: color }}
              />
            ))}
          </div>

          <div className="space-y-1">
            <Badge
              variant="outline"
              className={`
                text-xs border-gray-600 
                ${
                  palette.category === "neon"
                    ? "text-neon-green"
                    : palette.category === "electric"
                      ? "text-neon-blue"
                      : palette.category === "gradient"
                        ? "text-neon-pink"
                        : "text-neon-purple"
                }
              `}
            >
              {palette.category.toUpperCase()}
            </Badge>
            <p className="text-xs text-gray-400 line-clamp-2">{palette.description}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function ColorPaletteSelector() {
  const { currentPalette, setPalette, setPreviewPalette } = useColorPalette()
  const [selectedCategory, setSelectedCategory] = useState<string>("all")

  const categories = ["all", "neon", "electric", "gradient", "modern"]

  const filteredPalettes =
    selectedCategory === "all" ? colorPalettes : colorPalettes.filter((p) => p.category === selectedCategory)

  const handlePaletteSelect = (palette: ColorPalette) => {
    setPalette(palette)
    setPreviewPalette(null)
  }

  const handlePreview = (palette: ColorPalette) => {
    if (palette.id !== currentPalette.id) {
      setPreviewPalette(palette)
    }
  }

  const handlePreviewEnd = () => {
    setPreviewPalette(null)
  }

  return (
    <Card className="bg-dark-card border-dark-border">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <Palette className="w-5 h-5 text-neon-blue" />
          Color Palette
        </CardTitle>
        <CardDescription className="text-gray-400">
          Customize the color scheme for all charts and graphs
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Current Palette Display */}
        <div className="p-4 rounded-lg bg-dark-bg border border-dark-border">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h4 className="text-white font-medium">Current Palette</h4>
              <p className="text-sm text-gray-400">{currentPalette.name}</p>
            </div>
            <Sparkles className="w-5 h-5 text-neon-blue" />
          </div>
          <div className="flex gap-2">
            {currentPalette.colors.map((color, index) => (
              <div
                key={index}
                className="w-10 h-10 rounded-lg border-2 border-gray-600 shadow-lg"
                style={{ backgroundColor: color }}
              />
            ))}
          </div>
        </div>

        {/* Category Tabs */}
        <Tabs value={selectedCategory} onValueChange={setSelectedCategory}>
          <TabsList className="grid w-full grid-cols-5 bg-dark-bg border border-dark-border">
            {categories.map((category) => (
              <TabsTrigger
                key={category}
                value={category}
                className="data-[state=active]:bg-neon-blue/20 data-[state=active]:text-neon-blue text-gray-400"
              >
                {category.charAt(0).toUpperCase() + category.slice(1)}
              </TabsTrigger>
            ))}
          </TabsList>

          <TabsContent value={selectedCategory} className="mt-4">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {filteredPalettes.map((palette) => (
                <ColorPreview
                  key={palette.id}
                  palette={palette}
                  isSelected={palette.id === currentPalette.id}
                  isPreview={false}
                  onSelect={() => handlePaletteSelect(palette)}
                  onPreview={() => handlePreview(palette)}
                  onPreviewEnd={handlePreviewEnd}
                />
              ))}
            </div>
          </TabsContent>
        </Tabs>

        {/* Reset Button */}
        <div className="flex justify-end pt-4 border-t border-dark-border">
          <Button
            variant="outline"
            onClick={() => handlePaletteSelect(colorPalettes[0])}
            className="border-gray-600 text-gray-400 hover:bg-dark-bg hover:text-white"
          >
            Reset to Default
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
