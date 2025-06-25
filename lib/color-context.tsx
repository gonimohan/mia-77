"use client"

import type React from "react"
import { createContext, useContext, useState, useEffect } from "react"
import { type ColorPalette, defaultPalette, colorPalettes } from "./color-palettes"

interface ColorContextType {
  currentPalette: ColorPalette
  setPalette: (palette: ColorPalette) => void
  getChartColors: () => string[]
  previewPalette: ColorPalette | null
  setPreviewPalette: (palette: ColorPalette | null) => void
}

const ColorContext = createContext<ColorContextType | undefined>(undefined)

export function ColorProvider({ children }: { children: React.ReactNode }) {
  const [currentPalette, setCurrentPalette] = useState<ColorPalette>(defaultPalette)
  const [previewPalette, setPreviewPalette] = useState<ColorPalette | null>(null)

  // Load saved palette from localStorage on mount
  useEffect(() => {
    const savedPaletteId = localStorage.getItem("dashboard-color-palette")
    if (savedPaletteId) {
      const savedPalette = colorPalettes.find((p) => p.id === savedPaletteId)
      if (savedPalette) {
        setCurrentPalette(savedPalette)
      }
    }
  }, [])

  const setPalette = (palette: ColorPalette) => {
    setCurrentPalette(palette)
    localStorage.setItem("dashboard-color-palette", palette.id)

    // Update CSS custom properties for dynamic theming
    const root = document.documentElement
    palette.colors.forEach((color, index) => {
      root.style.setProperty(`--chart-color-${index + 1}`, color)
    })
  }

  const getChartColors = () => {
    return (previewPalette || currentPalette).colors
  }

  // Initialize CSS custom properties
  useEffect(() => {
    const root = document.documentElement
    currentPalette.colors.forEach((color, index) => {
      root.style.setProperty(`--chart-color-${index + 1}`, color)
    })
  }, [currentPalette])

  return (
    <ColorContext.Provider
      value={{
        currentPalette,
        setPalette,
        getChartColors,
        previewPalette,
        setPreviewPalette,
      }}
    >
      {children}
    </ColorContext.Provider>
  )
}

export function useColorPalette() {
  const context = useContext(ColorContext)
  if (context === undefined) {
    throw new Error("useColorPalette must be used within a ColorProvider")
  }
  return context
}
