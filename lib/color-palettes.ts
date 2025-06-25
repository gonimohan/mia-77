export interface ColorPalette {
  id: string
  name: string
  description: string
  colors: string[]
  preview: string[]
  category: "neon" | "electric" | "gradient" | "modern"
}

export const colorPalettes: ColorPalette[] = [
  {
    id: "neon-classic",
    name: "Neon Classic",
    description: "Original neon colors with electric vibes",
    colors: ["#00FFFF", "#39FF14", "#FF1493", "#BF00FF", "#FF6600"],
    preview: ["#00FFFF", "#39FF14", "#FF1493", "#BF00FF"],
    category: "neon",
  },
  {
    id: "electric-blue",
    name: "Electric Blue",
    description: "Cool blues and teals for a modern look",
    colors: ["#00D4FF", "#0099FF", "#3366FF", "#6633FF", "#9900FF"],
    preview: ["#00D4FF", "#0099FF", "#3366FF", "#6633FF"],
    category: "electric",
  },
  {
    id: "cyber-punk",
    name: "Cyber Punk",
    description: "Hot pinks and electric purples",
    colors: ["#FF0080", "#FF3399", "#CC00FF", "#9933FF", "#6600CC"],
    preview: ["#FF0080", "#FF3399", "#CC00FF", "#9933FF"],
    category: "neon",
  },
  {
    id: "neon-green",
    name: "Neon Green",
    description: "Vibrant greens and lime colors",
    colors: ["#39FF14", "#66FF33", "#99FF66", "#CCFF99", "#00FF80"],
    preview: ["#39FF14", "#66FF33", "#99FF66", "#CCFF99"],
    category: "neon",
  },
  {
    id: "electric-gradient",
    name: "Electric Gradient",
    description: "Smooth electric color transitions",
    colors: ["#FF006E", "#FB5607", "#FFBE0B", "#8338EC", "#3A86FF"],
    preview: ["#FF006E", "#FB5607", "#FFBE0B", "#8338EC"],
    category: "gradient",
  },
  {
    id: "aurora",
    name: "Aurora",
    description: "Northern lights inspired palette",
    colors: ["#00FFA3", "#03DAC6", "#BB86FC", "#CF6679", "#FF0266"],
    preview: ["#00FFA3", "#03DAC6", "#BB86FC", "#CF6679"],
    category: "gradient",
  },
  {
    id: "synthwave",
    name: "Synthwave",
    description: "Retro-futuristic 80s vibes",
    colors: ["#FF00FF", "#00FFFF", "#FFFF00", "#FF8000", "#8000FF"],
    preview: ["#FF00FF", "#00FFFF", "#FFFF00", "#FF8000"],
    category: "neon",
  },
  {
    id: "modern-dark",
    name: "Modern Dark",
    description: "Sophisticated dark theme colors",
    colors: ["#64FFDA", "#40C4FF", "#448AFF", "#7C4DFF", "#E040FB"],
    preview: ["#64FFDA", "#40C4FF", "#448AFF", "#7C4DFF"],
    category: "modern",
  },
  {
    id: "fire-ice",
    name: "Fire & Ice",
    description: "Hot reds and cool blues contrast",
    colors: ["#FF4081", "#FF6EC7", "#00E5FF", "#18FFFF", "#64FFDA"],
    preview: ["#FF4081", "#FF6EC7", "#00E5FF", "#18FFFF"],
    category: "electric",
  },
  {
    id: "matrix",
    name: "Matrix",
    description: "Green matrix-inspired colors",
    colors: ["#00FF41", "#39FF14", "#7FFF00", "#ADFF2F", "#32CD32"],
    preview: ["#00FF41", "#39FF14", "#7FFF00", "#ADFF2F"],
    category: "neon",
  },
]

export const defaultPalette = colorPalettes[0] // Neon Classic as default
