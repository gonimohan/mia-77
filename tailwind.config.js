/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./app/**/*.{ts,tsx}",
    "./src/**/*.{ts,tsx}",
    "*.{js,ts,jsx,tsx,mdx}",
  ],
  prefix: "",
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        // Custom neon colors
        "neon-blue": "#00FFFF",
        "neon-green": "#39FF14",
        "neon-pink": "#FF1493",
        "neon-purple": "#BF00FF",
        "neon-orange": "#FF6600",
        "dark-bg": "#1A1A1A",
        "dark-card": "#2C2C2C",
        "dark-border": "#404040",
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        "neon-pulse": {
          "0%, 100%": {
            boxShadow: "0 0 5px currentColor, 0 0 10px currentColor, 0 0 15px currentColor",
            textShadow: "0 0 5px currentColor",
          },
          "50%": {
            boxShadow: "0 0 10px currentColor, 0 0 20px currentColor, 0 0 30px currentColor",
            textShadow: "0 0 10px currentColor",
          },
        },
        glow: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.5" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "neon-pulse": "neon-pulse 2s ease-in-out infinite",
        glow: "glow 2s ease-in-out infinite",
      },
      boxShadow: {
        "neon-blue": "0 0 10px #00FFFF, 0 0 20px #00FFFF, 0 0 30px #00FFFF",
        "neon-green": "0 0 10px #39FF14, 0 0 20px #39FF14, 0 0 30px #39FF14",
        "neon-pink": "0 0 10px #FF1493, 0 0 20px #FF1493, 0 0 30px #FF1493",
        "neon-purple": "0 0 10px #BF00FF, 0 0 20px #BF00FF, 0 0 30px #BF00FF",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}
