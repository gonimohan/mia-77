import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { SidebarProvider } from "@/components/ui/sidebar"
import { AppSidebar } from "@/components/app-sidebar"
import { Toaster } from "@/components/ui/toaster"
import { ColorProvider } from "@/lib/color-context"
import { AuthProvider } from "@/components/auth-provider"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "Market Intel - Gen Z Dashboard",
  description: "Modern Market Intelligence Dashboard with neon aesthetics",
  generator: "v0.dev",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-dark-bg text-white antialiased`}>
        <ColorProvider>
          <AuthProvider>
            <SidebarProvider>
              <div className="flex min-h-screen w-full">
                <AppSidebar />
                <main className="flex-1 bg-dark-bg">{children}</main>
              </div>
            </SidebarProvider>
            <Toaster />
          </AuthProvider>
        </ColorProvider>
      </body>
    </html>
  )
}
