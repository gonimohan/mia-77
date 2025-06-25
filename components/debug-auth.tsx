"use client"

import { useAuth } from "./auth-provider"
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"

export function DebugAuth() {
  const { user, loading, isConfigured, error } = useAuth()

  if (process.env.NODE_ENV !== "development") {
    return null
  }

  return (
    <Card className="fixed bottom-4 right-4 w-80 bg-dark-card border-dark-border z-50">
      <CardHeader>
        <CardTitle className="text-white text-sm">Auth Debug</CardTitle>
      </CardHeader>
      <CardContent className="text-xs text-gray-400 space-y-2">
        <div>Configured: {isConfigured ? "✅" : "❌"}</div>
        <div>Loading: {loading ? "⏳" : "✅"}</div>
        <div>User: {user ? `✅ ${user.email}` : "❌ Not logged in"}</div>
        <div>Error: {error || "None"}</div>
        <div>Supabase URL: {process.env.NEXT_PUBLIC_SUPABASE_URL ? "✅" : "❌"}</div>
        <div>Supabase Key: {process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ? "✅" : "❌"}</div>
      </CardContent>
    </Card>
  )
}
