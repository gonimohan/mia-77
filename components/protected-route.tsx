"use client"

import type React from "react"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/components/auth-provider"

interface ProtectedRouteProps {
  children: React.ReactNode
  fallback?: React.ReactNode
}

export function ProtectedRoute({ children, fallback }: ProtectedRouteProps) {
  const { user, loading, isConfigured } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!loading && !user && isConfigured) {
      router.push("/login")
    }
  }, [user, loading, isConfigured, router])

  if (loading) {
    return (
      fallback || (
        <div className="flex items-center justify-center min-h-screen">
          <div className="flex items-center gap-2 text-white">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-purple-500 border-t-transparent" />
            <span>Loading...</span>
          </div>
        </div>
      )
    )
  }

  if (!isConfigured) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center text-white">
          <h2 className="text-xl font-semibold mb-2">Authentication Not Configured</h2>
          <p className="text-gray-400">Please configure Supabase environment variables.</p>
        </div>
      </div>
    )
  }

  if (!user) {
    return null // Will redirect to login
  }

  return <>{children}</>
}
