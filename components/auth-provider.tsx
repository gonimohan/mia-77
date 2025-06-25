"use client"

import type React from "react"
import { createContext, useContext, useEffect, useState } from "react"
import type { User, SupabaseClient } from "@supabase/supabase-js" // Import SupabaseClient
import { createBrowserClient, isSupabaseConfigured } from "@/lib/supabase" // Updated import

interface AuthContextType {
  user: User | null
  loading: boolean
  signOut: () => Promise<void>
  isConfigured: boolean
  error: string | null
  supabaseClient: SupabaseClient | null; // Add supabaseClient to context
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  loading: true,
  signOut: async () => {},
  isConfigured: false,
  error: null,
  supabaseClient: null, // Initialize with null
})

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const configured = isSupabaseConfigured()
  // Initialize the client-side Supabase client here
  const [supabaseClient, setSupabaseClient] = useState<SupabaseClient | null>(null);

  useEffect(() => {
    if (configured) {
      setSupabaseClient(createBrowserClient());
    }
  }, [configured]);

  useEffect(() => {
    if (!configured) {
      setLoading(false)
      setError("Supabase is not configured")
      return
    }

    if (!supabaseClient) {
      // Wait for supabaseClient to be initialized
      if (!loading) setLoading(true); // Show loading if not already
      return;
    }

    // Get initial session
    const getInitialSession = async () => {
      try {
        const {
          data: { session },
          error: sessionError, // Renamed to avoid conflict with outer error
        } = await supabaseClient.auth.getSession()
        if (sessionError) {
          console.error("Error getting session:", sessionError)
          setError(sessionError.message)
        } else {
          setUser(session?.user ?? null)
          setError(null)
        }
      } catch (catchError) { // Renamed to avoid conflict
        console.error("Error in getInitialSession:", catchError)
        setError("Failed to initialize authentication")
      } finally {
        setLoading(false)
      }
    }

    getInitialSession()

    // Listen for auth changes
    const {
      data: { subscription },
    } = supabaseClient.auth.onAuthStateChange(async (event, session) => {
      console.log("Auth state changed:", event, session?.user?.email)
      setUser(session?.user ?? null)
      setLoading(false) // Ensure loading is false after auth state change
      setError(null)
    })

    return () => subscription?.unsubscribe() // Add null check for subscription
  }, [supabaseClient]) // Add supabaseClient and loading to dependency array

  const signOut = async () => {
    if (!configured || !supabaseClient) return

    try {
      const { error: signOutError } = await supabaseClient.auth.signOut() // Renamed
      if (signOutError) {
        console.error("Error signing out:", signOutError)
        setError(signOutError.message)
      } else {
        setUser(null); // Explicitly set user to null on sign out
      }
    } catch (catchError) { // Renamed
      console.error("Error signing out:", catchError)
      setError("Failed to sign out")
    }
  }

  return (
    <AuthContext.Provider value={{ user, loading, signOut, isConfigured: configured, error, supabaseClient }}>
      {children}
    </AuthContext.Provider>
  )
}
