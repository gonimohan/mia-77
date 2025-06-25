import { createClient } from "@supabase/supabase-js";
import { createBrowserClient as createSupabaseBrowserClient } from "@supabase/ssr"; // Renamed to avoid conflict if needed

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  console.warn("Supabase environment variables are not configured for the generic client or browser client");
}

// This is the existing generic client. We should aim to phase this out or use it carefully.
export const supabase = createClient(
  supabaseUrl || "https://placeholder.supabase.co",
  supabaseAnonKey || "placeholder-key",
);

// Helper function to check if Supabase is properly configured
export const isSupabaseConfigured = (): boolean => {
  return !!(
    supabaseUrl &&
    supabaseAnonKey &&
    supabaseUrl !== "https://placeholder.supabase.co" &&
    supabaseAnonKey !== "placeholder-key"
  );
};

// Server-side client for admin operations (remains unchanged)
export const createServerAdminClient = () => { // Renamed for clarity from createServerClient
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!supabaseUrl || !serviceRoleKey) {
    throw new Error("Missing Supabase server configuration for admin client");
  }
  return createClient(supabaseUrl, serviceRoleKey);
};

// New function to create a client-side Supabase client
export const createBrowserClient = () => {
  if (!supabaseUrl || !supabaseAnonKey) {
    // This case should ideally be handled by isSupabaseConfigured check before calling this
    console.error("Supabase URL or Anon Key is not defined for the browser client.");
    // Return a dummy client or throw an error, depending on desired strictness
    // For now, let's align with how the generic client handles placeholder values.
    return createSupabaseBrowserClient(
        "https://placeholder.supabase.co",
        "placeholder-key"
    );
  }
  return createSupabaseBrowserClient(supabaseUrl, supabaseAnonKey);
};
