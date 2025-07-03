import { type NextRequest, NextResponse } from "next/server"

const PYTHON_API_BASE_URL = process.env.PYTHON_AGENT_API_BASE_URL || "http://0.0.0.0:8000"

import { createRouteHandlerClient } from "@supabase/auth-helpers-nextjs";
import { cookies } from "next/headers";

export async function POST(request: NextRequest) {
  const supabase = createRouteHandlerClient({ cookies });
  const {
    data: { session },
    error: sessionError,
  } = await supabase.auth.getSession();

  if (sessionError || !session) {
    console.error("API Analysis Route: No session found or error fetching session", sessionError);
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }

  try {
    const body = await request.json();
    const { query_str, market_domain_str, question_str, uploaded_file_ids } = body; // Added uploaded_file_ids

    if (!query_str || !market_domain_str) {
      return NextResponse.json({ 
        error: "query_str and market_domain_str are required",
        details: "Missing required parameters for analysis"
      }, { status: 400 });
    }

    // Optional: Health check can remain but might be redundant if calls are frequent.
    // For now, removing to simplify, assuming backend is generally available if other parts work.
    // try { ... health check ... } catch { ... }

    const pythonRequestBody = {
      query: query_str,
      market_domain: market_domain_str,
      question: question_str,
      uploaded_file_ids: uploaded_file_ids || [], // Pass an empty list if undefined
    };

    const response = await fetch(`${PYTHON_API_BASE_URL}/api/analyze`, { // Corrected endpoint to /api/analyze
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${session.access_token}`, // Added Auth token
      },
      body: JSON.stringify(pythonRequestBody),
      // Consider a longer timeout for analysis, agent runs can take time
      signal: AbortSignal.timeout(120000), // 120 seconds timeout (2 minutes)
    });

    if (!response.ok) {
      const errorText = await response.text()
      console.error(`Python API error: ${response.status} - ${errorText}`)
      
      return NextResponse.json({ 
        error: "Analysis service error",
        details: `Service returned ${response.status}`,
        
      }, { status: 502 })
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Analysis API error:", error)
    
    // Return fallback data instead of just error
    return NextResponse.json({ 
      error: "Failed to process analysis request",
      details: error instanceof Error ? error.message : "Unknown error",
      
    }, { status: 500 })
  }
}

export async function GET() {
  return NextResponse.json({ 
    status: "Analysis API is running",
    timestamp: new Date().toISOString(),
    python_api_url: PYTHON_API_BASE_URL
  })
}
