import { createRouteHandlerClient } from "@supabase/auth-helpers-nextjs";
import { cookies } from "next/headers";
import { type NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic"; // Ensure fresh data on each request

const PYTHON_API_BASE_URL = process.env.PYTHON_AGENT_API_BASE_URL || "http://localhost:8000";

export async function GET(request: NextRequest) {
  const supabase = createRouteHandlerClient({ cookies });
  const {
    data: { session },
    error: sessionError,
  } = await supabase.auth.getSession();

  if (sessionError || !session) {
    console.error("API Trends Route: No session found or error fetching session", sessionError);
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }

  try {
    // Forward any query parameters from the original request, e.g., analysis_id
    const { searchParams } = new URL(request.url);
    const analysisId = searchParams.get("analysis_id");

    let fetchUrl = `${PYTHON_API_BASE_URL}/api/trends`;
    if (analysisId) {
      fetchUrl += `?analysis_id=${encodeURIComponent(analysisId)}`;
    }

    const response = await fetch(fetchUrl, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${session.access_token}`,
        "Content-Type": "application/json",
      },
      signal: AbortSignal.timeout(15000), // 15 second timeout
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: `Python API /api/trends responded with status: ${response.status}` }));
      console.error(`Python API /api/trends error: Status ${response.status}`, errorData);
      return NextResponse.json(
        { error: "Failed to fetch trends data from backend", details: errorData.detail || response.statusText },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error("API Trends Route - Internal server error:", error);
    return NextResponse.json(
      { error: "Internal server error while fetching trends", details: error.message },
      { status: 500 }
    );
  }
}
