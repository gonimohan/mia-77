import { createRouteHandlerClient } from "@supabase/auth-helpers-nextjs";
import { cookies } from "next/headers";
import { type NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const PYTHON_API_BASE_URL = process.env.PYTHON_AGENT_API_BASE_URL || "http://localhost:8000";

interface Context {
  params: { state_id: string };
}

export async function GET(request: NextRequest, context: Context) {
  const supabase = createRouteHandlerClient({ cookies });
  const {
    data: { session },
    error: sessionError,
  } = await supabase.auth.getSession();

  if (sessionError || !session) {
    console.error("API Downloads-Info Route: No session found", sessionError);
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }

  const { state_id } = context.params;
  if (!state_id) {
    return NextResponse.json({ error: "State ID is required" }, { status: 400 });
  }

  try {
    const fetchUrl = `${PYTHON_API_BASE_URL}/api/analysis-states/${state_id}/downloads-info`;

    const response = await fetch(fetchUrl, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${session.access_token}`,
        "Content-Type": "application/json",
      },
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: `Python API downloads-info responded with status: ${response.status}` }));
      console.error(`Python API downloads-info error for state ${state_id}: Status ${response.status}`, errorData);
      return NextResponse.json(
        { error: "Failed to fetch downloads info from backend", details: errorData.detail || response.statusText },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error(`API Downloads-Info Route - Internal server error for state ${state_id}:`, error);
    return NextResponse.json(
      { error: "Internal server error while fetching downloads info", details: error.message },
      { status: 500 }
    );
  }
}
