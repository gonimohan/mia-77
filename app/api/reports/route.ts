import { createRouteHandlerClient } from "@supabase/auth-helpers-nextjs";
import { cookies } from "next/headers";
import { type NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const PYTHON_API_BASE_URL = process.env.PYTHON_AGENT_API_BASE_URL || "http://localhost:8000";

export async function GET(request: NextRequest) {
  const supabase = createRouteHandlerClient({ cookies });
  const {
    data: { session },
    error: sessionError,
  } = await supabase.auth.getSession();

  if (sessionError || !session) {
    console.error("API Reports Route: No session found or error fetching session", sessionError);
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }

  try {
    const { searchParams } = new URL(request.url);
    const limit = searchParams.get("limit") || "50";
    const offset = searchParams.get("offset") || "0";

    const fetchUrl = `${PYTHON_API_BASE_URL}/api/reports?limit=${limit}&offset=${offset}`;

    const response = await fetch(fetchUrl, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${session.access_token}`,
        "Content-Type": "application/json",
      },
      signal: AbortSignal.timeout(15000),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: `Python API /api/reports responded with status: ${response.status}` }));
      console.error(`Python API /api/reports error: Status ${response.status}`, errorData);
      return NextResponse.json(
        { error: "Failed to fetch reports from backend", details: errorData.detail || response.statusText },
        { status: response.status }
      );
    }

    const data = await response.json();
    // The python backend returns report_data->>state_id. Rename it to state_id for easier frontend use.
    const formattedData = data.map((report: any) => ({
      ...report,
      state_id: report['report_data->>state_id'],
      report_db_id: report.id // Keep 'id' as report_db_id for clarity if needed, or just use 'id'
    }));

    return NextResponse.json(formattedData);
  } catch (error: any) {
    console.error("API Reports Route - Internal server error:", error);
    return NextResponse.json(
      { error: "Internal server error while fetching reports", details: error.message },
      { status: 500 }
    );
  }
}
