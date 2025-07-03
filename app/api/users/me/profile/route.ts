import { createRouteHandlerClient } from "@supabase/auth-helpers-nextjs";
import { cookies } from "next/headers";
import { type NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const PYTHON_API_BASE_URL = process.env.PYTHON_AGENT_API_BASE_URL || "http://localhost:8000";

// GET handler for /api/users/me/profile
export async function GET(request: NextRequest) {
  const supabase = createRouteHandlerClient({ cookies });
  const {
    data: { session },
    error: sessionError,
  } = await supabase.auth.getSession();

  if (sessionError || !session) {
    console.error("API Profile GET Route: No session", sessionError);
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }

  try {
    const fetchUrl = `${PYTHON_API_BASE_URL}/api/users/me/profile`;
    const response = await fetch(fetchUrl, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${session.access_token}`,
        "Content-Type": "application/json",
      },
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: `Python API profile GET responded with status: ${response.status}` }));
      console.error(`Python API profile GET error: Status ${response.status}`, errorData);
      return NextResponse.json(
        { error: "Failed to fetch profile from backend", details: errorData.detail || response.statusText },
        { status: response.status }
      );
    }
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error("API Profile GET Route - Internal server error:", error);
    return NextResponse.json(
      { error: "Internal server error while fetching profile", details: error.message },
      { status: 500 }
    );
  }
}

// PUT handler for /api/users/me/profile
export async function PUT(request: NextRequest) {
  const supabase = createRouteHandlerClient({ cookies });
  const {
    data: { session },
    error: sessionError,
  } = await supabase.auth.getSession();

  if (sessionError || !session) {
    console.error("API Profile PUT Route: No session", sessionError);
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }

  try {
    const body = await request.json();
    const fetchUrl = `${PYTHON_API_BASE_URL}/api/users/me/profile`;
    const response = await fetch(fetchUrl, {
      method: "PUT",
      headers: {
        Authorization: `Bearer ${session.access_token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: `Python API profile PUT responded with status: ${response.status}` }));
      console.error(`Python API profile PUT error: Status ${response.status}`, errorData);
      return NextResponse.json(
        { error: "Failed to update profile on backend", details: errorData.detail || response.statusText },
        { status: response.status }
      );
    }
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error("API Profile PUT Route - Internal server error:", error);
    return NextResponse.json(
      { error: "Internal server error while updating profile", details: error.message },
      { status: 500 }
    );
  }
}
