import { createRouteHandlerClient } from "@supabase/auth-helpers-nextjs";
import { cookies } from "next/headers";
import { type NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const PYTHON_API_BASE_URL = process.env.PYTHON_AGENT_API_BASE_URL || "http://localhost:8000";

interface Context {
  params: {
    state_id: string;
    file_identifier: string;
  };
}

export async function GET(request: NextRequest, context: Context) {
  const supabase = createRouteHandlerClient({ cookies });
  const {
    data: { session },
    error: sessionError,
  } = await supabase.auth.getSession();

  if (sessionError || !session) {
    console.error("API Download-File Route: No session found", sessionError);
    // For file downloads, returning JSON error might not be ideal, but it's consistent.
    // Client might need to handle this if it expects a file stream.
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }

  const { state_id, file_identifier } = context.params;
  if (!state_id || !file_identifier) {
    return NextResponse.json({ error: "State ID and File Identifier are required" }, { status: 400 });
  }

  try {
    const fetchUrl = `${PYTHON_API_BASE_URL}/api/analysis-states/${state_id}/download-file/${encodeURIComponent(file_identifier)}`;

    const pythonResponse = await fetch(fetchUrl, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${session.access_token}`,
        // No 'Content-Type': 'application/json' as we expect a file stream
      },
      signal: AbortSignal.timeout(30000), // Timeout for file download
    });

    if (!pythonResponse.ok) {
      const errorData = await pythonResponse.json().catch(() => ({ detail: `Python API download-file responded with status: ${pythonResponse.status}` }));
      console.error(`Python API download-file error for state ${state_id}, file ${file_identifier}: Status ${pythonResponse.status}`, errorData);
      return NextResponse.json(
        { error: "Failed to download file from backend", details: errorData.detail || pythonResponse.statusText },
        { status: pythonResponse.status }
      );
    }

    // Stream the response from the Python backend
    const blob = await pythonResponse.blob();
    const headers = new Headers();
    // Copy relevant headers from the Python response for the client
    headers.set('Content-Type', pythonResponse.headers.get('Content-Type') || 'application/octet-stream');
    headers.set('Content-Disposition', pythonResponse.headers.get('Content-Disposition') || `attachment; filename="${file_identifier}"`);

    return new NextResponse(blob, { status: 200, statusText: "OK", headers });

  } catch (error: any) {
    console.error(`API Download-File Route - Internal server error for state ${state_id}, file ${file_identifier}:`, error);
    return NextResponse.json(
      { error: "Internal server error while downloading file", details: error.message },
      { status: 500 }
    );
  }
}
