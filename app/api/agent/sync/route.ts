import { type NextRequest, NextResponse } from "next/server"

const PYTHON_API_BASE_URL = process.env.PYTHON_AGENT_API_BASE_URL || "http://0.0.0.0:8000"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { action, data } = body

    if (!action) {
      return NextResponse.json({ 
        error: "action parameter is required" 
      }, { status: 400 })
    }

    // Try to sync with Python agent
    const response = await fetch(`${PYTHON_API_BASE_URL}/agent/sync`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ action, data }),
      signal: AbortSignal.timeout(15000),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: `Python API responded with status: ${response.status}` }));
      return NextResponse.json({
        error: errorData.detail || `Python API responded with status: ${response.status}`,
        details: errorData.message || "No additional details."
      }, { status: response.status });
    }

    const responseData = await response.json()
    return NextResponse.json(responseData)
  } catch (error) {
    console.error("Agent sync API error:", error)
    return NextResponse.json({ 
      error: "Failed to sync with agent",
      details: error instanceof Error ? error.message : "Unknown error"
    }, { status: 500 })
  }
}

export async function GET() {
  try {
    // Check agent status
    const response = await fetch(`${PYTHON_API_BASE_URL}/agent/status`, {
      method: "GET",
      signal: AbortSignal.timeout(5000),
    })

    if (!response.ok) {
      throw new Error("Agent not available")
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json({
      status: "offline",
      message: "Python agent is currently unavailable",
      timestamp: new Date().toISOString()
    })
  }
}
