import { type NextRequest, NextResponse } from "next/server"

const PYTHON_API_BASE_URL = process.env.PYTHON_AGENT_API_BASE_URL || "http://0.0.0.0:8000"

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const timeframe = searchParams.get('timeframe') || '30d'
    const category = searchParams.get('category') || 'all'
    const response = await fetch(`${PYTHON_API_BASE_URL}/kpi?timeframe=${timeframe}&category=${category}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
      signal: AbortSignal.timeout(10000),
    })
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: `Python API responded with status: ${response.status}` }));
      return NextResponse.json({
        error: errorData.detail || `Python API responded with status: ${response.status}`,
        details: errorData.message || "No additional details."
      }, { status: response.status });
    }
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("KPI API error:", error)
    return NextResponse.json({ 
      error: "Failed to fetch KPI data",
      details: error instanceof Error ? error.message : "Unknown error"
    }, { status: 500 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const response = await fetch(`${PYTHON_API_BASE_URL}/kpi`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(10000),
    })
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: `Python API responded with status: ${response.status}` }));
      return NextResponse.json({
        error: errorData.detail || `Python API responded with status: ${response.status}`,
        details: errorData.message || "No additional details."
      }, { status: response.status });
    }
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("KPI POST API error:", error)
    return NextResponse.json({ 
      error: "Failed to store KPI data",
      details: error instanceof Error ? error.message : "Unknown error"
    }, { status: 500 })
  }
}
