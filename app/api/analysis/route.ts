import { type NextRequest, NextResponse } from "next/server"

const PYTHON_API_BASE_URL = process.env.PYTHON_AGENT_API_BASE_URL || "http://0.0.0.0:8000"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { query_str, market_domain_str, question_str } = body

    if (!query_str || !market_domain_str) {
      return NextResponse.json({ 
        error: "query_str and market_domain_str are required",
        details: "Missing required parameters"
      }, { status: 400 })
    }

    // Check if Python API is available
    try {
      const healthCheck = await fetch(`${PYTHON_API_BASE_URL}/health`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
        signal: AbortSignal.timeout(5000), // 5 second timeout
      })
      
      if (!healthCheck.ok) {
        throw new Error("Python API health check failed")
      }
    } catch (healthError) {
      console.error("Python API not available:", healthError)
      return NextResponse.json({ 
        error: "Backend service unavailable",
        details: "The analysis service is currently unavailable. Please try again later.",
        
      }, { status: 503 })
    }

    const response = await fetch(`${PYTHON_API_BASE_URL}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: query_str,
        market_domain: market_domain_str,
        question: question_str,
      }),
      signal: AbortSignal.timeout(30000), // 30 second timeout
    })

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
