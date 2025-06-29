"use client"

import { createBrowserClient } from "@/lib/supabase"

export class ApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    public detail?: any
  ) {
    super(`API Error: ${status} ${statusText}`)
    this.name = "ApiError"
  }
}

class ApiClient {
  private baseUrl: string
  private supabase = createBrowserClient()

  constructor() {
    const pythonAgentApiBaseUrl =
      process.env.NEXT_PUBLIC_PYTHON_AGENT_API_BASE_URL
    if (!pythonAgentApiBaseUrl) {
      if (process.env.NODE_ENV === "development") {
        console.warn(
          "NEXT_PUBLIC_PYTHON_AGENT_API_BASE_URL is not set. Using default http://localhost:8000"
        )
        this.baseUrl = "http://localhost:8000"
      } else {
        throw new Error(
          "NEXT_PUBLIC_PYTHON_AGENT_API_BASE_URL is not set in production environment."
        )
      }
    } else {
      this.baseUrl = pythonAgentApiBaseUrl
    }
  }

  private async getAuthToken(): Promise<string | null> {
    const {
      data: { session },
    } = await this.supabase.auth.getSession()
    return session?.access_token || null
  }

  private async request(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<any> {
    const token = await this.getAuthToken()
    const headers = new Headers(options.headers || {})

    if (token) {
      headers.set("Authorization", `Bearer ${token}`)
    }

    // Do not set Content-Type for FormData, the browser does it with the correct boundary.
    if (!(options.body instanceof FormData)) {
      headers.set("Content-Type", "application/json")
    }

    const config: RequestInit = {
      ...options,
      headers,
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, config)

    if (!response.ok) {
      let errorDetail
      try {
        errorDetail = await response.json()
      } catch (e) {
        errorDetail = response.statusText
      }
      throw new ApiError(response.status, response.statusText, errorDetail)
    }

    // Handle cases with no content
    if (response.status === 204) {
      return null
    }

    return response.json()
  }

  async get(endpoint: string, options?: RequestInit): Promise<any> {
    return this.request(endpoint, { ...options, method: "GET" })
  }

  async post(
    endpoint: string,
    body: any,
    options?: RequestInit
  ): Promise<any> {
    const isFormData = body instanceof FormData
    return this.request(endpoint, {
      ...options,
      method: "POST",
      body: isFormData ? body : JSON.stringify(body),
    })
  }

  async put(endpoint: string, body: any, options?: RequestInit): Promise<any> {
    return this.request(endpoint, {
      ...options,
      method: "PUT",
      body: JSON.stringify(body),
    })
  }

  async delete(endpoint: string, options?: RequestInit): Promise<any> {
    return this.request(endpoint, { ...options, method: "DELETE" })
  }
}

export const apiClient = new ApiClient()

// Specific API functions

// Files
export const fetchFiles = () => apiClient.get("/api/files")
export const fetchFileDetails = (fileId: string) =>
  apiClient.get(`/api/files/${fileId}`)
export const uploadFile = (formData: FormData) =>
  apiClient.post("/api/upload", formData)
export const deleteFile = (fileId: string) =>
  apiClient.delete(`/api/files/${fileId}`)

// Agent
export const generateReport = (documentId: string) =>
  apiClient.get(`/api/agent/generate-report/${documentId}`)
export const getSyncStatus = () => apiClient.get("/api/agent/sync")
export const startSync = (data: {
  sources: string[]
  market_domain: string
  sync_type: string
}) => apiClient.post("/api/agent/sync", data)

// Data Sources / API Keys
// Assuming there are endpoints for these, placeholders for now
export const addApiKey = (data: any) => apiClient.post("/api/keys", data)
export const updateApiKey = (keyId: string, data: any) =>
  apiClient.put(`/api/keys/${keyId}`, data)
export const deleteApiKey = (keyId: string) =>
  apiClient.delete(`/api/keys/${keyId}`)
