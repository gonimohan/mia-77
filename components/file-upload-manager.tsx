"use client"

import React, { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { 
  Upload, 
  File, 
  FileText, 
  Table, 
  FileSpreadsheet, 
  Download,
  Trash2,
  Eye,
  BarChart3,
  Loader2,
  CheckCircle,
  AlertCircle,
  Search,
  Brain,
  TrendingUp
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { useToast } from "@/hooks/use-toast"
import { useAuth } from "@/components/auth-provider"
import { apiClient, ApiError, uploadFile, fetchFiles, fetchFileDetails, deleteFile, generateReport } from "@/lib/api"

interface UploadedFile {
  file_id: string
  filename: string
  file_type: string
  file_size: number
  processing_status: string
  upload_timestamp: string
  processed_data?: any
  ai_analysis?: any
}

interface FileUploadManagerProps {
  onFileAnalyzed?: (fileData: UploadedFile) => void
}

export function FileUploadManager({ onFileAnalyzed }: FileUploadManagerProps) {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isLoadingFiles, setIsLoadingFiles] = useState(false)
  const { toast } = useToast()
  const { user } = useAuth()

  const handleApiError = (error: any, context: string) => {
    console.error(`${context} error:`, error)
    let description = "An unexpected error occurred."
    if (error instanceof ApiError) {
      description = error.detail?.detail || error.message
    } else if (error instanceof Error) {
      description = error.message
    }
    toast({
      title: `${context} Failed`,
      description,
      variant: "destructive",
    })
  }

  const fetchUploadedFiles = useCallback(async () => {
    if (!user) return

    setIsLoadingFiles(true)
    try {
      const result = await fetchFiles()
      setUploadedFiles(result.files || [])
    } catch (error) {
      handleApiError(error, "Fetch Files")
    } finally {
      setIsLoadingFiles(false)
    }
  }, [user, toast])

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (!user) {
      toast({
        title: "Authentication Required",
        description: "Please log in to upload files.",
        variant: "destructive"
      })
      return
    }

    const file = acceptedFiles[0]
    if (!file) return

    const allowedTypes = ['.pdf', '.docx', '.txt', '.csv', '.xlsx']
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase()

    if (!allowedTypes.includes(fileExtension)) {
      toast({
        title: "Invalid File Type",
        description: `Please upload files with extensions: ${allowedTypes.join(', ')}`,
        variant: "destructive"
      })
      return
    }

    if (file.size > 50 * 1024 * 1024) {
      toast({
        title: "File Too Large",
        description: "Please upload files smaller than 50MB.",
        variant: "destructive"
      })
      return
    }

    setUploading(true)
    setUploadProgress(0)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('analysis_type', 'comprehensive')

      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90))
      }, 100)

      const result = await uploadFile(formData)

      clearInterval(progressInterval)
      setUploadProgress(100)

      toast({
        title: "Upload Initiated",
        description: `${result.original_filename} uploaded. Document ID: ${result.document_id}. Processing in background.`,
        duration: 5000
      })
      setLastUploadedDocId(result.document_id);
      
      // Refresh file list after a short delay to allow processing to start
      setTimeout(fetchUploadedFiles, 2000)

    } catch (error) {
      handleApiError(error, "Upload")
    } finally {
      setUploading(false)
      setUploadProgress(0)
    }
  }, [user, onFileAnalyzed, toast, fetchUploadedFiles])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
    multiple: false,
    disabled: uploading
  })

  // Helper to get file extension for icon mapping
  const getFileExtensionForIcon = (filenameOrPath: string) => {
    if (!filenameOrPath) return ''
    return ('.' + filenameOrPath.split('.').pop()?.toLowerCase()) || ''
  }

  const getFileIcon = (filenameOrPath: string) => { // Changed to take filename
    const fileTypeExt = getFileExtensionForIcon(filenameOrPath)
    switch (fileTypeExt) {
      case '.csv':
        return <Table className="w-5 h-5 text-neon-green" />
      case '.xlsx':
        // case '.xls': // .xls is not in our allowed list
        return <FileSpreadsheet className="w-5 h-5 text-neon-blue" />
      case '.pdf':
        return <File className="w-5 h-5 text-neon-pink" />
      case '.txt':
        return <FileText className="w-5 h-5 text-neon-orange" />
      case '.docx':
        return <FileText className="w-5 h-5 text-purple-400" /> // Example color for docx
      default:
        return <File className="w-5 h-5 text-gray-400" />
    }
  }

  const getStatusBadge = (status: string) => {
    // Adding new statuses from our MongoDB pipeline
    switch (status) {
      case 'completed': // This was from old Supabase logic, maps to 'analyzed' now
      case 'analyzed': // completed maps to analyzed
        return (
          <Badge className="bg-neon-green/20 text-neon-green border-neon-green/50">
            <CheckCircle className="w-3 h-3 mr-1" />
            Analyzed
          </Badge>
        )
      case 'processing':
      case 'uploaded': // Initial state, still processing
      case 'text_extracted': // Intermediate state, still processing
        return (
          <Badge className="bg-neon-blue/20 text-neon-blue border-neon-blue/50">
            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
            {status.charAt(0).toUpperCase() + status.slice(1).replace('_', ' ')}
          </Badge>
        )
      case 'error': // Generic error
      case 'extraction_failed':
      case 'processing_failed':
        return (
          <Badge className="bg-neon-pink/20 text-neon-pink border-neon-pink/50">
            <AlertCircle className="w-3 h-3 mr-1" />
            Error
          </Badge>
        )
      default:
        return (
          <Badge className="bg-gray-500/20 text-gray-400 border-gray-500/50">
            Unknown
          </Badge>
        )
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const handleViewFile = async (fileId: string) => {
    try {
      const fileDetails = await fetchFileDetails(fileId)
      toast({
        title: "File Details",
        description: `${fileDetails.filename} - ${formatFileSize(fileDetails.file_size)}`,
      })
    } catch (error) {
      handleApiError(error, "View File")
    }
  }

  const handleDeleteFile = async (fileId: string, filename: string) => {
    if (!confirm(`Are you sure you want to delete ${filename}?`)) return

    try {
      await deleteFile(fileId)
      toast({
        title: "Success",
        description: `${filename} has been deleted.`,
      })
      fetchUploadedFiles() // Refresh the list
    } catch (error) {
      handleApiError(error, "Delete File")
    }
  }

  const handleGenerateReport = async (documentId: string) => {
    if (!documentId) {
      toast({ title: "Error", description: "Document ID is missing.", variant: "destructive" });
      return;
    }
    try {
      const reportData = await generateReport(documentId)
      toast({
        title: `Report for ${reportData.original_filename || documentId}`,
        description: (
          <pre className="mt-2 w-[340px] rounded-md bg-slate-950 p-4 overflow-x-auto">
            <code className="text-white">{JSON.stringify(reportData, null, 2)}</code>
          </pre>
        ),
        duration: 10000
      });
    } catch (error) {
      handleApiError(error, "Generate Report")
    }
  }

  const [lastUploadedDocId, setLastUploadedDocId] = useState<string | null>(null);
  const dropzoneMessage = "Supports PDF, DOCX, TXT, CSV, XLSX (max 50MB)";

  React.useEffect(() => {
    if (user) {
      fetchUploadedFiles()
    }
  }, [user, fetchUploadedFiles])

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <Card className="bg-dark-card border-dark-border">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Upload className="w-5 h-5 text-neon-blue" />
            File Upload & Analysis
          </CardTitle>
          <CardDescription className="text-gray-400">
            Upload CSV, Excel, PDF, or text files for AI-powered market intelligence analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
              isDragActive
                ? 'border-neon-blue bg-neon-blue/10'
                : 'border-gray-600 hover:border-neon-blue/50 hover:bg-dark-bg/50'
            } ${uploading ? 'pointer-events-none opacity-50' : ''}`}
          >
            <input {...getInputProps()} />

            {uploading ? (
              <div className="space-y-4">
                <Loader2 className="w-12 h-12 text-neon-blue mx-auto animate-spin" />
                <div className="space-y-2">
                  <p className="text-white font-medium">Uploading and processing...</p>
                  <Progress value={uploadProgress} className="w-full max-w-xs mx-auto" />
                  <p className="text-sm text-gray-400">{uploadProgress}% complete</p>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <Upload className="w-12 h-12 text-gray-400 mx-auto" />
                <div>
                  <p className="text-white font-medium mb-2">
                    {isDragActive ? 'Drop your file here' : 'Drag & drop a file here, or click to select'}
                  </p>
                  <p className="text-sm text-gray-400">
                    {dropzoneMessage}
                  </p>
                </div>
                <Button
                  variant="outline"
                  className="border-neon-blue/50 text-neon-blue hover:bg-neon-blue/10"
                >
                  Choose File
                </Button>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Uploaded Files List */}
      <Card className="bg-dark-card border-dark-border">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-white flex items-center gap-2">
              <File className="w-5 h-5 text-neon-green" />
              Uploaded Files
            </CardTitle>
            <Button
              variant="outline"
              size="sm"
              onClick={fetchUploadedFiles}
              disabled={isLoadingFiles}
              className="border-gray-600 text-gray-400 hover:bg-dark-bg"
            >
              {isLoadingFiles ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                "Refresh"
              )}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {uploadedFiles.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <File className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p className="text-lg font-medium mb-2">No files uploaded yet</p>
              <p className="text-sm">Upload your first file to get started with AI analysis.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {uploadedFiles.map((file) => (
                <div
                  key={file.file_id}
                  className="flex items-center justify-between p-4 border border-dark-border rounded-lg bg-dark-bg/50 hover:bg-dark-bg/80 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    {getFileIcon(file.file_type)}
                    <div>
                      <h4 className="text-white font-medium">{file.filename}</h4>
                      <div className="flex items-center gap-4 text-sm text-gray-400">
                        <span>{formatFileSize(file.file_size)}</span>
                        <span>{new Date(file.upload_timestamp).toLocaleDateString()}</span>
                        {getStatusBadge(file.processing_status)}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-gray-400 hover:text-white"
                      onClick={() => handleViewFile(file.file_id)}
                      title="View Details"
                    >
                      <Eye className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-gray-400 hover:text-neon-blue"
                      onClick={() => handleGenerateReport(file.file_id)}
                      title="Generate Report"
                    >
                      <BarChart3 className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-gray-400 hover:text-neon-green"
                      title="Download"
                    >
                      <Download className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-gray-400 hover:text-neon-pink"
                      onClick={() => handleDeleteFile(file.file_id, file.filename)}
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}