"use client"

import { useState } from "react"
import { Key, Eye, EyeOff, Copy, Check, RefreshCw, AlertTriangle, Loader2, Edit, Trash2, Plus, Play, Pause } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useToast } from "@/hooks/use-toast"

// Define DataSourceFromParent based on app/data-integration/page.tsx state
interface DataSourceFromParent {
  id: string;
  name: string;
  type: string;
  config: Record<string, any>;
  status?: string;
  category?: string;
  description?: string;
  last_sync?: string | null;
}

interface APIKeyManagerProps {
  dataSources: DataSourceFromParent[];
  onTestConnection: (sourceId: string) => void;
  onEditSource: (sourceId: string) => void;
  onRefreshData: () => void;
  isLoadingTest?: string | null;
}

interface ProcessedKey {
  id: string;
  name: string;
  service: string;
  key: string;
  status: "active" | "inactive" | "error" | "unknown";
  lastUsed: string;
  category: "news" | "financial" | "search" | "ai" | "general";
  description: string;
  endpoint: string;
}

interface APIKeyFormData {
  name: string;
  service: string;
  key: string;
  category: string;
  description: string;
  endpoint: string;
}

export function APIKeyManager({ 
  dataSources, 
  onTestConnection, 
  onEditSource, 
  onRefreshData,
  isLoadingTest 
}: APIKeyManagerProps) {
  const [visibleKeys, setVisibleKeys] = useState<Set<string>>(new Set())
  const [copiedKey, setCopiedKey] = useState<string | null>(null)
  const [selectedCategory, setSelectedCategory] = useState<string>("all")
  const [isDialogOpen, setIsDialogOpen] = useState(false)
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false)
  const [editingKey, setEditingKey] = useState<ProcessedKey | null>(null)
  const [deletingKey, setDeletingKey] = useState<ProcessedKey | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  
  const [formData, setFormData] = useState<APIKeyFormData>({
    name: "",
    service: "",
    key: "",
    category: "",
    description: "",
    endpoint: ""
  })
  
  const { toast } = useToast()

  const processedApiKeys: ProcessedKey[] = dataSources.map(ds => {
    const dsCategory = ds.category?.toLowerCase() || ds.type?.toLowerCase() || "general";
    let mappedCategory: ProcessedKey['category'] = "general";
    if (["news", "financial", "search", "ai"].includes(dsCategory)) {
      mappedCategory = dsCategory as ProcessedKey['category'];
    } else if (dsCategory.includes("news")) mappedCategory = "news";
    else if (dsCategory.includes("financ")) mappedCategory = "financial";
    else if (dsCategory.includes("search") || dsCategory.includes("tavily") || dsCategory.includes("serp")) mappedCategory = "search";
    else if (dsCategory.includes("ai") || dsCategory.includes("gemini") || dsCategory.includes("google")) mappedCategory = "ai";

    let currentStatus: ProcessedKey['status'] = "unknown";
    if (ds.status && ["active", "inactive", "error"].includes(ds.status.toLowerCase())) {
      currentStatus = ds.status.toLowerCase() as ProcessedKey['status'];
    }

    return {
      id: ds.id,
      name: ds.name,
      service: ds.config?.endpoint || ds.type || "N/A",
      key: ds.config?.apiKey || "Not Configured",
      status: currentStatus,
      lastUsed: ds.last_sync || "Never",
      category: mappedCategory,
      description: ds.description || "No description available.",
      endpoint: ds.config?.endpoint || "N/A",
    };
  });

  const toggleKeyVisibility = (keyId: string) => {
    const newVisible = new Set(visibleKeys)
    if (newVisible.has(keyId)) {
      newVisible.delete(keyId)
    } else {
      newVisible.add(keyId)
    }
    setVisibleKeys(newVisible)
  }

  const copyToClipboard = async (key: string, keyId: string) => {
    try {
      await navigator.clipboard.writeText(key)
      setCopiedKey(keyId)
      toast({
        title: "API Key Copied",
        description: "The API key has been copied to your clipboard.",
      })
      setTimeout(() => setCopiedKey(null), 2000)
    } catch (err) {
      toast({
        title: "Copy Failed",
        description: "Failed to copy API key to clipboard.",
        variant: "destructive",
      })
    }
  }

  const getStatusBadge = (status: ProcessedKey['status']) => {
    const variants = {
      active: "bg-neon-green/20 text-neon-green border-neon-green/50",
      inactive: "bg-gray-500/20 text-gray-400 border-gray-500/50",
      error: "bg-neon-pink/20 text-neon-pink border-neon-pink/50",
      unknown: "bg-yellow-500/20 text-yellow-400 border-yellow-500/50",
    };
    const currentStatusStyle = variants[status] || variants.unknown;

    return (
      <Badge className={`${currentStatusStyle} border text-xs`}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </Badge>
    );
  }

  const getCategoryColor = (category: ProcessedKey['category']) => {
    const colors = {
      news: "text-neon-blue",
      financial: "text-neon-green",
      search: "text-neon-orange",
      ai: "text-neon-purple",
      general: "text-gray-400",
    };
    return colors[category] || "text-gray-400";
  }

  const handleOpenAddDialog = () => {
    setEditingKey(null)
    setFormData({
      name: "",
      service: "",
      key: "",
      category: "",
      description: "",
      endpoint: ""
    })
    setIsDialogOpen(true)
  }

  const handleOpenEditDialog = (apiKey: ProcessedKey) => {
    setEditingKey(apiKey)
    setFormData({
      name: apiKey.name,
      service: apiKey.service,
      key: "", // Keep blank for security
      category: apiKey.category,
      description: apiKey.description,
      endpoint: apiKey.endpoint
    })
    setIsDialogOpen(true)
  }

  const handleOpenDeleteDialog = (apiKey: ProcessedKey) => {
    setDeletingKey(apiKey)
    setIsDeleteDialogOpen(true)
  }

  const handleSubmit = async () => {
    if (!formData.name.trim() || !formData.category) {
      toast({
        title: "Validation Error",
        description: "Name and category are required fields.",
        variant: "destructive",
      })
      return
    }

    setIsSubmitting(true)
    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      toast({
        title: "Success",
        description: `API key ${editingKey ? 'updated' : 'added'} successfully.`,
      })
      
      setIsDialogOpen(false)
      onRefreshData() // Refresh the data sources
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to save API key. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleDelete = async () => {
    if (!deletingKey) return
    
    setIsSubmitting(true)
    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 500))
      
      toast({
        title: "Success",
        description: "API key deleted successfully.",
      })
      
      setIsDeleteDialogOpen(false)
      setDeletingKey(null)
      onRefreshData() // Refresh the data sources
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to delete API key. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleToggleStatus = async (apiKey: ProcessedKey) => {
    try {
      const newStatus = apiKey.status === "active" ? "inactive" : "active"
      
      toast({
        title: "Status Updated",
        description: `API key ${newStatus === "active" ? "activated" : "paused"}.`,
      })
      
      onRefreshData() // Refresh the data sources
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to update status. Please try again.",
        variant: "destructive",
      })
    }
  }

  const filteredKeys = selectedCategory === "all"
    ? processedApiKeys
    : processedApiKeys.filter((key) => key.category === selectedCategory);

  const categories = ["all", "news", "financial", "search", "ai", "general"];

  return (
    <Card className="bg-dark-card border-dark-border">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-white flex items-center gap-2">
              <Key className="w-5 h-5 text-neon-green" />
              API Key Management
            </CardTitle>
            <CardDescription className="text-gray-400">
              Manage API keys for real-time data collection and AI analysis
            </CardDescription>
          </div>
          <Button
            onClick={handleOpenAddDialog}
            className="bg-neon-green/20 border border-neon-green/50 text-neon-green hover:bg-neon-green/30 hover:shadow-neon-green/50 hover:shadow-lg transition-all duration-300"
          >
            <Plus className="w-4 h-4 mr-2" />
            Add New Key
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs value={selectedCategory} onValueChange={setSelectedCategory}>
          <TabsList className="grid w-full grid-cols-6 bg-dark-bg border border-dark-border mb-6">
            {categories.map((category) => (
              <TabsTrigger
                key={category}
                value={category}
                className="data-[state=active]:bg-neon-blue/20 data-[state=active]:text-neon-blue text-gray-400"
              >
                {category.charAt(0).toUpperCase() + category.slice(1)}
              </TabsTrigger>
            ))}
          </TabsList>

          <TabsContent value={selectedCategory} className="space-y-4">
            {/* Enhanced Table View */}
            <div className="rounded-md border border-dark-border overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow className="border-dark-border hover:bg-dark-bg/50">
                    <TableHead className="text-gray-300">API Key Name</TableHead>
                    <TableHead className="text-gray-300">Service</TableHead>
                    <TableHead className="text-gray-300">Category</TableHead>
                    <TableHead className="text-gray-300">Key</TableHead>
                    <TableHead className="text-gray-300">Status</TableHead>
                    <TableHead className="text-gray-300">Last Used</TableHead>
                    <TableHead className="text-gray-300 text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredKeys.map((apiKey) => (
                    <TableRow 
                      key={apiKey.id} 
                      className="border-dark-border hover:bg-dark-bg/30 transition-colors"
                    >
                      <TableCell className="font-medium text-white">
                        <div>
                          <div className="font-semibold">{apiKey.name}</div>
                          <div className="text-xs text-gray-400 mt-1">{apiKey.description}</div>
                        </div>
                      </TableCell>
                      <TableCell className="text-gray-300">
                        <div className="text-sm">{apiKey.service}</div>
                        <div className="text-xs text-gray-500 break-all">{apiKey.endpoint}</div>
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline" className={`text-xs ${getCategoryColor(apiKey.category)}`}>
                          {apiKey.category.toUpperCase()}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2 max-w-[200px]">
                          <Input
                            type={visibleKeys.has(apiKey.id) ? "text" : "password"}
                            value={apiKey.key}
                            readOnly
                            className="bg-dark-bg border-dark-border text-white text-xs h-8"
                          />
                          <div className="flex gap-1">
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6 text-gray-400 hover:text-white"
                              onClick={() => toggleKeyVisibility(apiKey.id)}
                            >
                              {visibleKeys.has(apiKey.id) ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6 text-gray-400 hover:text-white"
                              onClick={() => copyToClipboard(apiKey.key, apiKey.id)}
                            >
                              {copiedKey === apiKey.id ? (
                                <Check className="w-3 h-3 text-neon-green" />
                              ) : (
                                <Copy className="w-3 h-3" />
                              )}
                            </Button>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          {getStatusBadge(apiKey.status)}
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 text-gray-400 hover:text-white"
                            onClick={() => handleToggleStatus(apiKey)}
                            title={apiKey.status === "active" ? "Pause" : "Resume"}
                          >
                            {apiKey.status === "active" ? 
                              <Pause className="w-3 h-3" /> : 
                              <Play className="w-3 h-3" />
                            }
                          </Button>
                        </div>
                      </TableCell>
                      <TableCell className="text-gray-400 text-sm">
                        {apiKey.lastUsed === "Never" ? "Never" : new Date(apiKey.lastUsed).toLocaleDateString()}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex gap-1 justify-end">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => onTestConnection(apiKey.id)}
                            className="border-neon-blue/50 text-neon-blue hover:bg-neon-blue/10 h-7 px-2"
                            disabled={isLoadingTest === apiKey.id}
                          >
                            {isLoadingTest === apiKey.id ? (
                              <Loader2 className="w-3 h-3 animate-spin" />
                            ) : (
                              <RefreshCw className="w-3 h-3" />
                            )}
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            className="border-gray-600 text-gray-400 hover:bg-dark-bg h-7 px-2"
                            onClick={() => handleOpenEditDialog(apiKey)}
                          >
                            <Edit className="w-3 h-3" />
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            className="border-neon-pink/50 text-neon-pink hover:bg-neon-pink/10 h-7 px-2"
                            onClick={() => handleOpenDeleteDialog(apiKey)}
                          >
                            <Trash2 className="w-3 h-3" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              
              {filteredKeys.length === 0 && (
                <div className="p-8 text-center text-gray-400">
                  <Key className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p className="text-lg font-medium mb-2">No API keys found</p>
                  <p className="text-sm">Add your first API key to get started with data integration.</p>
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>

        {/* Usage Statistics */}
        <div className="mt-6 p-4 border border-dark-border rounded-lg bg-dark-bg/30">
          <h4 className="text-white font-medium mb-3 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-neon-orange" />
            Usage Statistics
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-center">
            <div>
              <p className="text-2xl font-bold text-neon-green">
                {processedApiKeys.filter((k) => k.status === "active").length}
              </p>
              <p className="text-xs text-gray-400">Active Keys</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-400">
                {processedApiKeys.filter((k) => k.status === "inactive").length}
              </p>
              <p className="text-xs text-gray-400">Inactive Keys</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-neon-pink">
                {processedApiKeys.filter((k) => k.status === "error").length}
              </p>
              <p className="text-xs text-gray-400">Error Keys</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-neon-blue">
                {processedApiKeys.length}
              </p>
              <p className="text-xs text-gray-400">Total Keys</p>
            </div>
          </div>
        </div>
      </CardContent>

      {/* Add/Edit Dialog */}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="bg-dark-card border-dark-border text-white sm:max-w-[525px]">
          <DialogHeader>
            <DialogTitle className="text-neon-green">
              {editingKey ? "Edit API Key" : "Add New API Key"}
            </DialogTitle>
            <DialogDescription>
              {editingKey ? "Update the details of your API key." : "Configure a new API key for data integration."}
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="name">API Key Name *</Label>
              <Input
                id="name"
                value={formData.name}
                onChange={(e) => setFormData({...formData, name: e.target.value})}
                placeholder="e.g., News API Key"
                className="bg-dark-bg border-dark-border"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="service">Service/Provider</Label>
              <Input
                id="service"
                value={formData.service}
                onChange={(e) => setFormData({...formData, service: e.target.value})}
                placeholder="e.g., NewsAPI, Alpha Vantage"
                className="bg-dark-bg border-dark-border"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="category">Category *</Label>
              <Select value={formData.category} onValueChange={(value) => setFormData({...formData, category: value})}>
                <SelectTrigger className="bg-dark-bg border-dark-border">
                  <SelectValue placeholder="Select a category" />
                </SelectTrigger>
                <SelectContent className="bg-dark-card border-dark-border">
                  <SelectItem value="news">News</SelectItem>
                  <SelectItem value="financial">Financial</SelectItem>
                  <SelectItem value="search">Search</SelectItem>
                  <SelectItem value="ai">AI/LLM</SelectItem>
                  <SelectItem value="general">General</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid gap-2">
              <Label htmlFor="endpoint">API Endpoint</Label>
              <Input
                id="endpoint"
                value={formData.endpoint}
                onChange={(e) => setFormData({...formData, endpoint: e.target.value})}
                placeholder="e.g., https://api.example.com/v1"
                className="bg-dark-bg border-dark-border"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="key">API Key {editingKey && "(leave blank to keep current)"}</Label>
              <Input
                id="key"
                type="password"
                value={formData.key}
                onChange={(e) => setFormData({...formData, key: e.target.value})}
                placeholder="Enter your API key"
                className="bg-dark-bg border-dark-border"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="description">Description</Label>
              <Input
                id="description"
                value={formData.description}
                onChange={(e) => setFormData({...formData, description: e.target.value})}
                placeholder="Brief description of this API key"
                className="bg-dark-bg border-dark-border"
              />
            </div>
          </div>
          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => setIsDialogOpen(false)}
              className="border-gray-600 text-gray-300 hover:bg-dark-bg"
            >
              Cancel
            </Button>
            <Button 
              onClick={handleSubmit} 
              disabled={isSubmitting}
              className="bg-neon-green hover:bg-neon-green/90 text-white"
            >
              {isSubmitting ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : null}
              {editingKey ? "Update Key" : "Add Key"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <DialogContent className="bg-dark-card border-dark-border text-white sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle className="text-neon-pink">Confirm Deletion</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete the API key "{deletingKey?.name}"? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => setIsDeleteDialogOpen(false)}
              className="border-gray-600 text-gray-300 hover:bg-dark-bg"
            >
              Cancel
            </Button>
            <Button 
              onClick={handleDelete} 
              disabled={isSubmitting}
              variant="destructive" 
              className="bg-neon-pink hover:bg-neon-pink/80"
            >
              {isSubmitting ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : null}
              Delete Key
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  )
}
