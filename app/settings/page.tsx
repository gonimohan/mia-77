"use client"

import { useState, useEffect } from "react";
import { Settings, Palette, Bell, Database, User, Moon, Loader2 } from "lucide-react";
import { SidebarInset, SidebarTrigger } from "@/components/ui/sidebar";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ColorPaletteSelector } from "@/components/color-palette-selector";
import { useAuth } from "@/components/auth-provider";
import { useToast } from "@/hooks/use-toast";

export default function SettingsPage() {
  const { user, supabaseClient, loading: authLoading, isConfigured } = useAuth(); // Use supabaseClient from useAuth
  const { toast } = useToast();
  // const supabase = createClientComponentClient(); // Removed

  const [displayName, setDisplayName] = useState("");
  const [initialDisplayName, setInitialDisplayName] = useState("");
  const [userEmail, setUserEmail] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  // isPageLoading can be mostly replaced by authLoading and isConfigured checks
  // but we might still want a distinct page loading state if there are other async ops for settings page itself.
  // For now, let's rely on authLoading primarily.

  useEffect(() => {
    if (!authLoading && user) {
      const currentFullName = user.user_metadata?.full_name || user.user_metadata?.name || "";
      setDisplayName(currentFullName);
      setInitialDisplayName(currentFullName);
      setUserEmail(user.email || "");
    }
    // If !authLoading and !user, it means user is not logged in, AuthProvider should handle redirects.
  }, [user, authLoading]);

  const getAuthToken = async () => {
    if (!supabaseClient) return null; // Check supabaseClient
    const { data: { session }, error } = await supabaseClient.auth.getSession(); // Use supabaseClient
    if (error) {
      toast({ title: "Authentication Error", description: error.message, variant: "destructive" });
      return null;
    }
    return session?.access_token;
  };

  const handleSaveChanges = async () => {
    if (!supabaseClient || !isConfigured) { // Check supabaseClient and configuration
      toast({ title: "Error", description: "Supabase client not available or not configured.", variant: "destructive" });
      return;
    }
    setIsSaving(true);
    const token = await getAuthToken();
    if (!token) {
      // getAuthToken already shows a toast for auth errors
      setIsSaving(false);
      return;
    }

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_PYTHON_AGENT_API_BASE_URL}/users/me/profile`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ full_name: displayName }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "Failed to update profile.");
      }

      toast({ title: "Success", description: "Profile updated successfully." });
      setInitialDisplayName(displayName);

      // Refresh Supabase session to get updated user metadata
      await supabaseClient.auth.refreshSession(); // Use supabaseClient

    } catch (error: any) {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    } finally {
      setIsSaving(false);
    }
  };

  const handleCancelChanges = () => {
    setDisplayName(initialDisplayName);
  };

  const handleClearCache = () => {
    // Clear chat history from localStorage
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith('chat_messages_')) {
        localStorage.removeItem(key);
      }
    }
    // Clear current chat session ID
    localStorage.removeItem('chatSessionId');

    toast({
      title: "Cache Cleared",
      description: "Chat history and session data have been cleared.",
    });
  };

  // Display loading state while auth is being resolved or if Supabase is not yet configured
  if (authLoading) {
    return (
      <SidebarInset className="bg-dark-bg flex flex-col h-screen">
        <header className="flex h-16 shrink-0 items-center gap-2 border-b border-dark-border bg-dark-card/50 backdrop-blur-sm px-4">
           <SidebarTrigger className="-ml-1 text-white hover:bg-dark-card" />
           <Separator orientation="vertical" className="mr-2 h-4 bg-dark-border" />
           <div className="flex items-center gap-2">
             <Settings className="w-5 h-5 text-gray-400" />
             <h1 className="text-lg font-semibold text-white">Settings</h1>
           </div>
        </header>
        <div className="flex flex-1 items-center justify-center">
          <Loader2 className="w-10 h-10 text-neon-blue animate-spin" />
          <p className="ml-3 text-white">Loading account details...</p>
        </div>
      </SidebarInset>
    );
  }

  if (!isConfigured) {
     return (
      <SidebarInset className="bg-dark-bg flex flex-col h-screen">
        <header className="flex h-16 shrink-0 items-center gap-2 border-b border-dark-border bg-dark-card/50 backdrop-blur-sm px-4">
           <SidebarTrigger className="-ml-1 text-white hover:bg-dark-card" />
           <Separator orientation="vertical" className="mr-2 h-4 bg-dark-border" />
           <div className="flex items-center gap-2">
             <Settings className="w-5 h-5 text-gray-400" />
             <h1 className="text-lg font-semibold text-white">Settings</h1>
           </div>
        </header>
        <div className="flex flex-1 items-center justify-center text-white">
          Supabase is not configured. Please check your environment variables.
        </div>
      </SidebarInset>
    );
  }

  if (!user && !authLoading) { // Should be handled by ProtectedRoute
    return (
      <SidebarInset className="bg-dark-bg flex flex-col h-screen">
        <header className="flex h-16 shrink-0 items-center gap-2 border-b border-dark-border bg-dark-card/50 backdrop-blur-sm px-4">
           <SidebarTrigger className="-ml-1 text-white hover:bg-dark-card" />
           <Separator orientation="vertical" className="mr-2 h-4 bg-dark-border" />
           <div className="flex items-center gap-2">
             <Settings className="w-5 h-5 text-gray-400" />
             <h1 className="text-lg font-semibold text-white">Settings</h1>
           </div>
        </header>
        <div className="flex flex-1 items-center justify-center text-white">
          Redirecting to login...
        </div>
      </SidebarInset>
    );
  }

  return (
    <SidebarInset className="bg-dark-bg flex flex-col h-screen">
      {/* Header */}
      <header className="flex h-16 shrink-0 items-center gap-2 border-b border-dark-border bg-dark-card/50 backdrop-blur-sm px-4">
        <SidebarTrigger className="-ml-1 text-white hover:bg-dark-card" />
        <Separator orientation="vertical" className="mr-2 h-4 bg-dark-border" />
        <div className="flex items-center gap-2">
          <Settings className="w-5 h-5 text-gray-400" />
          <h1 className="text-lg font-semibold text-white">Settings</h1>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex flex-1 flex-col gap-6 p-6 overflow-y-auto"> {/* Added overflow-y-auto */}
        <Tabs defaultValue="appearance" className="w-full">
          <TabsList className="grid w-full grid-cols-4 bg-dark-card border border-dark-border">
            <TabsTrigger
              value="appearance"
              className="data-[state=active]:bg-neon-blue/20 data-[state=active]:text-neon-blue"
            >
              <Palette className="w-4 h-4 mr-2" />
              Appearance
            </TabsTrigger>
            <TabsTrigger
              value="notifications"
              className="data-[state=active]:bg-neon-green/20 data-[state=active]:text-neon-green"
            >
              <Bell className="w-4 h-4 mr-2" />
              Notifications
            </TabsTrigger>
            <TabsTrigger
              value="data"
              className="data-[state=active]:bg-neon-pink/20 data-[state=active]:text-neon-pink"
            >
              <Database className="w-4 h-4 mr-2" />
              Data
            </TabsTrigger>
            <TabsTrigger
              value="account"
              className="data-[state=active]:bg-neon-purple/20 data-[state=active]:text-neon-purple"
            >
              <User className="w-4 h-4 mr-2" />
              Account
            </TabsTrigger>
          </TabsList>

          {/* Appearance Tab */}
          <TabsContent value="appearance" className="space-y-6">
            <ColorPaletteSelector />

            <Card className="bg-dark-card border-dark-border">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Moon className="w-5 h-5 text-neon-purple" />
                  Theme Settings
                </CardTitle>
                <CardDescription className="text-gray-400">
                  Customize the overall appearance of the dashboard
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label className="text-white">Dark Mode</Label>
                    <p className="text-sm text-gray-400">Always enabled for optimal neon aesthetics</p>
                  </div>
                  <Switch checked disabled />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label className="text-white">Neon Effects</Label>
                    <p className="text-sm text-gray-400">Enable glowing effects and animations</p>
                  </div>
                  <Switch defaultChecked />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label className="text-white">Reduced Motion</Label>
                    <p className="text-sm text-gray-400">Minimize animations for better performance</p>
                  </div>
                  <Switch />
                </div>

                <div className="space-y-2">
                  <Label className="text-white">Animation Speed</Label>
                  <Select defaultValue="normal">
                    <SelectTrigger className="bg-dark-bg border-dark-border text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-dark-card border-dark-border">
                      <SelectItem value="slow">Slow</SelectItem>
                      <SelectItem value="normal">Normal</SelectItem>
                      <SelectItem value="fast">Fast</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Notifications Tab */}
          <TabsContent value="notifications" className="space-y-6">
            <Card className="bg-dark-card border-dark-border">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Bell className="w-5 h-5 text-neon-green" />
                  Notification Preferences
                </CardTitle>
                <CardDescription className="text-gray-400">
                  Configure how you receive updates and alerts
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label className="text-white">Market Alerts</Label>
                    <p className="text-sm text-gray-400">Get notified about significant market changes</p>
                  </div>
                  <Switch defaultChecked />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label className="text-white">Competitor Updates</Label>
                    <p className="text-sm text-gray-400">Receive alerts about competitor activities</p>
                  </div>
                  <Switch defaultChecked />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label className="text-white">Data Sync Notifications</Label>
                    <p className="text-sm text-gray-400">Get notified when data sources sync</p>
                  </div>
                  <Switch />
                </div>

                <div className="space-y-2">
                  <Label className="text-white">Notification Frequency</Label>
                  <Select defaultValue="realtime">
                    <SelectTrigger className="bg-dark-bg border-dark-border text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-dark-card border-dark-border">
                      <SelectItem value="realtime">Real-time</SelectItem>
                      <SelectItem value="hourly">Hourly</SelectItem>
                      <SelectItem value="daily">Daily</SelectItem>
                      <SelectItem value="weekly">Weekly</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Data Tab */}
          <TabsContent value="data" className="space-y-6">
            <Card className="bg-dark-card border-dark-border">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Database className="w-5 h-5 text-neon-pink" />
                  Data Management
                </CardTitle>
                <CardDescription className="text-gray-400">Configure data retention and sync settings</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-white">Data Retention Period</Label>
                  <Select defaultValue="90days">
                    <SelectTrigger className="bg-dark-bg border-dark-border text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-dark-card border-dark-border">
                      <SelectItem value="30days">30 Days</SelectItem>
                      <SelectItem value="90days">90 Days</SelectItem>
                      <SelectItem value="1year">1 Year</SelectItem>
                      <SelectItem value="unlimited">Unlimited</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label className="text-white">Auto Sync Frequency</Label>
                  <Select defaultValue="hourly">
                    <SelectTrigger className="bg-dark-bg border-dark-border text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-dark-card border-dark-border">
                      <SelectItem value="realtime">Real-time</SelectItem>
                      <SelectItem value="hourly">Every Hour</SelectItem>
                      <SelectItem value="daily">Daily</SelectItem>
                      <SelectItem value="manual">Manual Only</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label className="text-white">Cache Data Locally</Label>
                    <p className="text-sm text-gray-400">Store data locally for faster loading</p>
                  </div>
                  <Switch defaultChecked />
                </div>

                <div className="pt-4 border-t border-dark-border">
                  <Button
                    variant="outline"
                    className="border-neon-pink/50 text-neon-pink hover:bg-neon-pink/10"
                    onClick={handleClearCache}
                  >
                    Clear All Cache
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Account Tab */}
          <TabsContent value="account" className="space-y-6">
            <Card className="bg-dark-card border-dark-border">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <User className="w-5 h-5 text-neon-purple" />
                  Account Information
                </CardTitle>
                <CardDescription className="text-gray-400">Manage your account details and preferences</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label className="text-white" htmlFor="displayName">Display Name</Label>
                    <Input
                      id="displayName"
                      value={displayName}
                      onChange={(e) => setDisplayName(e.target.value)}
                      className="bg-dark-bg border-dark-border text-white"
                      disabled={authLoading || isSaving || !isConfigured}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-white" htmlFor="email">Email</Label>
                    <Input
                      id="email"
                      value={userEmail}
                      readOnly
                      className="bg-dark-bg border-dark-border text-gray-400"
                      disabled={authLoading || !isConfigured}
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label className="text-white">Time Zone</Label>
                  <Select defaultValue="utc" disabled={authLoading || !isConfigured}>
                    <SelectTrigger className="bg-dark-bg border-dark-border text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-dark-card border-dark-border">
                      <SelectItem value="utc">UTC</SelectItem>
                      <SelectItem value="est">Eastern Time</SelectItem>
                      <SelectItem value="pst">Pacific Time</SelectItem>
                      <SelectItem value="cet">Central European Time</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label className="text-white">Two-Factor Authentication</Label>
                    <p className="text-sm text-gray-400">Add an extra layer of security</p>
                  </div>
                  <Switch disabled={authLoading || !isConfigured} />
                </div>

                <div className="pt-4 border-t border-dark-border flex gap-2">
                  <Button
                    onClick={handleSaveChanges}
                    disabled={isSaving || authLoading || displayName === initialDisplayName || !isConfigured || !supabaseClient}
                    className="bg-neon-purple/20 border border-neon-purple/50 text-neon-purple hover:bg-neon-purple/30"
                  >
                    {isSaving ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
                    Save Changes
                  </Button>
                  <Button
                    variant="outline"
                    className="border-gray-600 text-gray-400"
                    onClick={handleCancelChanges}
                    disabled={isSaving || authLoading || !isConfigured}
                  >
                    Cancel
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </SidebarInset>
  )
}
