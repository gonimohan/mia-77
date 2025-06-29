"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  TrendingUp,
  Users,
  Eye,
  Database,
  Download,
  Settings,
  Search,
  Bell,
  Zap,
  LogOut,
} from "lucide-react"; // Added LogOut
import { useAuth } from "./auth-provider"; // Added useAuth

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInput,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

const menuItems = [
  {
    title: "Dashboard",
    url: "/dashboard",
    icon: BarChart3,
    color: "text-neon-blue",
  },
  {
    title: "Competitor Analysis",
    url: "/competitors",
    icon: Users,
    color: "text-neon-green",
  },
  {
    title: "Market Trends",
    url: "/trends",
    icon: TrendingUp,
    color: "text-neon-pink",
  },
  {
    title: "Customer Insights",
    url: "/insights",
    icon: Eye,
    color: "text-neon-purple",
  },
  {
    title: "Data Integration",
    url: "/data-integration",
    icon: Database,
    color: "text-neon-orange",
  },
  {
    title: "Downloads",
    url: "/downloads",
    icon: Download,
    color: "text-neon-blue",
  },
];

export function AppSidebar() {
  const pathname = usePathname();
  const { user, signOut } = useAuth(); // Added this line

  return (
    <Sidebar className="bg-dark-bg border-dark-border">
      <SidebarHeader className="border-b border-dark-border">
        <div className="flex items-center gap-2 px-4 py-2">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-r from-neon-blue to-neon-purple">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <div className="flex flex-col">
            <span className="text-lg font-bold text-white text-center">
              MIA{" "}
            </span>
            <span className="text-xs text-gray-400">
              Market Intelligent Agent
            </span>
          </div>
        </div>

        <div className="px-4 pb-2">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <SidebarInput
              placeholder="Search insights..."
              className="pl-10 bg-dark-card border-dark-border text-white placeholder:text-gray-400 focus:border-neon-blue focus:ring-1 focus:ring-neon-blue"
            />
          </div>
        </div>
      </SidebarHeader>

      <SidebarContent className="bg-dark-bg">
        <SidebarGroup>
          <SidebarGroupLabel className="text-gray-400 text-xs uppercase tracking-wider">
            Analytics
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {menuItems.map((item) => {
                const isActive =
                  pathname === item.url || pathname.startsWith(item.url + "/");
                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton
                      asChild
                      isActive={isActive}
                      className={`
                        group relative overflow-hidden transition-all duration-300
                        ${
                          isActive
                            ? `bg-dark-card border border-opacity-50 ${item.color.replace(
                                "text-",
                                "border-"
                              )} shadow-lg`
                            : "hover:bg-dark-card hover:border hover:border-opacity-30 hover:border-gray-600"
                        }
                      `}
                    >
                      <Link
                        href={item.url}
                        className="flex items-center gap-3 w-full"
                      >
                        <item.icon
                          className={`w-5 h-5 transition-all duration-300 ${
                            isActive
                              ? `${item.color} drop-shadow-lg animate-pulse`
                              : "text-gray-400 group-hover:text-white"
                          }`}
                        />
                        <span
                          className={`font-medium transition-all duration-300 ${
                            isActive
                              ? "text-white font-semibold"
                              : "text-gray-300 group-hover:text-white"
                          }`}
                        >
                          {item.title}
                        </span>
                        {isActive && (
                          <div
                            className={`absolute right-0 top-0 bottom-0 w-1 ${item.color.replace(
                              "text-",
                              "bg-"
                            )} rounded-l-full`}
                          />
                        )}
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel className="text-gray-400 text-xs uppercase tracking-wider">
            System
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton
                  asChild
                  isActive={pathname === "/settings"}
                  className={`
                    group relative overflow-hidden transition-all duration-300
                    ${
                      pathname === "/settings"
                        ? "bg-dark-card border border-opacity-50 border-gray-400 shadow-lg"
                        : "hover:bg-dark-card hover:border hover:border-opacity-30 hover:border-gray-600"
                    }
                  `}
                >
                  <Link href="/settings" className="flex items-center gap-3">
                    <Settings
                      className={`w-5 h-5 transition-all duration-300 ${
                        pathname === "/settings"
                          ? "text-gray-300 drop-shadow-lg"
                          : "text-gray-400 group-hover:text-white"
                      }`}
                    />
                    <span
                      className={`font-medium transition-all duration-300 ${
                        pathname === "/settings"
                          ? "text-white font-semibold"
                          : "text-gray-300 group-hover:text-white"
                      }`}
                    >
                      Settings
                    </span>
                    {pathname === "/settings" && (
                      <div className="absolute right-0 top-0 bottom-0 w-1 bg-gray-400 rounded-l-full" />
                    )}
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="border-t border-dark-border bg-dark-bg">
        {user && ( // Only show if user is logged in
          <SidebarMenu>
            <SidebarMenuItem>
              <div className="flex items-center gap-3 px-2 py-3">
                <Avatar className="w-8 h-8 border-2 border-neon-blue">
                  <AvatarImage
                    src={
                      user.user_metadata?.avatar_url || "/placeholder-user.jpg"
                    }
                  />
                  <AvatarFallback className="bg-neon-blue text-dark-bg font-bold">
                    {user.email?.charAt(0).toUpperCase() || "U"}
                  </AvatarFallback>
                </Avatar>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-white truncate">
                    {user.user_metadata?.full_name || user.email}
                  </p>
                  <p className="text-xs text-gray-400 truncate">{user.email}</p>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="w-8 h-8 text-gray-400 hover:text-neon-pink hover:bg-dark-card"
                  onClick={async () => {
                    await signOut();
                    // Router push to /login is usually handled by middleware or ProtectedRoute after state update
                  }}
                  title="Logout"
                >
                  <LogOut className="w-4 h-4" />
                </Button>
              </div>
            </SidebarMenuItem>
          </SidebarMenu>
        )}
      </SidebarFooter>

      <SidebarRail />
    </Sidebar>
  );
}
