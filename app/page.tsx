"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/components/auth-provider"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { BarChart3, TrendingUp, Users, Target, Brain, Zap, Shield, ArrowRight, Sparkles } from "lucide-react"
import Link from "next/link"

export default function HomePage() {
  const { user, loading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!loading && user) {
      router.push("/dashboard")
    }
  }, [user, loading, router])

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900 flex items-center justify-center">
        <div className="flex items-center gap-2 text-white">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-purple-500 border-t-transparent" />
          <span>Loading...</span>
        </div>
      </div>
    )
  }

  if (user) {
    return null // Will redirect to dashboard
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900">
      {/* Navigation */}
      <nav className="border-b border-purple-500/20 bg-black/20 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-8 w-8 text-purple-400" />
              <span className="text-xl font-bold text-white">Market Intelligence</span>
            </div>
            <div className="flex items-center gap-4">
              <Link href="/login">
                <Button variant="ghost" className="text-white hover:text-purple-300">
                  Sign In
                </Button>
              </Link>
              <Link href="/register">
                <Button className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700">
                  Get Started
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative py-20 px-4 sm:px-6 lg:px-8">
        <div className="absolute inset-0 bg-[url('/grid.svg')] bg-center [mask-image:linear-gradient(180deg,white,rgba(255,255,255,0))]" />

        <div className="max-w-7xl mx-auto text-center relative">
          <Badge className="mb-4 bg-purple-500/20 text-purple-300 border-purple-500/30">
            <Sparkles className="w-3 h-3 mr-1" />
            AI-Powered Market Intelligence
          </Badge>

          <h1 className="text-4xl md:text-6xl font-bold text-white mb-6">
            Transform Your
            <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              {" "}
              Market Intelligence
            </span>
          </h1>

          <p className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto">
            Harness the power of AI to analyze market trends, track competitors, and discover opportunities with
            real-time data from multiple sources.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/register">
              <Button
                size="lg"
                className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
              >
                Start Free Trial
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
            <Link href="/login">
              <Button
                size="lg"
                variant="outline"
                className="border-purple-500/50 text-purple-300 hover:bg-purple-500/10"
              >
                Sign In
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Powerful Features for Market Leaders</h2>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Everything you need to stay ahead of the competition and make data-driven decisions.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                icon: Brain,
                title: "AI-Powered Analysis",
                description:
                  "Advanced machine learning algorithms analyze market data and provide actionable insights.",
              },
              {
                icon: TrendingUp,
                title: "Real-Time Trends",
                description: "Track market trends as they happen with live data from multiple sources.",
              },
              {
                icon: Users,
                title: "Competitor Intelligence",
                description: "Monitor competitor activities, pricing, and strategies in real-time.",
              },
              {
                icon: Target,
                title: "Customer Insights",
                description: "Deep dive into customer behavior and preferences with advanced analytics.",
              },
              {
                icon: Zap,
                title: "Instant Alerts",
                description: "Get notified immediately when important market changes occur.",
              },
              {
                icon: Shield,
                title: "Secure & Reliable",
                description: "Enterprise-grade security with 99.9% uptime guarantee.",
              },
            ].map((feature, index) => (
              <Card
                key={index}
                className="bg-black/40 border-purple-500/20 backdrop-blur-xl hover:border-purple-500/40 transition-colors"
              >
                <CardHeader>
                  <feature.icon className="h-12 w-12 text-purple-400 mb-4" />
                  <CardTitle className="text-white">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-gray-300">{feature.description}</CardDescription>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-black/20">
        <div className="max-w-7xl mx-auto text-center">
          <h2 className="text-2xl md:text-3xl font-bold text-white mb-4">
            Why Choose Our Platform?
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-12">
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-400 mb-2">Real-time</div>
              <div className="text-gray-300">Market Data Analysis</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-400 mb-2">AI-Powered</div>
              <div className="text-gray-300">Intelligence Insights</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-400 mb-2">Enterprise</div>
              <div className="text-gray-300">Grade Security</div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Ready to Transform Your Market Intelligence?
          </h2>
          <p className="text-xl text-gray-300 mb-8">
            Join thousands of businesses already using our platform to gain competitive advantage.
          </p>
          <Link href="/register">
            <Button
              size="lg"
              className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
            >
              Start Your Free Trial
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-purple-500/20 bg-black/20 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center gap-2 mb-4 md:mb-0">
              <BarChart3 className="h-6 w-6 text-purple-400" />
              <span className="text-lg font-semibold text-white">Market Intelligence</span>
            </div>
            <div className="text-gray-400 text-sm">© 2024 Market Intelligence Dashboard. All rights reserved.</div>
          </div>
        </div>
      </footer>
    </div>
  )
}
