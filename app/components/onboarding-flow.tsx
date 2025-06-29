"use client"

import React, { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { ArrowRight, CheckCircle, Database, MessageSquare, User, Sparkles } from "lucide-react"
import Link from "next/link"

interface OnboardingFlowProps {
  onComplete: () => void;
  userName: string;
}

export const OnboardingFlow: React.FC<OnboardingFlowProps> = ({ onComplete, userName }) => {
  const [step, setStep] = useState(0)

  const steps = [
    {
      title: `Welcome, ${userName || "User"}!`,
      description: "Let's get you set up to make the most of your Market Intelligence Dashboard.",
      icon: <Sparkles className="w-12 h-12 text-neon-blue" />,
      action: <Button onClick={() => setStep(1)} className="bg-neon-blue hover:bg-neon-blue/90">Get Started <ArrowRight className="ml-2 h-4 w-4" /></Button>,
    },
    {
      title: "Step 1: Complete Your Profile",
      description: "Ensure your profile is up-to-date for a personalized experience.",
      icon: <User className="w-12 h-12 text-neon-purple" />,
      action: (
        <div className="flex gap-2">
          <Link href="/settings" passHref>
            <Button className="bg-neon-purple hover:bg-neon-purple/90">Go to Settings <ArrowRight className="ml-2 h-4 w-4" /></Button>
          </Link>
          <Button variant="outline" onClick={() => setStep(2)}>Skip for now</Button>
        </div>
      ),
    },
    {
      title: "Step 2: Integrate Your Data",
      description: "Connect your data sources to unlock powerful market insights.",
      icon: <Database className="w-12 h-12 text-neon-green" />,
      action: (
        <div className="flex gap-2">
          <Link href="/data-integration" passHref>
            <Button className="bg-neon-green hover:bg-neon-green/90">Add Data Source <ArrowRight className="ml-2 h-4 w-4" /></Button>
          </Link>
          <Button variant="outline" onClick={() => setStep(3)}>Skip for now</Button>
        </div>
      ),
    },
    {
      title: "Step 3: Try the RAG Chat",
      description: "Ask questions about your data and get AI-powered answers.",
      icon: <MessageSquare className="w-12 h-12 text-neon-orange" />,
      action: (
        <div className="flex gap-2">
          <Link href="/chat" passHref>
            <Button className="bg-neon-orange hover:bg-neon-orange/90">Start Chatting <ArrowRight className="ml-2 h-4 w-4" /></Button>
          </Link>
          <Button variant="outline" onClick={() => setStep(4)}>Skip for now</Button>
        </div>
      ),
    },
    {
      title: "You're All Set!",
      description: "You've completed the onboarding. Enjoy your Market Intelligence Dashboard!",
      icon: <CheckCircle className="w-12 h-12 text-neon-green" />,
      action: <Button onClick={onComplete} className="bg-neon-green hover:bg-neon-green/90">Go to Dashboard</Button>,
    },
  ]

  const currentStep = steps[step]
  const progress = ((step + 1) / steps.length) * 100

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <Card className="w-full max-w-2xl bg-dark-card border-dark-border text-white shadow-lg">
        <CardHeader className="text-center">
          <CardTitle className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
            {currentStep.title}
          </CardTitle>
          <CardDescription className="text-gray-300 mt-2">{currentStep.description}</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center space-y-6 p-6">
          <div className="mb-4">{currentStep.icon}</div>
          <div className="w-full flex justify-center">{currentStep.action}</div>
          {step < steps.length - 1 && (
            <div className="w-full mt-4">
              <Progress value={progress} className="h-2 bg-dark-bg" />
              <p className="text-center text-sm text-gray-400 mt-2">Step {step + 1} of {steps.length -1}</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
