#!/bin/bash

echo "🚀 Setting up API Keys for Market Intelligence Dashboard"
echo "=================================================="

# Check if .env.local exists
if [ ! -f .env.local ]; then
    echo "📝 Creating .env.local file..."
    cp .env.example .env.local 2>/dev/null || touch .env.local
fi

echo ""
echo "📋 API Keys Setup Checklist:"
echo ""

# Function to check if API key exists
check_api_key() {
    local key_name=$1
    local description=$2
    
    if grep -q "^${key_name}=" .env.local && ! grep -q "^${key_name}=$" .env.local; then
        echo "✅ $description - Configured"
    else
        echo "❌ $description - Missing"
        echo "   Add: ${key_name}=your_key_here"
    fi
}

# Check all API keys
check_api_key "NEWS_API_KEY" "NewsAPI"
check_api_key "MEDIASTACK_API_KEY" "MediaStack"
check_api_key "GNEWS_API_KEY" "GNews"
check_api_key "ALPHA_VANTAGE_API_KEY" "Alpha Vantage"
check_api_key "FINANCIAL_MODELING_PREP_API_KEY" "Financial Modeling Prep"
check_api_key "SERPAPI_API_KEY" "SerpAPI"
check_api_key "TAVILY_API_KEY" "Tavily"
check_api_key "GOOGLE_API_KEY" "Google AI (Gemini)"

echo ""
echo "🔧 Next Steps:"
echo "1. Get missing API keys from the providers"
echo "2. Add them to your .env.local file"
echo "3. Run: python api/test_api_keys.py"
echo "4. Start your development server: npm run dev"
echo ""
echo "📚 Full setup guide: https://github.com/your-repo/README.md"
