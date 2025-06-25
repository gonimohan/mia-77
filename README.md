# Market Intelligence Dashboard

A comprehensive AI-powered market intelligence platform with real-time data analysis, competitor tracking, and customer insights.

## 🚀 Features

### Frontend Features
- ✅ **Authentication System** - Complete login/register with Supabase
- ✅ **Dashboard** - Real-time KPIs and analytics
- ✅ **Market Trends** - AI-powered trend analysis
- ✅ **Competitor Analysis** - Comprehensive competitor tracking
- ✅ **Customer Insights** - Advanced customer segmentation
- ✅ **Data Integration** - Multiple data source management
- ✅ **Downloads** - Export reports and data
- ✅ **Chat Interface** - AI-powered market intelligence chat
- ✅ **Settings** - Theme customization and preferences
- ✅ **Responsive Design** - Works on all devices
- ✅ **Dark Theme** - Cyberpunk neon aesthetics

### Backend Features
- ✅ **FastAPI Server** - High-performance Python backend
- ✅ **AI Agent Logic** - LangChain-powered market intelligence
- ✅ **Multiple Data Sources** - News, financial, search APIs
- ✅ **Vector Store** - FAISS-powered document search
- ✅ **Chart Generation** - Matplotlib/Seaborn visualizations
- ✅ **Database Integration** - SQLite + Supabase
- ✅ **Error Handling** - Comprehensive error management
- ✅ **API Testing** - Automated API connection testing

## 🛠️ Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Shadcn/ui** - Modern UI components
- **Supabase** - Authentication and database
- **Recharts** - Data visualization

### Backend
- **FastAPI** - Modern Python web framework
- **LangChain** - AI/LLM integration
- **Google Gemini** - AI language model
- **SQLite** - Local database
- **Supabase** - Cloud database
- **Matplotlib/Seaborn** - Chart generation
- **FAISS** - Vector similarity search
- **Pandas/NumPy** - Data processing

## 🔧 Setup Instructions

### Prerequisites
- Node.js 18+ and npm
- Python 3.9+
- Git

### 1. Clone Repository
\`\`\`bash
git clone <repository-url>
cd market-intelligence-dashboard
\`\`\`

### 2. Environment Setup
Copy the environment files and update with your credentials:

\`\`\`bash
# Frontend environment
cp .env.local.example .env.local

# Backend environment  
cp api/.env.example api/.env
\`\`\`

### 3. Install Dependencies
\`\`\`bash
# Frontend
npm install

# Backend
cd api
pip install -r requirements.txt
cd ..
\`\`\`

### 4. Database Setup
\`\`\`bash
# Run database migrations
npm run db:setup

# Or manually run SQL scripts
psql $DATABASE_URL -f scripts/01_create_tables.sql
psql $DATABASE_URL -f scripts/02_seed_data.sql
\`\`\`

### 5. Start Development Servers
\`\`\`bash
# Terminal 1: Frontend
npm run dev

# Terminal 2: Backend
npm run backend
\`\`\`

### 6. Test API Connections
\`\`\`bash
npm run test-apis
\`\`\`

## 🔑 Environment Variables

### Frontend (.env.local)
\`\`\`env
# Supabase
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# API Keys
NEWS_API_KEY=your_news_api_key
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key

# Backend URL
NEXT_PUBLIC_PYTHON_AGENT_API_BASE_URL=http://localhost:8000
\`\`\`

### Backend (api/.env)
\`\`\`env
# Same as frontend plus:
PORT=8000
HOST=0.0.0.0
ENVIRONMENT=development
\`\`\`

## 📱 Usage

1. **Register/Login** - Create account or sign in
2. **Dashboard** - View real-time market KPIs
3. **Data Integration** - Configure your data sources
4. **Run Analysis** - Generate market intelligence reports
5. **View Insights** - Explore trends, competitors, customers
6. **Download Reports** - Export data and visualizations
7. **Chat** - Ask AI questions about your market data

## 🔍 API Endpoints

### Frontend API Routes
- `GET /api/analysis` - Market analysis
- `POST /api/chat` - AI chat interface
- `GET /api/kpi` - KPI data
- `POST /api/agent/sync` - Data synchronization

### Backend API Routes
- `POST /analyze` - Run market intelligence analysis
- `POST /chat` - Chat with AI agent
- `GET /kpi` - Retrieve KPIs
- `POST /agent/sync` - Sync agent data

## 🧪 Testing

\`\`\`bash
# Test API connections
npm run test-apis

# Run frontend tests
npm run test

# Test backend
cd api && python -m pytest
\`\`\`

## 🚀 Deployment

### Frontend (Vercel)
\`\`\`bash
npm run build
vercel deploy
\`\`\`

### Backend (Railway/Render)
\`\`\`bash
cd api
python main.py
\`\`\`

## 📊 Features Checklist

### ✅ Completed Features
- [x] Authentication (Login/Register)
- [x] Dashboard with real-time KPIs
- [x] Market trends analysis
- [x] Competitor intelligence
- [x] Customer insights
- [x] Data source integration
- [x] Download management
- [x] AI chat interface
- [x] Settings and customization
- [x] Responsive design
- [x] Error handling
- [x] API testing suite

### 🔄 In Progress
- [ ] Advanced analytics
- [ ] Email notifications
- [ ] Team collaboration
- [ ] API rate limiting

### 📋 Future Enhancements
- [ ] Mobile app
- [ ] Advanced AI models
- [ ] Custom integrations
- [ ] Enterprise features

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📧 Email: support@marketintelligence.com
- 💬 Discord: [Join our community](https://discord.gg/marketintel)
- 📖 Documentation: [docs.marketintelligence.com](https://docs.marketintelligence.com)

---

**Ready to transform your market intelligence? Start your journey today!** 🚀
