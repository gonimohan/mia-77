-- Market Intelligence Dashboard Database Schema

-- Users table for authentication
CREATE TABLE IF NOT EXISTS users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    avatar_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Data sources configuration
CREATE TABLE IF NOT EXISTS data_sources (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL, -- 'api', 'database', 'file', etc.
    config JSONB NOT NULL, -- Store API keys, connection strings, etc.
    status VARCHAR(50) DEFAULT 'inactive', -- 'active', 'inactive', 'error'
    last_sync TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Market analysis reports
CREATE TABLE IF NOT EXISTS reports (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    market_domain VARCHAR(255) NOT NULL,
    query_text TEXT,
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    report_data JSONB,
    file_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- KPI metrics for dashboard
CREATE TABLE IF NOT EXISTS kpi_metrics (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15,2),
    metric_unit VARCHAR(50),
    change_percentage DECIMAL(5,2),
    period_start DATE,
    period_end DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Market trends data
CREATE TABLE IF NOT EXISTS market_trends (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    trend_name VARCHAR(255) NOT NULL,
    market_domain VARCHAR(255),
    trend_data JSONB NOT NULL,
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Competitor analysis data
CREATE TABLE IF NOT EXISTS competitors (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    company_name VARCHAR(255) NOT NULL,
    market_domain VARCHAR(255),
    market_share DECIMAL(5,2),
    competitor_data JSONB,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_data_sources_user_id ON data_sources(user_id);
CREATE INDEX IF NOT EXISTS idx_reports_user_id ON reports(user_id);
CREATE INDEX IF NOT EXISTS idx_kpi_metrics_user_id ON kpi_metrics(user_id);
CREATE INDEX IF NOT EXISTS idx_market_trends_user_id ON market_trends(user_id);
CREATE INDEX IF NOT EXISTS idx_competitors_user_id ON competitors(user_id);
