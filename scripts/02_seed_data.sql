-- Seed data for Market Intelligence Dashboard

-- Insert sample KPI metrics
INSERT INTO kpi_metrics (metric_name, metric_value, metric_unit, change_percentage, period_start, period_end) VALUES
('Market Growth', 15.7, '%', 2.3, '2024-01-01', '2024-12-31'),
('Competitor Activity', 847, 'companies', -5.2, '2024-01-01', '2024-12-31'),
('Consumer Sentiment', 78.5, 'score', 12.1, '2024-01-01', '2024-12-31'),
('Market Share', 23.4, '%', 0.8, '2024-01-01', '2024-12-31');

-- Insert sample market trends
INSERT INTO market_trends (trend_name, market_domain, trend_data, confidence_score) VALUES
('AI Integration', 'Technology', '{"description": "Rapid adoption of AI across industries", "impact": "High", "timeframe": "Short-term"}', 0.92),
('Sustainable Tech', 'Technology', '{"description": "Growing focus on environmental sustainability", "impact": "Medium", "timeframe": "Long-term"}', 0.78),
('Remote Work Tools', 'SaaS', '{"description": "Continued demand for collaboration tools", "impact": "High", "timeframe": "Medium-term"}', 0.85);

-- Insert sample competitors
INSERT INTO competitors (company_name, market_domain, market_share, competitor_data) VALUES
('TechCorp Inc', 'Technology', 18.5, '{"revenue": "2.5B", "employees": 15000, "growth_rate": 12.3}'),
('InnovateLabs', 'Technology', 14.2, '{"revenue": "1.8B", "employees": 8500, "growth_rate": 8.7}'),
('FutureSoft', 'SaaS', 22.1, '{"revenue": "3.2B", "employees": 12000, "growth_rate": 15.6}');
