#!/bin/bash

echo "🚀 Setting up complete Market Intelligence Dashboard..."

# Create necessary directories
mkdir -p api/reports1
mkdir -p public/uploads
mkdir -p logs

# Set permissions
chmod +x scripts/*.sh

# Frontend setup
echo "📦 Installing frontend dependencies..."
npm install

# Backend setup
echo "🐍 Setting up Python backend..."
cd api
pip install -r requirements.txt
cd ..

# Database setup
echo "🗄️ Setting up database..."
if command -v psql &> /dev/null; then
    echo "Running database migrations..."
    # Run SQL scripts if PostgreSQL is available
    # psql $DATABASE_URL -f scripts/01_create_tables.sql
    # psql $DATABASE_URL -f scripts/02_seed_data.sql
fi

# Test API connections
echo "🔑 Testing API connections..."
cd api
python test_api_keys.py
cd ..

# Build frontend
echo "🏗️ Building frontend..."
npm run build

echo "✅ Setup complete! You can now run:"
echo "   Frontend: npm run dev"
echo "   Backend:  cd api && python main.py"
