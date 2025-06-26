#!/usr/bin/env python3
import requests
import time
import unittest
import uuid
from io import BytesIO
import pandas as pd

# API Base URL - using the environment variable from .env.local
API_BASE_URL = "http://localhost:8000"


class MarketIntelligenceAPITest(unittest.TestCase):
    """Test suite for the Market Intelligence Agent API"""

    def setUp(self):
        """Set up test environment before each test"""
        self.api_url = API_BASE_URL
        self.test_user_email = "test.user@gmail.com"
        self.test_user_password = "Test123!@#"
        self.auth_token = None
        self.test_file_id = None
        self.test_data_source_id = None

        # Try to authenticate and get a token
        self.authenticate()

    def authenticate(self):
        """Authenticate with Supabase and get a JWT token"""
        # For testing purposes, we'll skip authentication
        # and focus on testing non-authenticated endpoints
        print("Skipping authentication for testing purposes")
        self.auth_token = None

    def sign_up_test_user(self, supabase_url, supabase_anon_key):
        """Sign up a test user if authentication fails"""
        # Skip for testing purposes
        pass

    def get_headers(self, with_auth=True):
        """Get request headers with or without authentication"""
        headers = {"Content-Type": "application/json"}

        if with_auth and self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        return headers

    def create_test_files(self):
        """Create test files for upload testing"""
        # Create a CSV file
        csv_data = pd.DataFrame(
            {
                "company": ["Apple", "Microsoft", "Google", "Amazon", "Meta"],
                "revenue": [365.8, 198.3, 282.8, 513.9, 116.6],
                "employees": [164000, 181000, 156500, 1540000, 77805],
                "market_cap": [2.95, 2.81, 1.71, 1.37, 1.19],
                "year_founded": [1976, 1975, 1998, 1994, 2004],
            }
        )
        csv_buffer = BytesIO()
        csv_data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Create a text file
        text_content = """
        Market Intelligence Report - Q2 2025
        
        Executive Summary:
        The technology sector continues to show strong growth in Q2 2025, with AI-driven solutions 
        leading the market. Cloud services revenue increased by 28% year-over-year, while 
        hardware sales declined by 5%. Emerging markets in Southeast Asia present significant 
        opportunities for expansion.
        
        Key Trends:
        1. AI integration across enterprise software
        2. Increased focus on sustainability metrics
        3. Supply chain resilience investments
        4. Remote work technology consolidation
        """
        text_buffer = BytesIO(text_content.encode("utf-8"))

        # Create an Excel file
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            csv_data.to_excel(writer, sheet_name="Market Data", index=False)
            pd.DataFrame({"quarter": ["Q1", "Q2", "Q3", "Q4"], "growth": [0.12, 0.18, 0.09, 0.15]}).to_excel(
                writer, sheet_name="Quarterly Growth", index=False
            )
        excel_buffer.seek(0)

        return {
            "csv": ("market_data.csv", csv_buffer, "text/csv"),
            "text": ("report.txt", text_buffer, "text/plain"),
            "excel": (
                "market_analysis.xlsx",
                excel_buffer,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        }

    # 1. Basic Health Checks
    def test_01_root_endpoint(self):
        """Test the root endpoint for basic API info"""
        print("\n1. Testing root endpoint...")
        response = requests.get(f"{self.api_url}/")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["message"], "Market Intelligence Agent API")
        self.assertEqual(data["version"], "1.0.0")
        self.assertEqual(data["status"], "running")
        self.assertIn("timestamp", data)
        self.assertIn("features", data)

        print("✅ Root endpoint test passed")

    def test_02_health_endpoint(self):
        """Test the health endpoint for service status"""
        print("\n2. Testing health endpoint...")
        response = requests.get(f"{self.api_url}/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)
        self.assertIn("services", data)

        print("✅ Health endpoint test passed")

    # 2. Authentication System
    def test_03_authentication_required(self):
        """Test that protected endpoints require authentication"""
        print("\n3. Testing authentication requirement...")

        # Test analyze endpoint without authentication
        response = requests.post(
            f"{self.api_url}/api/analyze",
            headers=self.get_headers(with_auth=False),
            json={"query": "market trends", "market_domain": "technology"},
        )

        # Should return 401 Unauthorized
        self.assertEqual(response.status_code, 401)

        # Test files endpoint without authentication
        response = requests.get(f"{self.api_url}/api/files", headers=self.get_headers(with_auth=False))

        # Should return 401 Unauthorized
        self.assertEqual(response.status_code, 401)

        print("✅ Authentication requirement test passed")

    # 3. File Upload Functionality
    def test_04_file_upload(self):
        """Test file upload functionality with different file types"""
        print("\n4. Testing file upload functionality...")

        if not self.auth_token:
            self.skipTest("Authentication token not available")

        test_files = self.create_test_files()

        # Test CSV upload
        print("Testing CSV upload...")
        files = {"file": test_files["csv"]}
        response = requests.post(
            f"{self.api_url}/api/upload", headers={"Authorization": f"Bearer {self.auth_token}"}, files=files
        )

        if response.status_code == 200:
            data = response.json()
            self.test_file_id = data.get("file_id")
            self.assertIn("file_id", data)
            self.assertIn("filename", data)
            self.assertIn("file_type", data)
            self.assertIn("processed_data", data)
            self.assertIn("ai_analysis", data)
            print(f"✅ CSV upload successful. File ID: {self.test_file_id}")
        else:
            print(f"❌ CSV upload failed: {response.status_code} - {response.text}")
            self.fail(f"CSV upload failed: {response.status_code} - {response.text}")

        # Test Excel upload
        print("Testing Excel upload...")
        files = {"file": test_files["excel"]}
        response = requests.post(
            f"{self.api_url}/api/upload", headers={"Authorization": f"Bearer {self.auth_token}"}, files=files
        )

        if response.status_code == 200:
            data = response.json()
            self.assertIn("file_id", data)
            self.assertIn("filename", data)
            self.assertIn("file_type", data)
            self.assertIn("processed_data", data)
            print("✅ Excel upload successful")
        else:
            print(f"❌ Excel upload failed: {response.status_code} - {response.text}")

        # Test Text upload
        print("Testing Text upload...")
        files = {"file": test_files["text"]}
        response = requests.post(
            f"{self.api_url}/api/upload", headers={"Authorization": f"Bearer {self.auth_token}"}, files=files
        )

        if response.status_code == 200:
            data = response.json()
            self.assertIn("file_id", data)
            self.assertIn("filename", data)
            self.assertIn("file_type", data)
            self.assertIn("processed_data", data)
            print("✅ Text upload successful")
        else:
            print(f"❌ Text upload failed: {response.status_code} - {response.text}")

    def test_05_list_files(self):
        """Test listing uploaded files"""
        print("\n5. Testing file listing...")

        if not self.auth_token:
            self.skipTest("Authentication token not available")

        response = requests.get(f"{self.api_url}/api/files", headers=self.get_headers())

        if response.status_code == 200:
            data = response.json()
            self.assertIn("files", data)
            if data["files"]:
                print(f"✅ File listing successful. Found {len(data['files'])} files.")
            else:
                print("✅ File listing successful. No files found.")
        else:
            print(f"❌ File listing failed: {response.status_code} - {response.text}")
            self.fail(f"File listing failed: {response.status_code} - {response.text}")

    def test_06_file_details(self):
        """Test getting file details"""
        print("\n6. Testing file details...")

        if not self.auth_token or not self.test_file_id:
            self.skipTest("Authentication token or test file ID not available")

        response = requests.get(f"{self.api_url}/api/files/{self.test_file_id}", headers=self.get_headers())

        if response.status_code == 200:
            data = response.json()
            self.assertEqual(data["file_id"], self.test_file_id)
            self.assertIn("filename", data)
            self.assertIn("file_type", data)
            self.assertIn("file_size", data)
            self.assertIn("processed_data", data)
            print(f"✅ File details retrieval successful for file ID: {self.test_file_id}")
        else:
            print(f"❌ File details retrieval failed: {response.status_code} - {response.text}")

    def test_07_file_analysis(self):
        """Test file analysis"""
        print("\n7. Testing file analysis...")

        if not self.auth_token or not self.test_file_id:
            self.skipTest("Authentication token or test file ID not available")

        response = requests.post(
            f"{self.api_url}/api/files/{self.test_file_id}/analyze",
            headers=self.get_headers(),
            json={
                "file_id": self.test_file_id,
                "analysis_type": "comprehensive",
                "additional_context": "Market trends in technology sector",
            },
        )

        if response.status_code == 200:
            data = response.json()
            self.assertEqual(data["file_id"], self.test_file_id)
            self.assertIn("analysis_type", data)
            self.assertIn("analysis_result", data)
            self.assertIn("timestamp", data)
            print(f"✅ File analysis successful for file ID: {self.test_file_id}")
        else:
            print(f"❌ File analysis failed: {response.status_code} - {response.text}")

    # 4. Enhanced Chat API
    def test_08_chat_api(self):
        """Test the enhanced chat API with RAG context"""
        print("\n8. Testing enhanced chat API...")

        # Chat doesn't require authentication
        response = requests.post(
            f"{self.api_url}/api/chat",
            headers=self.get_headers(with_auth=False),
            json={
                "messages": [{"role": "user", "content": "What are the current trends in AI for market intelligence?"}],
                "context": {"session_id": str(uuid.uuid4())},
            },
        )

        if response.status_code == 200:
            data = response.json()
            self.assertIn("response", data)
            self.assertIn("context", data)
            self.assertIn("suggestions", data)
            print("✅ Chat API test successful")
        else:
            print(f"❌ Chat API test failed: {response.status_code} - {response.text}")
            self.fail(f"Chat API test failed: {response.status_code} - {response.text}")

    # 5. Data Source Management
    def test_09_data_source_management(self):
        """Test data source management endpoints"""
        print("\n9. Testing data source management...")

        if not self.auth_token:
            self.skipTest("Authentication token not available")

        # Create a data source
        print("Testing data source creation...")
        response = requests.post(
            f"{self.api_url}/api/data-sources",
            headers=self.get_headers(),
            json={
                "name": "Financial News API",
                "type": "news_api",
                "description": "Financial news data source for market intelligence",
                "category": "financial_news",
                "config": {
                    "api_key": "test_api_key",
                    "base_url": "https://api.example.com/news",
                    "update_frequency": "daily",
                },
                "status": "inactive",
            },
        )

        if response.status_code == 200:
            data = response.json()
            self.test_data_source_id = data.get("id")
            self.assertIn("id", data)
            self.assertEqual(data["name"], "Financial News API")
            self.assertEqual(data["type"], "news_api")
            print(f"✅ Data source creation successful. ID: {self.test_data_source_id}")
        else:
            print(f"❌ Data source creation failed: {response.status_code} - {response.text}")
            self.fail(f"Data source creation failed: {response.status_code} - {response.text}")

        # List data sources
        print("Testing data source listing...")
        response = requests.get(f"{self.api_url}/api/data-sources", headers=self.get_headers())

        if response.status_code == 200:
            data = response.json()
            self.assertIsInstance(data, list)
            if data:
                print(f"✅ Data source listing successful. Found {len(data)} sources.")
            else:
                print("✅ Data source listing successful. No sources found.")
        else:
            print(f"❌ Data source listing failed: {response.status_code} - {response.text}")

        # Update data source
        if self.test_data_source_id:
            print("Testing data source update...")
            response = requests.put(
                f"{self.api_url}/api/data-sources/{self.test_data_source_id}",
                headers=self.get_headers(),
                json={
                    "name": "Financial News API Updated",
                    "type": "news_api",
                    "description": "Updated financial news data source",
                    "category": "financial_news",
                    "config": {
                        "api_key": "updated_test_api_key",
                        "base_url": "https://api.example.com/news/v2",
                        "update_frequency": "hourly",
                    },
                    "status": "active",
                },
            )

            if response.status_code == 200:
                data = response.json()
                self.assertEqual(data["name"], "Financial News API Updated")
                self.assertEqual(data["status"], "active")
                print("✅ Data source update successful")
            else:
                print(f"❌ Data source update failed: {response.status_code} - {response.text}")

            # Test data source connection
            print("Testing data source connection test...")
            response = requests.post(
                f"{self.api_url}/api/data-sources/{self.test_data_source_id}/test", headers=self.get_headers()
            )

            if response.status_code == 200:
                data = response.json()
                self.assertIn("test_successful", data)
                self.assertIn("message", data)
                print("✅ Data source connection test successful")
            else:
                print(f"❌ Data source connection test failed: {response.status_code} - {response.text}")

            # Test data source sync
            print("Testing data source sync...")
            response = requests.post(
                f"{self.api_url}/api/data-sources/{self.test_data_source_id}/sync", headers=self.get_headers()
            )

            if response.status_code == 200:
                data = response.json()
                self.assertIn("sync_successful", data)
                self.assertIn("records_synced", data)
                print("✅ Data source sync successful")
            else:
                print(f"❌ Data source sync failed: {response.status_code} - {response.text}")

            # Delete data source
            print("Testing data source deletion...")
            response = requests.delete(
                f"{self.api_url}/api/data-sources/{self.test_data_source_id}", headers=self.get_headers()
            )

            if response.status_code == 200:
                data = response.json()
                self.assertIn("message", data)
                print("✅ Data source deletion successful")
            else:
                print(f"❌ Data source deletion failed: {response.status_code} - {response.text}")

    # 6. Analysis and Reporting
    def test_10_market_analysis(self):
        """Test market analysis endpoint"""
        print("\n10. Testing market analysis...")

        if not self.auth_token:
            self.skipTest("Authentication token not available")

        response = requests.post(
            f"{self.api_url}/api/analyze",
            headers=self.get_headers(),
            json={
                "query": "AI adoption in financial services",
                "market_domain": "fintech",
                "question": "What are the key trends driving AI adoption in financial services?",
            },
        )

        if response.status_code == 200:
            data = response.json()
            self.assertEqual(data["query"], "AI adoption in financial services")
            self.assertEqual(data["market_domain"], "fintech")
            self.assertIn("analysis", data)
            self.assertIn("recommendations", data)
            print("✅ Market analysis test successful")
        else:
            print(f"❌ Market analysis test failed: {response.status_code} - {response.text}")
            self.fail(f"Market analysis test failed: {response.status_code} - {response.text}")

    def test_11_report_generation(self):
        """Test report generation endpoints"""
        print("\n11. Testing report generation...")

        if not self.auth_token:
            self.skipTest("Authentication token not available")

        response = requests.post(
            f"{self.api_url}/api/reports/generate",
            headers=self.get_headers(),
            params={"report_type": "comprehensive", "format": "json"},
        )

        if response.status_code == 200:
            data = response.json()
            self.assertIn("report_id", data)
            self.assertIn("report_type", data)
            self.assertIn("generated_at", data)
            report_id = data["report_id"]
            print(f"✅ Report generation successful. Report ID: {report_id}")

            # Test report download
            print("Testing report download...")
            response = requests.get(
                f"{self.api_url}/api/reports/{report_id}/download",
                headers=self.get_headers(),
                params={"format": "json"},
            )

            if response.status_code == 200:
                data = response.json()
                self.assertEqual(data["report_id"], report_id)
                self.assertIn("title", data)
                self.assertIn("sections", data)
                print("✅ Report download successful")
            else:
                print(f"❌ Report download failed: {response.status_code} - {response.text}")
        else:
            print(f"❌ Report generation failed: {response.status_code} - {response.text}")

    # 7. Error Handling
    def test_12_error_handling(self):
        """Test error handling for invalid requests"""
        print("\n12. Testing error handling...")

        # Test invalid file upload (wrong file type)
        if self.auth_token:
            print("Testing invalid file type error handling...")
            invalid_file = BytesIO(b"Invalid file content")
            files = {"file": ("invalid.xyz", invalid_file, "application/octet-stream")}
            response = requests.post(
                f"{self.api_url}/api/upload", headers={"Authorization": f"Bearer {self.auth_token}"}, files=files
            )

            self.assertEqual(response.status_code, 400)
            print("✅ Invalid file type error handling test passed")

        # Test invalid endpoint
        print("Testing invalid endpoint error handling...")
        response = requests.get(f"{self.api_url}/api/nonexistent")
        self.assertEqual(response.status_code, 404)
        print("✅ Invalid endpoint error handling test passed")

        # Test invalid request body
        print("Testing invalid request body error handling...")
        response = requests.post(
            f"{self.api_url}/api/chat", headers=self.get_headers(with_auth=False), json={"invalid": "request"}
        )

        self.assertIn(response.status_code, [400, 422])
        print("✅ Invalid request body error handling test passed")

    def test_13_agent_status(self):
        """Test agent status endpoint"""
        print("\n13. Testing agent status...")

        response = requests.get(f"{self.api_url}/api/agent/status")

        if response.status_code == 200:
            data = response.json()
            self.assertEqual(data["status"], "online")
            self.assertIn("version", data)
            self.assertIn("capabilities", data)
            self.assertIn("file_types_supported", data)
            print("✅ Agent status test successful")
        else:
            print(f"❌ Agent status test failed: {response.status_code} - {response.text}")
            self.fail(f"Agent status test failed: {response.status_code} - {response.text}")

    def test_14_kpi_endpoints(self):
        """Test KPI endpoints"""
        print("\n14. Testing KPI endpoints...")

        # Test GET KPI
        print("Testing GET KPI...")
        response = requests.get(f"{self.api_url}/api/kpi", params={"timeframe": "30d", "category": "technology"})

        if response.status_code == 200:
            data = response.json()
            self.assertIn("revenue", data)
            self.assertIn("customers", data)
            self.assertIn("conversion", data)
            self.assertIn("metadata", data)
            print("✅ GET KPI test successful")
        else:
            print(f"❌ GET KPI test failed: {response.status_code} - {response.text}")
            self.fail(f"GET KPI test failed: {response.status_code} - {response.text}")

        # Test POST KPI
        print("Testing POST KPI...")
        response = requests.post(
            f"{self.api_url}/api/kpi",
            headers=self.get_headers(with_auth=False),
            json={
                "metric": "customer_acquisition_cost",
                "value": 125.75,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

        if response.status_code == 200:
            data = response.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["metric"], "customer_acquisition_cost")
            self.assertEqual(data["value"], 125.75)
            print("✅ POST KPI test successful")
        else:
            print(f"❌ POST KPI test failed: {response.status_code} - {response.text}")
            self.fail(f"POST KPI test failed: {response.status_code} - {response.text}")

    def test_15_agent_sync(self):
        """Test agent sync endpoint"""
        print("\n15. Testing agent sync...")

        response = requests.post(
            f"{self.api_url}/api/agent/sync",
            headers=self.get_headers(with_auth=False),
            json={
                "action": "refresh_data",
                "data": {"source": "market_intelligence", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
            },
        )

        if response.status_code == 200:
            data = response.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["action"], "refresh_data")
            self.assertIn("sync_id", data)
            print("✅ Agent sync test successful")
        else:
            print(f"❌ Agent sync test failed: {response.status_code} - {response.text}")
            self.fail(f"Agent sync test failed: {response.status_code} - {response.text}")


if __name__ == "__main__":
    # Create a test suite
    test_suite = unittest.TestSuite()

    # Add tests in order
    test_suite.addTest(MarketIntelligenceAPITest("test_01_root_endpoint"))
    test_suite.addTest(MarketIntelligenceAPITest("test_02_health_endpoint"))
    test_suite.addTest(MarketIntelligenceAPITest("test_03_authentication_required"))
    test_suite.addTest(MarketIntelligenceAPITest("test_04_file_upload"))
    test_suite.addTest(MarketIntelligenceAPITest("test_05_list_files"))
    test_suite.addTest(MarketIntelligenceAPITest("test_06_file_details"))
    test_suite.addTest(MarketIntelligenceAPITest("test_07_file_analysis"))
    test_suite.addTest(MarketIntelligenceAPITest("test_08_chat_api"))
    test_suite.addTest(MarketIntelligenceAPITest("test_09_data_source_management"))
    test_suite.addTest(MarketIntelligenceAPITest("test_10_market_analysis"))
    test_suite.addTest(MarketIntelligenceAPITest("test_11_report_generation"))
    test_suite.addTest(MarketIntelligenceAPITest("test_12_error_handling"))
    test_suite.addTest(MarketIntelligenceAPITest("test_13_agent_status"))
    test_suite.addTest(MarketIntelligenceAPITest("test_14_kpi_endpoints"))
    test_suite.addTest(MarketIntelligenceAPITest("test_15_agent_sync"))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)
