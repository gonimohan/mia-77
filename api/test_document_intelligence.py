import unittest
from unittest.mock import patch, MagicMock, ANY
from fastapi.testclient import TestClient
from fastapi import UploadFile, BackgroundTasks
import os
import tempfile
from pathlib import Path
import shutil

# Add api directory to sys.path to allow local imports if necessary for testing
# This might be needed if tests are run from a different working directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

# Import the FastAPI app and other necessary components
# Assuming 'app' is the FastAPI instance in main.py
# Assuming database and text_processor are accessible for mocking or direct import
try:
    from main import app # FastAPI app instance
    from database import get_db, insert_document, get_document_by_id, update_document_by_id
    from text_processor import extract_text_from_file, analyze_text_keywords
except ImportError as e:
    print(f"Error importing modules for testing: {e}. Ensure your main app and modules are structured correctly.")
    # Fallback for running tests if main.py is complex to import directly in test env
    # This is a common issue with FastAPI app structure and testing.
    # A more robust solution involves a conftest.py or specific app factory for tests.
    app = None
    get_db = MagicMock()
    insert_document = MagicMock()
    get_document_by_id = MagicMock()
    update_document_by_id = MagicMock()
    extract_text_from_file = MagicMock()
    analyze_text_keywords = MagicMock()


# Create a TestClient instance if app was imported
client = TestClient(app) if app else None

# Temp directory for creating test files
TEST_FILES_DIR = Path(tempfile.mkdtemp(prefix="test_doc_intel_"))

def create_dummy_file(filename: str, content: str = "dummy content", extension: str = ".txt"):
    filepath = TEST_FILES_DIR / f"{filename}{extension}"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath

def create_dummy_pdf(filename: str, text_content: str = "Sample PDF content."):
    filepath = TEST_FILES_DIR / f"{filename}.pdf"
    try:
        import fitz # PyMuPDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 72), text_content)
        doc.save(filepath)
        doc.close()
        return filepath
    except ImportError:
        print("PyMuPDF (fitz) not installed, cannot create dummy PDF for testing text_processor. Skipping PDF creation.")
        # Create an empty file as a placeholder if fitz is not available
        open(filepath, 'a').close()
        return filepath
    except Exception as e:
        print(f"Error creating dummy PDF: {e}")
        open(filepath, 'a').close()
        return filepath


def create_dummy_docx(filename: str, text_content: str = "Sample DOCX content."):
    filepath = TEST_FILES_DIR / f"{filename}.docx"
    try:
        from docx import Document
        doc = Document()
        doc.add_paragraph(text_content)
        doc.save(filepath)
        return filepath
    except ImportError:
        print("python-docx not installed, cannot create dummy DOCX for testing text_processor. Skipping DOCX creation.")
        open(filepath, 'a').close()
        return filepath
    except Exception as e:
        print(f"Error creating dummy DOCX: {e}")
        open(filepath, 'a').close()
        return filepath

def create_dummy_csv(filename: str, data=None):
    filepath = TEST_FILES_DIR / f"{filename}.csv"
    if data is None:
        data = {"col1": ["val1", "val2"], "col2": ["infoA", "infoB"]}
    try:
        import pandas as pd
        pd.DataFrame(data).to_csv(filepath, index=False)
        return filepath
    except ImportError:
        print("pandas not installed, cannot create dummy CSV. Skipping.")
        open(filepath, 'a').close()
        return filepath

def create_dummy_xlsx(filename: str, data=None):
    filepath = TEST_FILES_DIR / f"{filename}.xlsx"
    if data is None:
        data = {"Sheet1": {"col1": ["val1"], "col2": ["infoA"]}}
    try:
        import pandas as pd
        with pd.ExcelWriter(filepath) as writer:
            for sheet_name, sheet_data in data.items():
                pd.DataFrame(sheet_data).to_excel(writer, sheet_name=sheet_name, index=False)
        return filepath
    except ImportError:
        print("pandas or openpyxl not installed, cannot create dummy XLSX. Skipping.")
        open(filepath, 'a').close()
        return filepath

class TestTextProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TEST_FILES_DIR.exists():
            TEST_FILES_DIR.mkdir(parents=True, exist_ok=True)

        cls.dummy_txt_path = create_dummy_file("sample_text", "This is a test text with market trends.")
        cls.dummy_pdf_path = create_dummy_pdf("sample_pdf", "PDF content discussing competitive landscape.")
        cls.dummy_docx_path = create_dummy_docx("sample_docx", "DOCX file about customer behavior.")
        cls.dummy_csv_path = create_dummy_csv("sample_csv", {"header1": ["growth strategy data1"], "header2": ["other data1"]})
        cls.dummy_xlsx_path = create_dummy_xlsx("sample_xlsx", {"Sheet1": {"colA": ["xlsx growth strategy"], "colB": ["value"]}})


    def test_extract_text_from_txt(self):
        if not self.dummy_txt_path.exists() or os.path.getsize(self.dummy_txt_path) == 0 :
            self.skipTest(f"TXT file {self.dummy_txt_path} not created or empty, skipping test.")
        result = extract_text_from_file(str(self.dummy_txt_path), ".txt")
        self.assertIsNotNone(result)
        self.assertIn("test text", result["text"])
        self.assertEqual(result["word_count"], 7)

    def test_extract_text_from_pdf(self):
        if not self.dummy_pdf_path.exists() or os.path.getsize(self.dummy_pdf_path) == 0 :
            self.skipTest(f"PDF file {self.dummy_pdf_path} not created or empty, skipping test.")
        result = extract_text_from_file(str(self.dummy_pdf_path), ".pdf")
        self.assertIsNotNone(result)
        # Exact text depends on PyMuPDF's rendering, so check for keywords
        self.assertIn("pdf content", result["text"].lower())
        self.assertGreater(result["word_count"], 2)


    def test_extract_text_from_docx(self):
        if not self.dummy_docx_path.exists() or os.path.getsize(self.dummy_docx_path) == 0 :
            self.skipTest(f"DOCX file {self.dummy_docx_path} not created or empty, skipping test.")
        result = extract_text_from_file(str(self.dummy_docx_path), ".docx")
        self.assertIsNotNone(result)
        self.assertIn("docx file", result["text"].lower())
        self.assertGreater(result["word_count"], 3)

    def test_extract_text_from_csv(self):
        if not self.dummy_csv_path.exists() or os.path.getsize(self.dummy_csv_path) == 0 :
            self.skipTest(f"CSV file {self.dummy_csv_path} not created or empty, skipping test.")
        result = extract_text_from_file(str(self.dummy_csv_path), ".csv")
        self.assertIsNotNone(result)
        self.assertIn("growth strategy data1", result["text"].lower())
        self.assertIn("other data1", result["text"].lower())
        self.assertGreaterEqual(result["word_count"], 4) # "growth strategy data1 other data1"

    def test_extract_text_from_xlsx(self):
        if not self.dummy_xlsx_path.exists() or os.path.getsize(self.dummy_xlsx_path) == 0 :
            self.skipTest(f"XLSX file {self.dummy_xlsx_path} not created or empty, skipping test.")
        result = extract_text_from_file(str(self.dummy_xlsx_path), ".xlsx")
        self.assertIsNotNone(result)
        self.assertIn("xlsx growth strategy", result["text"].lower())
        self.assertGreaterEqual(result["word_count"], 3)


    def test_analyze_text_keywords(self):
        text_with_keywords = "This document discusses market trends and the competitive landscape."
        result = analyze_text_keywords(text_with_keywords)
        self.assertTrue(result["market trends"])
        self.assertTrue(result["competitive landscape"])
        self.assertFalse(result["customer behavior"])
        self.assertFalse(result["growth strategy"])

        text_without_keywords = "This is a generic document."
        result = analyze_text_keywords(text_without_keywords)
        self.assertFalse(result["market trends"])
        self.assertFalse(result["competitive landscape"])

    @classmethod
    def tearDownClass(cls):
        # Clean up the temporary directory and files
        # Check if directory exists before removing
        if TEST_FILES_DIR.exists():
            try:
                shutil.rmtree(TEST_FILES_DIR)
            except Exception as e:
                print(f"Error removing test directory {TEST_FILES_DIR}: {e}")


@patch('main.database', autospec=True) # Mock the entire database module used by main
@patch('main.text_processor', autospec=True) # Mock the text_processor module
class TestAPIEndpoints(unittest.TestCase):

    def setUp(self):
        if not client:
            self.skipTest("FastAPI TestClient not initialized. Skipping API tests.")
        # Ensure UPLOAD_DIR exists for endpoint tests, using a temporary one for isolation
        self.test_upload_dir = Path(tempfile.mkdtemp(prefix="test_api_uploads_"))
        # If main.UPLOAD_DIR is used directly, we might need to patch it or ensure it's writable
        # For now, assuming main.UPLOAD_DIR is correctly set and writable in test env or mocked.
        # If your app creates UPLOAD_DIR, that's fine. If not, create it:
        # if hasattr(app, 'UPLOAD_DIR'):
        #    app.UPLOAD_DIR.mkdir(exist_ok=True)
        # else:
        #    Path("/app/uploads").mkdir(exist_ok=True) # Default from main.py

    def tearDown(self):
        if self.test_upload_dir.exists():
            try:
                shutil.rmtree(self.test_upload_dir)
            except Exception as e:
                print(f"Error removing API test upload directory {self.test_upload_dir}: {e}")


    @patch('main.BackgroundTasks', MagicMock()) # Mock BackgroundTasks
    @patch('main.get_current_user') # Mock user dependency
    def test_upload_file_success(self, mock_get_current_user, mock_db_ops, mock_text_processor_ops):
        mock_user = MagicMock()
        mock_user.id = "test_user_123"
        mock_get_current_user.return_value = mock_user

        mock_db_ops.insert_document.return_value = "mock_document_id_123"

        # Simulate file upload
        dummy_file_path = create_dummy_file("api_test_upload", "API test content", ".txt")

        with open(dummy_file_path, "rb") as f:
            response = client.post(
                "/api/upload",
                files={"file": ("api_test_upload.txt", f, "text/plain")}
            )

        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["message"], "File uploaded successfully. Processing started in background.")
        self.assertEqual(json_response["document_id"], "mock_document_id_123")
        self.assertEqual(json_response["original_filename"], "api_test_upload.txt")

        mock_db_ops.insert_document.assert_called_once()
        args, kwargs = mock_db_ops.insert_document.call_args
        self.assertEqual(args[0]['original_filename'], "api_test_upload.txt")
        self.assertEqual(args[0]['uploader_id'], "test_user_123")
        self.assertEqual(args[0]['status'], "uploaded")

        # Check that background task was added (by checking if the pipeline was called via mock if not using BackgroundTasks directly)
        # If BackgroundTasks is directly used and mocked, its add_task method would be checked.
        # For now, we assume the call to background_tasks.add_task happened.
        # The actual process_document_pipeline is complex to test here without more intricate mocking of its internals.
        # We are essentially testing that the endpoint tries to start it.

    @patch('main.get_current_user')
    def test_upload_file_invalid_type(self, mock_get_current_user, mock_db_ops, mock_text_processor_ops):
        mock_user = MagicMock()
        mock_user.id = "test_user_123"
        mock_get_current_user.return_value = mock_user

        dummy_file_path = create_dummy_file("invalid_type", "content", ".zip")
        with open(dummy_file_path, "rb") as f:
            response = client.post(
                "/api/upload",
                files={"file": ("invalid_type.zip", f, "application/zip")}
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json()["detail"])


    def test_generate_report_success(self, mock_db_ops, mock_text_processor_ops):
        mock_doc_id = "analyzed_doc_1"
        mock_db_ops.get_document_by_id.return_value = {
            "_id": mock_doc_id,
            "original_filename": "report_me.pdf",
            "upload_time": "2023-01-01T10:00:00Z",
            "word_count": 150,
            "analysis": {"market trends": True, "customer behavior": False},
            "text_preview": "This is a preview...",
            "status": "analyzed"
        }
        response = client.get(f"/api/agent/generate-report/{mock_doc_id}")
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["original_filename"], "report_me.pdf")
        self.assertEqual(json_response["word_count"], 150)
        self.assertTrue(json_response["analysis"]["market trends"])

    def test_generate_report_not_found(self, mock_db_ops, mock_text_processor_ops):
        mock_doc_id = "non_existent_doc_1"
        mock_db_ops.get_document_by_id.return_value = None
        response = client.get(f"/api/agent/generate-report/{mock_doc_id}")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Document not found.")

    def test_generate_report_not_analyzed(self, mock_db_ops, mock_text_processor_ops):
        mock_doc_id = "processing_doc_1"
        mock_db_ops.get_document_by_id.return_value = {
            "_id": mock_doc_id,
            "original_filename": "processing.pdf",
            "status": "processing" # or "uploaded", "text_extracted"
        }
        response = client.get(f"/api/agent/generate-report/{mock_doc_id}")
        self.assertEqual(response.status_code, 422) # Unprocessable Entity
        self.assertIn("Document processing not complete", response.json()["detail"])


if __name__ == "__main__":
    # Note: Running this directly might have issues with FastAPI app discovery
    # depending on how 'main.app' is structured and PYTHONPATH.
    # It's often better to run tests via a test runner like 'python -m unittest api.test_document_intelligence'
    # from the repository root, or 'pytest'.
    if app is None:
        print("Skipping tests as FastAPI app could not be loaded. Ensure main.py and imports are correct.")
    else:
      unittest.main()
