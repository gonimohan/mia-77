from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import io
import uuid
from datetime import datetime
import traceback
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import pandas as pd
from supabase import create_client, Client
from pathlib import Path
import json
import database # For Supabase operations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Market Intelligence Agent API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Optional[Client] = None

if supabase_url and supabase_service_key:
    try:
        supabase = create_client(supabase_url, supabase_service_key)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
else:
    logger.warning("Supabase credentials not found in environment variables")

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# Pydantic models
class AnalysisRequest(BaseModel):
    query: str
    market_domain: str
    question: Optional[str] = None
    uploaded_file_ids: Optional[List[str]] = None


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    context: Optional[Dict[str, Any]] = None


class AgentSyncRequest(BaseModel):
    action: str
    data: Optional[Dict[str, Any]] = None


class KPIRequest(BaseModel):
    metric: str
    value: float
    timestamp: Optional[str] = None


class DataSource(BaseModel):
    name: str
    type: str
    description: Optional[str] = None
    category: Optional[str] = None
    config: Dict[str, Any] = {}
    status: str = "inactive"


class FileAnalysisRequest(BaseModel):
    file_id: str
    analysis_type: str = "comprehensive"
    additional_context: Optional[str] = None


class UserProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    # Add other updatable fields as necessary


class UserPreferences(BaseModel):
    theme_settings: Optional[Dict[str, Any]] = None
    notification_settings: Optional[Dict[str, Any]] = None
    data_settings: Optional[Dict[str, Any]] = None


class AppSettingItem(BaseModel):
    setting_key: str
    setting_value: Dict[str, Any]  # Assuming value is always JSON for flexibility
    description: Optional[str] = None


class AppSettingsUpdateRequest(BaseModel):
    settings: List[AppSettingItem]


# Authentication dependency
def get_current_user(request: Request):
    # Extract JWT token from Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    token = auth_header.split(" ")[1]

    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")

    try:
        # Verify JWT token with Supabase
        user = supabase.auth.get_user(token)
        return user.user
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Invalid authentication token")


# File processing utilities
class FileProcessor:
    @staticmethod
    def process_csv(file_content: bytes) -> Dict[str, Any]:
        """Process CSV file and return structured data"""
        try:
            df = pd.read_csv(io.BytesIO(file_content))
            return {
                "type": "csv",
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "summary": df.describe().to_dict(),
                "sample_data": df.head(10).to_dict(orient="records"),
                "null_counts": df.isnull().sum().to_dict(),
            }
        except Exception as e:
            logger.error(f"CSV processing error: {e}")
            raise HTTPException(status_code=400, detail=f"CSV processing failed: {str(e)}")

    @staticmethod
    def process_excel(file_content: bytes) -> Dict[str, Any]:
        """Process Excel file and return structured data"""
        try:
            # Read all sheets
            xl_file = pd.ExcelFile(io.BytesIO(file_content))
            sheets_data = {}

            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)
                sheets_data[sheet_name] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "data_types": df.dtypes.astype(str).to_dict(),
                    "summary": df.describe().to_dict(),
                    "sample_data": df.head(5).to_dict(orient="records"),
                    "null_counts": df.isnull().sum().to_dict(),
                }

            return {"type": "excel", "sheets": list(xl_file.sheet_names), "sheets_data": sheets_data}
        except Exception as e:
            logger.error(f"Excel processing error: {e}")
            raise HTTPException(status_code=400, detail=f"Excel processing failed: {str(e)}")

    @staticmethod
    def process_text(file_content: bytes) -> Dict[str, Any]:
        """Process text file and return analysis"""
        try:
            text = file_content.decode("utf-8", errors="ignore")
            lines = text.split("\n")
            words = text.split()

            return {
                "type": "text",
                "character_count": len(text),
                "word_count": len(words),
                "line_count": len(lines),
                "sample_content": text[:500] + "..." if len(text) > 500 else text,
                "encoding": "utf-8",
            }
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            raise HTTPException(status_code=400, detail=f"Text processing failed: {str(e)}")

    @staticmethod
    def process_pdf(file_content: bytes, temp_file_path: str = None) -> Dict[str, Any]:
        """Process PDF file with full text extraction"""
        try:
            # Save content to temporary file for processing
            if not temp_file_path:
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                    tmp_file.write(file_content)
                    temp_file_path = tmp_file.name

            # Import text processor
            # Assuming text_processor.py functions are still relevant or agent_logic handles this.
            # For now, keeping the import if process_pdf is still used directly.
            from .text_processor import extract_text_from_file, analyze_text_keywords, extract_entities

            # Extract text
            extraction_result = extract_text_from_file(temp_file_path, ".pdf")

            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

            if extraction_result:
                # Analyze the extracted text
                keyword_analysis = analyze_text_keywords(extraction_result["text"])
                entities = extract_entities(extraction_result["text"])

                return {
                    "type": "pdf",
                    "size_bytes": len(file_content),
                    "text_extracted": (
                        extraction_result["text"][:1000] + "..."
                        if len(extraction_result["text"]) > 1000
                        else extraction_result["text"]
                    ),
                    "word_count": extraction_result["word_count"],
                    "metadata": extraction_result.get("metadata", {}),
                    "keyword_analysis": keyword_analysis,
                    "entities": entities,
                    "processing_status": "completed",
                    "text_quality": keyword_analysis.get("text_quality", "unknown"),
                }
            else:
                return {
                    "type": "pdf",
                    "size_bytes": len(file_content),
                    "note": "PDF text extraction failed. Document may be image-based or corrupted.",
                    "processing_status": "failed",
                    "text_quality": "poor",
                }
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            raise HTTPException(status_code=400, detail=f"PDF processing failed: {str(e)}")


# Market Intelligence AI Agent
class MarketIntelligenceAgent:
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
    async def generate_insights(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate AI-powered market intelligence insights"""
        try:
            # Simulated AI analysis for now
            # In production, integrate with actual LLM APIs
            insights = {
                "query": query,
                "insights": [
                    f"Market analysis for '{query}' shows emerging trends in digital transformation",
                    f"Competitive landscape analysis reveals 3 key players in the {context.get('market_domain', 'general')} space",
                    f"Growth opportunities identified in {context.get('market_domain', 'target market')} segment",
                    f"Risk factors include market volatility and regulatory changes"
                ],
                "recommendations": [
                    "Focus on digital-first approach to capture emerging market segments",
                    "Invest in customer experience improvements",
                    "Monitor competitive pricing strategies closely",
                    "Diversify market presence to reduce concentration risk"
                ],
                "confidence_score": 0.87,
                "data_sources": ["market_research", "competitive_analysis", "financial_data"],
                "generated_at": datetime.now().isoformat()
            }
            
            return insights
        except Exception as e:
            logger.error(f"AI insights generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")

    async def process_file_with_ai(self, file_data: Dict[str, Any], query: str = None) -> Dict[str, Any]:
        """Process uploaded file data with AI analysis"""
        try:
            analysis = {
                "file_type": file_data.get("type"),
                "processing_timestamp": datetime.now().isoformat(),
                "ai_insights": [],
                "recommendations": [],
                "visualizations": []
            }
            
            if file_data.get("type") == "csv":
                # Analyze CSV data
                analysis["ai_insights"] = [
                    f"Dataset contains {file_data.get('rows', 0)} records across {file_data.get('columns', 0)} dimensions",
                    "Data quality assessment: " + ("High" if file_data.get('null_counts', {}) else "Moderate"),
                    "Potential for trend analysis and predictive modeling identified"
                ]
                
                analysis["recommendations"] = [
                    "Consider time-series analysis if temporal data is present",
                    "Implement data validation for missing values",
                    "Explore correlation patterns between key variables"
                ]
            
            elif file_data.get("type") == "text":
                # Analyze text content
                word_count = file_data.get("word_count", 0)
                analysis["ai_insights"] = [
                    f"Document contains {word_count} words across {file_data.get('line_count', 0)} lines",
                    "Text complexity: " + ("High" if word_count > 1000 else "Moderate"),
                    "Suitable for sentiment analysis and content categorization"
                ]
                
                analysis["recommendations"] = [
                    "Perform sentiment analysis to gauge market perception",
                    "Extract key entities and topics for market intelligence",
                    "Consider competitive mention analysis"
                ]
                
            return analysis
            
        except Exception as e:
            logger.error(f"File AI processing error: {e}")
            raise HTTPException(status_code=500, detail=f"AI file processing failed: {str(e)}")

# Initialize AI agent
ai_agent = MarketIntelligenceAgent()

# Initialize Supabase connection on startup - This is good.
@app.on_event("startup")
async def startup_db_client():
    database.connect_to_supabase()  # Ensures from .database is called


@app.on_event("shutdown")
async def shutdown_db_client():
    logger.info("Shutting down Market Intelligence Agent API")


# Import real agent functions
from api import agent_logic # Adjusted for clarity if main.py is run from outside api dir sometimes
                            # but for uvicorn from api dir, `import agent_logic` or `from . import agent_logic` is fine.
                            # Let's stick to `from api import agent_logic` for robustness in various run contexts.
                            # Or, more standardly for a package: `from . import agent_logic`

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "details": str(exc), "timestamp": datetime.now().isoformat()},
    )


@app.get("/")
async def root():
    return {
        "message": "Market Intelligence Agent API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "features": ["market_analysis", "file_processing", "rag_chat", "data_integration", "ai_insights"],
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "database": "connected" if supabase else "not_configured",
            "ai_agent": "ready",
            "file_processor": "ready",
        },
    }


@app.post("/api/analyze")
async def analyze(analysis_request: AnalysisRequest, user=Depends(get_current_user)):
    try:
        logger.info(f"Analysis request received: Query='{analysis_request.query}', Domain='{analysis_request.market_domain}', UserID='{user.id}', FileIDs='{analysis_request.uploaded_file_ids}'")

        # Call the actual agent logic
        # Ensure agent_logic is imported correctly at the top of the file
        # e.g., from . import agent_logic or from api import agent_logic
        agent_result = await agent_logic.run_market_intelligence_agent(
            query_str=analysis_request.query,
            market_domain_str=analysis_request.market_domain,
            question_str=analysis_request.question,
            user_id=str(user.id),
            uploaded_document_ids=analysis_request.uploaded_file_ids
        )

        report_status = "completed" if agent_result.get("success") else "failed"
        report_file_path = None
        if agent_result.get("success") and agent_result.get("report_dir_relative") and agent_result.get("report_filename"):
            # Construct a conceptual path. Actual storage/retrieval might differ.
            report_file_path = f"{agent_result.get('report_dir_relative')}/{agent_result.get('report_filename')}"
        
        report_title = f"Analysis for '{analysis_request.query}' in '{analysis_request.market_domain}'"

        # Store a reference in the 'reports' table
        if supabase:
            report_record = {
                "user_id": str(user.id),
                "title": report_title[:255], # Ensure title fits in VARCHAR(255)
                "market_domain": analysis_request.market_domain,
                "query_text": analysis_request.query,
                "status": report_status,
                "report_data": { # Store key results or references
                    "state_id": agent_result.get("state_id"),
                    "agent_query_response": agent_result.get("query_response"),
                    "error_message": agent_result.get("error") if not agent_result.get("success") else None,
                    "chart_filenames": agent_result.get("chart_filenames"),
                    "download_files": agent_result.get("download_files")
                },
                "file_path": report_file_path, # Path to the main markdown report
                # created_at and updated_at will be set by DB defaults
            }
            try:
                insert_op = supabase.table("reports").insert(report_record).execute()
                if insert_op.data:
                    logger.info(f"Report record stored in Supabase 'reports' table with ID: {insert_op.data[0]['id']}")
                    # Return the agent_result which includes state_id etc.
                    # The frontend might use state_id to later fetch report details or files.
                    return {
                        "success": agent_result.get("success"),
                        "message": "Analysis processing initiated." if agent_result.get("success") else "Analysis processing failed.",
                        "state_id": agent_result.get("state_id"),
                        "report_db_id": insert_op.data[0]['id'], # ID from 'reports' table
                        "details": agent_result # Full agent result for client if needed
                    }
                else:
                    logger.error(f"Failed to store report record in Supabase 'reports' table: {insert_op.error.message if insert_op.error else 'Unknown error'}")
                    # Even if DB store fails, the agent might have run. Return agent status.
                    # Or raise an error if this DB store is critical for the flow.
                    # For now, log error and proceed with agent result.
                    # Fall through to return agent_result directly but log the DB error.
            except Exception as db_error:
                logger.error(f"Exception storing report record to Supabase 'reports': {db_error}\n{traceback.format_exc()}")
                # Fall through to return agent_result, but with a warning about DB persistence.

        # Fallback return if Supabase interaction failed but agent ran
        return {
            "success": agent_result.get("success"),
            "message": "Analysis processing completed but report metadata might not have been saved.",
            "state_id": agent_result.get("state_id"),
            "details": agent_result
        }

    except Exception as e:
        logger.error(f"Error in /api/analyze endpoint: {str(e)}\n{traceback.format_exc()}")
        # Attempt to store a 'failed' record in reports table even if agent run fails early
        if supabase and user and analysis_request: # Ensure we have necessary info
             try:
                report_title_on_error = f"Failed Analysis for '{analysis_request.query}'"
                supabase.table("reports").insert({
                    "user_id": str(user.id),
                    "title": report_title_on_error[:255],
                    "market_domain": analysis_request.market_domain,
                    "query_text": analysis_request.query,
                    "status": "failed",
                    "report_data": {"error_message": str(e)[:1000]} # Store snippet of error
                }).execute()
             except Exception as db_e_on_fail:
                 logger.error(f"Failed to store error report: {db_e_on_fail}")

        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/reports", response_model=List[Dict[str, Any]])
async def list_reports_for_user(user=Depends(get_current_user), limit: int = 50, offset: int = 0):
    """
    Lists all analysis reports for the authenticated user from the 'reports' table.
    """
    logger.info(f"GET /api/reports: UserID='{user.id}', Limit={limit}, Offset={offset}")
    if not supabase:
        raise HTTPException(status_code=500, detail="Database connection not available.")

    try:
        # Select specific fields to return for the list view.
        # Include 'id' (report_db_id), 'title', 'market_domain', 'status', 'created_at'.
        # Also extract 'state_id' from 'report_data' JSONB field.
        # Note: Accessing JSONB fields in select might vary slightly based on Supabase/Postgres version or client library features.
        # The `->>` operator casts JSONB value to text.
        reports_response = supabase.table("reports") \
            .select("id, title, market_domain, status, created_at, report_data->>state_id") \
            .eq("user_id", str(user.id)) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .offset(offset) \
            .execute()

        if reports_response.data:
            # Rename 'report_data->>state_id' to 'state_id' in the response if needed by frontend,
            # or ensure frontend can handle the default naming if direct.
            # For now, let's assume the default naming is acceptable or frontend adapts.
            # If a direct rename is needed:
            # formatted_reports = []
            # for report in reports_response.data:
            #     new_report = {**report}
            #     new_report["state_id"] = new_report.pop("report_data->>state_id", None) # Example rename
            #     formatted_reports.append(new_report)
            # return formatted_reports
            return reports_response.data
        else:
            return [] # Return empty list if no reports found or error in data

    except Exception as e:
        logger.error(f"Error listing reports for user {user.id}: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")


@app.post("/api/chat")
async def chat(chat_request: ChatRequest, user=Depends(get_current_user)):  # Added user dependency for user_id
    try:
        logger.info(f"Chat request with {len(request.messages)} messages")

        last_message = request.messages[-1] if request.messages else {}
        user_content = last_message.get("content", "")
        session_id = request.context.get("session_id") if request.context else None

        # Enhanced RAG-powered response generation
        # In production, this would integrate with vector databases and retrieval systems
        context_info = ""
        if request.context:
            context_info = f" (Session: {session_id})"

        # Generate AI response using market intelligence agent
        ai_response = await ai_agent.generate_insights(
            user_content,
            {"type": "chat", "session_id": session_id, "context": request.context}
        )

        response = {
            "response": f"Based on my market intelligence analysis{context_info}, here are insights about '{user_content}': " + 
                       " ".join(ai_response.get("insights", ["I can help you with market analysis and insights."])[:2]),
            "context": {
                "message_count": len(request.messages),
                "query_type": "market_intelligence",
                "confidence": ai_response.get("confidence_score", 0.92),
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id
            },
            "suggestions": ai_response.get("recommendations", [
                "Would you like more details about market trends?",
                "Should I analyze competitor positioning?",
                "Do you need strategic recommendations?"
            ])[:3]
        }

        return response
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


async def process_document_pipeline(
    document_id: str, internal_filename: str, original_filename: str, file_extension: str, saved_file_path: str
):
    """
    Background task to process a document: extract text, analyze keywords, and update database.
    """
    logger.info(
        f"Background task started for document_id: {document_id}, file: {original_filename} (path: {saved_file_path})"
    )
    try:
        # 1. Update status to processing
        if not supabase:
            logger.error("Supabase client not available for document processing")
            return

        update_result = supabase.table("documents").update({"status": "processing"}).eq("id", document_id).execute()
        if not update_result.data:
            logger.error(f"Failed to update status to 'processing' for document_id: {document_id}. Aborting pipeline.")
            return

        # 2. Full text extraction and analysis
        from .text_processor import extract_text_from_file, analyze_text_keywords, extract_entities

        extraction_result = extract_text_from_file(saved_file_path, file_extension)

        if extraction_result:
            # Extract full text and metadata
            extracted_text = extraction_result["text"]
            word_count = extraction_result["word_count"]
            metadata = extraction_result.get("metadata", {})

            # Perform keyword analysis
            keyword_analysis = analyze_text_keywords(extracted_text)

            # Extract entities
            entities = extract_entities(extracted_text)

            # Create text preview (first 500 characters)
            text_preview = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text

            # Comprehensive analysis
            comprehensive_analysis = {
                "file_type": file_extension,
                "file_size": os.path.getsize(saved_file_path),
                "processing_timestamp": datetime.now().isoformat(),
                "extraction_metadata": metadata,
                "keyword_analysis": keyword_analysis,
                "entities": entities,
                "text_quality": keyword_analysis.get("text_quality", "unknown"),
                "quality_score": keyword_analysis.get("quality_score", 0),
                "business_relevance": keyword_analysis.get("matched_categories_count", 0),
                "processing_method": metadata.get("processing_method", "unknown"),
            }

            # Update document with full analysis
            update_data = {
                "text": extracted_text,
                "word_count": word_count,
                "analysis": comprehensive_analysis,
                "text_preview": text_preview,
                "status": "analyzed",
            }
        else:
            # If text extraction failed, still provide basic analysis
            basic_analysis = {
                "file_type": file_extension,
                "file_size": os.path.getsize(saved_file_path),
                "processing_timestamp": datetime.now().isoformat(),
                "error": "Text extraction failed",
                "text_quality": "poor",
            }

            update_data = {
                "analysis": basic_analysis,
                "status": "processing_failed",
                "error_message": "Text extraction failed",
            }

        update_result = supabase.table("documents").update(update_data).eq("id", document_id).execute()

        if not update_result.data:
            logger.error(f"Failed to update DB after processing for document_id: {document_id}.")
        else:
            logger.info(
                f"Successfully processed document_id: {document_id} with {'full analysis' if extraction_result else 'basic info'}"
            )

    except Exception as e:
        logger.error(f"Error in processing pipeline for document_id {document_id}: {e}\n{traceback.format_exc()}")
        if supabase:
            supabase.table("documents").update({"status": "processing_failed", "error_message": str(e)}).eq(
                "id", document_id
            ).execute()


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".xlsx"}
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@app.post("/api/upload")
async def upload_document_for_intelligence(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user=Depends(get_current_user),  # Assuming get_current_user provides user object with an id attribute
):
    """
    Uploads a document, stores metadata in MongoDB, and triggers a background task
    for text extraction and analysis.
    """
    original_filename = file.filename
    file_extension = Path(original_filename).suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file_extension}'. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read file content to check size and save
    file_content = await file.read()
    file_size = len(file_content)

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,  # Payload Too Large
            detail=f"File size {file_size / (1024*1024):.2f}MB exceeds limit of {MAX_FILE_SIZE_MB}MB.",
        )

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Cannot upload empty file.")

    # Generate a unique internal filename
    internal_filename = f"{str(uuid.uuid4())}{file_extension}"
    saved_file_path = UPLOAD_DIR / internal_filename

    try:
        with open(saved_file_path, "wb") as f:
            f.write(file_content)
    except Exception as e:
        logger.error(f"Failed to save uploaded file {original_filename} to {saved_file_path}: {e}")
        raise HTTPException(status_code=500, detail="Could not save uploaded file.")

    uploader_id_str = None
    if user and hasattr(user, "id"):
        uploader_id_str = str(user.id)

    initial_doc_data = {
        "filename": internal_filename,  # Internal unique name
        "original_filename": original_filename,
        "file_type": file.content_type,  # MIME type
        "file_extension": file_extension,
        "file_size": file_size,
        "uploader_id": uploader_id_str,
        "upload_time": datetime.now().isoformat(),
        "status": "uploaded",  # Initial status
        "text": None,
        "word_count": None,
        "analysis": None,
        "text_preview": None,
        "error_message": None,
    }

    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Database not configured")

        result = supabase.table("documents").insert(initial_doc_data).execute()
        document_id = result.data[0]["id"] if result.data else None

        if not document_id:
            raise Exception("Failed to get document ID from insert")

        logger.info(
            f"File '{original_filename}' (ID: {document_id}) metadata stored in Supabase. Path: {saved_file_path}"
        )
    except Exception as e:
        logger.error(f"Failed to insert document metadata into Supabase for {original_filename}: {e}")
        # Potentially clean up the saved file if DB insert fails
        if saved_file_path.exists():
            saved_file_path.unlink()
        raise HTTPException(status_code=500, detail="Failed to store document metadata.")

    # Add the processing task to background
    # background_tasks.add_task(
    #     process_document_pipeline,
    #     document_id=document_id,
    #     internal_filename=internal_filename,
    #     original_filename=original_filename,
    #     file_content_type=file.content_type, # This should be file_extension for our text_processor
    #     saved_file_path=str(saved_file_path)
    # )
    background_tasks.add_task(
        process_document_pipeline,
        document_id=document_id,
        internal_filename=internal_filename,
        original_filename=original_filename,
        file_extension=file_extension,  # Corrected to pass file_extension
        saved_file_path=str(saved_file_path),
    )
    logger.info(f"Background task added for document ID {document_id} to process file {original_filename}")

    return {
        "message": "File uploaded successfully. Processing started in background.",
        "document_id": document_id,
        "original_filename": original_filename,
        "internal_filename": internal_filename,
    }


@app.get("/api/agent/generate-report/{document_id}")
async def generate_document_report(document_id: str):
    """
    Generates a JSON report for a processed document.
    """
    logger.info(f"Report generation request for document_id: {document_id}")

    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")

    result = supabase.table("documents").select("*").eq("id", document_id).execute()
    doc = result.data[0] if result.data else None

    if not doc:
        logger.warning(f"Report generation: Document not found for ID {document_id}")
        raise HTTPException(status_code=404, detail="Document not found.")

    # Check status - report should only be generated if analysis is complete
    if doc.get("status") != "analyzed":
        logger.warning(
            f"Report generation: Document {document_id} not yet analyzed. Current status: {doc.get('status')}"
        )
        raise HTTPException(
            status_code=422,  # Unprocessable Entity or 409 Conflict could also work
            detail=f"Document processing not complete. Current status: {doc.get('status', 'Unknown')}. Please try again later.",
        )

    # Construct the report from the document fields
    report = {
        "document_id": document_id,  # Good to include the ID in the report
        "original_filename": doc.get("original_filename"),
        "upload_time": doc.get("upload_time"),
        "word_count": doc.get("word_count"),
        "analysis": doc.get("analysis"),
        "text_preview": doc.get("text_preview"),
        # Ensure all these fields are actually populated by the pipeline
    }

    # Filter out None values from report if any field wasn't populated as expected
    # Though ideally, 'analyzed' status means they should be.
    report_cleaned = {k: v for k, v in report.items() if v is not None}

    """Generates a JSON report (summary & insights) for a processed document."""
    # This endpoint is being refactored. The new approach will be separate
    # /summary and /insights endpoints. This one can be deprecated or
    # modified to just fetch existing stored insights.
    # For now, let's make it fetch stored insights.
    logger.info(f"Fetching insights report for document_id: {document_id}, user_id: {user.id}")

    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")

    doc_insight = supabase.table("document_insights").select("document_id, summary, insights_data, generated_at").eq("document_id", document_id).eq("user_id", str(user.id)).order("generated_at", desc=True).limit(1).maybe_single().execute()

    if not doc_insight.data:
        logger.warning(f"No insights found for document ID {document_id} by user {user.id}")
        raise HTTPException(status_code=404, detail="No insights found for this document.")

    return doc_insight.data


# Document Management Endpoints (Refactored from /api/files)
@app.get("/api/documents", response_model=List[Dict[str, Any]])
async def list_documents(user=Depends(get_current_user)):
    """Lists all documents for the authenticated user from the 'documents' table."""
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Database not configured")
        
        user_id_str = str(user.id)
        # Fetch from 'documents' table, not 'uploaded_files'
        result = supabase.table("documents").select(
            "id, original_filename, file_extension, file_size, status, upload_time, text_preview, uploader_id"
        ).eq("uploader_id", user_id_str).order("upload_time", desc=True).execute()

        return result.data if result.data else []

    except Exception as e:
        logger.error(f"Document listing error for user {user.id}: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.get("/api/documents/{document_id}/details")
async def get_document_details(document_id: str, user=Depends(get_current_user)):
    """Gets detailed information about a specific document, including its analysis if available."""
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Database not configured")
        
        user_id_str = str(user.id)
        # Fetch from 'documents' table
        doc_result = supabase.table("documents").select(
            "*" # Select all columns from documents
        ).eq("id", document_id).eq("uploader_id", user_id_str).maybe_single().execute()

        if not doc_result.data:
            raise HTTPException(status_code=404, detail="Document not found or not owned by user.")

        document_details = doc_result.data

        # Fetch latest insights from 'document_insights'
        insights_result = supabase.table("document_insights").select(
            "summary, insights_data, generated_at"
        ).eq("document_id", document_id).order("generated_at", desc=True).limit(1).maybe_single().execute()
        
        if insights_result.data:
            document_details["latest_summary"] = insights_result.data.get("summary")
            document_details["latest_insights"] = insights_result.data.get("insights_data")
            document_details["insights_generated_at"] = insights_result.data.get("generated_at")
        else:
            document_details["latest_summary"] = None
            document_details["latest_insights"] = None
            document_details["insights_generated_at"] = None

        return document_details

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document details error for doc_id {document_id}, user {user.id}: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get document details: {str(e)}")

@app.post("/api/documents/{document_id}/summary", response_model=Dict[str, Any])
async def generate_document_summary_endpoint(document_id: str, user=Depends(get_current_user)):
    """Generates and returns a summary for a document, stores it."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    user_id_str = str(user.id)

    doc_res = supabase.table("documents").select("text, status, uploader_id").eq("id", document_id).eq("uploader_id", user_id_str).maybe_single().execute()
    if not doc_res.data:
        raise HTTPException(status_code=404, detail="Document not found or not owned by user.")

    doc_text = doc_res.data.get("text")
    if not doc_text: # If text is missing, even if status is 'analyzed' (which implies text should be there)
        logger.error(f"Document {document_id} text is missing for summary generation. Status: {doc_res.data.get('status')}")
        raise HTTPException(status_code=422, detail="Document has no text content to summarize. It might not have been processed correctly.")
        
    summary = await agent_logic.generate_document_summary_with_gemini(doc_text, user_id=user_id_str)

    # Store/Update the summary in document_insights
    try:
        existing_insight_res = supabase.table("document_insights").select("id").eq("document_id", document_id).eq("user_id", user_id_str).maybe_single().execute()
        
        current_utc_time = datetime.now(timezone.utc).isoformat()
        insight_payload = {
            "summary": summary,
            "updated_at": current_utc_time
        }
        
        if existing_insight_res.data:
            insight_id = existing_insight_res.data["id"]
            update_op = supabase.table("document_insights").update(insight_payload).eq("id", insight_id).execute()
            if update_op.error: logger.error(f"Error updating summary for doc {document_id}: {update_op.error.message}")
            else: logger.info(f"Updated summary for document {document_id} (insight ID: {insight_id})")
        else:
            insight_payload["id"] = str(uuid.uuid4())
            insight_payload["document_id"] = document_id
            insight_payload["user_id"] = user_id_str
            insight_payload["generated_at"] = current_utc_time
            # insights_data can be null if only summary is generated
            insight_payload["insights_data"] = {} # Or some default if schema requires it
            insert_op = supabase.table("document_insights").insert(insight_payload).execute()
            if insert_op.error: logger.error(f"Error inserting summary for doc {document_id}: {insert_op.error.message}")
            else: logger.info(f"Stored new summary for document {document_id}")

    except Exception as db_e:
        logger.error(f"DB error storing summary for doc {document_id}: {db_e}\n{traceback.format_exc()}")
        # Continue to return summary even if DB store fails for now

    return {"document_id": document_id, "summary": summary}

@app.post("/api/documents/{document_id}/insights", response_model=Dict[str, Any])
async def generate_document_insights_endpoint(document_id: str, user=Depends(get_current_user)):
    """Generates and returns insights for a document, stores them."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    user_id_str = str(user.id)

    doc_res = supabase.table("documents").select("text, status, uploader_id").eq("id", document_id).eq("uploader_id", user_id_str).maybe_single().execute()
    if not doc_res.data:
        raise HTTPException(status_code=404, detail="Document not found or not owned by user.")

    doc_text = doc_res.data.get("text")
    if not doc_text:
        logger.error(f"Document {document_id} text is missing for insights generation. Status: {doc_res.data.get('status')}")
        raise HTTPException(status_code=422, detail="Document has no text content for insights. It might not have been processed correctly.")

    insights = await agent_logic.generate_document_insights_with_gemini(doc_text, user_id=user_id_str)

    # Store/Update insights in document_insights
    try:
        existing_insight_res = supabase.table("document_insights").select("id, summary").eq("document_id", document_id).eq("user_id", user_id_str).maybe_single().execute()
        current_utc_time = datetime.now(timezone.utc).isoformat()
        
        insight_payload = {
            "insights_data": insights,
            "updated_at": current_utc_time
        }

        if existing_insight_res.data:
            insight_id = existing_insight_res.data["id"]
            # Preserve existing summary if this endpoint only generates insights
            if existing_insight_res.data.get("summary"):
                 insight_payload["summary"] = existing_insight_res.data.get("summary")

            update_op = supabase.table("document_insights").update(insight_payload).eq("id", insight_id).execute()
            if update_op.error: logger.error(f"Error updating insights for doc {document_id}: {update_op.error.message}")
            else: logger.info(f"Updated insights for document {document_id} (insight ID: {insight_id})")
        else:
            insight_payload["id"] = str(uuid.uuid4())
            insight_payload["document_id"] = document_id
            insight_payload["user_id"] = user_id_str
            insight_payload["generated_at"] = current_utc_time
            # If only insights are generated, summary might be null or a default
            insight_payload["summary"] = "Summary not generated via this operation."
            insert_op = supabase.table("document_insights").insert(insight_payload).execute()
            if insert_op.error: logger.error(f"Error inserting insights for doc {document_id}: {insert_op.error.message}")
            else: logger.info(f"Stored new insights for document {document_id}")

    except Exception as db_e:
        logger.error(f"DB error storing insights for doc {document_id}: {db_e}\n{traceback.format_exc()}")

    return {"document_id": document_id, "insights": insights}


# The old /api/files/{file_id}/analyze endpoint is now effectively replaced by
# /api/documents/{document_id}/insights and /summary.
# If specific 'analysis_type' logic from the old endpoint is still needed beyond general summary/insights,
# it would require further refactoring or a new dedicated endpoint.
# For now, we assume the new summary/insights cover the primary AI analysis needs for documents.

# Data Sources Management
@app.get("/api/data-sources")
async def get_data_sources(user=Depends(get_current_user)):
    """Get all data sources for the user"""
    try:
        if not supabase:
            return []

        result = supabase.table("data_sources").select("*").eq("user_id", user.id).execute()
        return result.data

    except Exception as e:
        logger.error(f"Data sources fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data sources: {str(e)}")


@app.post("/api/data-sources")
async def create_data_source(data_source: DataSource, user=Depends(get_current_user)):
    """Create a new data source"""
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Database not configured")

        data_source_record = {
            "user_id": user.id,
            "name": data_source.name,
            "type": data_source.type,
            "description": data_source.description,
            "category": data_source.category,
            "config": data_source.config,
            "status": data_source.status,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        result = supabase.table("data_sources").insert(data_source_record).execute()
        return result.data[0]

    except Exception as e:
        logger.error(f"Data source creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create data source: {str(e)}")


@app.put("/api/data-sources/{source_id}")
async def update_data_source(source_id: str, data_source: DataSource, user=Depends(get_current_user)):
    """Update an existing data source"""
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Database not configured")

        update_data = {
            "name": data_source.name,
            "type": data_source.type,
            "description": data_source.description,
            "category": data_source.category,
            "config": data_source.config,
            "status": data_source.status,
            "updated_at": datetime.now().isoformat(),
        }

        result = supabase.table("data_sources").update(update_data).eq("id", source_id).eq("user_id", user.id).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Data source not found")

        return result.data[0]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data source update error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update data source: {str(e)}")


@app.delete("/api/data-sources/{source_id}")
async def delete_data_source(source_id: str, user=Depends(get_current_user)):
    """Delete a data source"""
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Database not configured")

        result = supabase.table("data_sources").delete().eq("id", source_id).eq("user_id", user.id).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Data source not found")

        return {"message": "Data source deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data source deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete data source: {str(e)}")


@app.post("/api/data-sources/{source_id}/test")
async def test_data_source(source_id: str, user=Depends(get_current_user)):
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Database not configured")
            
        result = supabase.table("data_sources").select("*").eq("id", source_id).eq("user_id", user.id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Data source not found")
            
        data_source = result.data[0]
        
        # Mock connection test - in production, implement actual API testing
        # --- START REAL IMPLEMENTATION ---
        from .data_source_tester import test_data_source_connection # Relative import

        source_name = data_source.get("name", "Unknown Source")
        source_type = data_source.get("type", "unknown")
        source_config = data_source.get("config", {})

        logger.info(f"Testing connection for source ID: {source_id}, Name: {source_name}, Type: {source_type}")

        # Update status to 'testing' before starting
        supabase.table("data_sources").update({
            "status": "testing",
            "updated_at": datetime.now(dt_timezone.utc).isoformat()
        }).eq("id", source_id).execute()

        is_successful, message, response_time_ms_float = test_data_source_connection(source_type, source_config)
        
        response_time_ms_int = int(response_time_ms_float) if response_time_ms_float is not None else None


        current_timestamp = datetime.now(dt_timezone.utc).isoformat()
        db_update_payload = {
            "status": "active" if is_successful else "error",
            "last_tested_at": current_timestamp,
            "last_test_error": None if is_successful else message[:1000], # Limit error message length
            "updated_at": current_timestamp
        }

        update_result = supabase.table("data_sources").update(db_update_payload).eq("id", source_id).execute()
        if not update_result.data: # Or check for errors in update_result if client provides
            logger.error(f"Failed to update data source status after test for ID: {source_id}")
            # Decide if this should fail the request or just log

        api_response = {
            "test_successful": is_successful,
            "tested_service_type": source_type,
            "message": message,
            "response_time_ms": response_time_ms_int,
            "timestamp": current_timestamp
        }
        # --- END REAL IMPLEMENTATION ---

        return api_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data source test error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data source test failed: {str(e)}")


@app.post("/api/data-sources/{source_id}/sync")
async def sync_data_source(source_id: str, user=Depends(get_current_user)):
    """Sync data from a data source"""
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Database not configured")

        result = supabase.table("data_sources").select("*").eq("id", source_id).eq("user_id", user.id).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Data source not found")
            
        data_source = result.data[0]

        # --- START REAL SYNC IMPLEMENTATION ---
        from .data_source_sync_handler import sync_data_source, SyncedArticle
        
        user_id_str = str(user.id) # Ensure user_id is a string for DB operations
        source_name = data_source.get("name", "Unknown Source")

        logger.info(f"Starting sync for data source ID: {source_id}, Name: {source_name}, User: {user_id_str}")

        # Update status to 'syncing'
        supabase.table("data_sources").update({
            "status": "syncing",
            "updated_at": datetime.now(dt_timezone.utc).isoformat()
        }).eq("id", source_id).execute()

        synced_articles: List[SyncedArticle] = sync_data_source(data_source)
        
        documents_to_insert = []
        if synced_articles:
            for article in synced_articles:
                # Convert SyncedArticle to a dictionary suitable for 'documents' table
                # The user_id for the document should be the user who owns the data_source
                doc_record = article.to_document_db_record(user_id=user_id_str, source_id=source_id)
                documents_to_insert.append(doc_record)

            if documents_to_insert:
                try:
                    # Batch insert into 'documents' table
                    insert_res = supabase.table("documents").insert(documents_to_insert).execute()
                    if insert_res.data:
                        logger.info(f"Successfully inserted {len(insert_res.data)} documents from source '{source_name}'.")
                        # Optionally, trigger background processing for these new documents here
                        # For example, by calling process_document_pipeline for each new doc_id
                        # for doc_db_entry in insert_res.data:
                        #    background_tasks.add_task(process_document_pipeline, ...) # Needs more params
                    else:
                        logger.error(f"Failed to insert documents from source '{source_name}'. Error: {insert_res.error}")
                        # Handle partial failure or log appropriately
                except Exception as db_e:
                    logger.error(f"Database error inserting synced documents for source '{source_name}': {db_e}")
                    # Update source status to error if DB insert fails catastrophically
                    supabase.table("data_sources").update({
                        "status": "error",
                        "last_test_error": f"DB insert failed: {str(db_e)[:200]}",
                        "updated_at": datetime.now(dt_timezone.utc).isoformat()
                    }).eq("id", source_id).execute()
                    raise HTTPException(status_code=500, detail=f"Failed to store synced documents: {str(db_e)}")

        # Update data source status and last_sync time
        final_status = "active" # Assume active if sync ran, even if 0 articles fetched (could be no new content)
        if not synced_articles and documents_to_insert == []: # If handler ran but returned nothing, could be an issue or just no new data
             pass # Keep status as active, or maybe a specific "synced_empty" if needed

        current_timestamp = datetime.now(dt_timezone.utc).isoformat()
        supabase.table("data_sources").update({
            "status": final_status,
            "last_sync": current_timestamp,
            "updated_at": current_timestamp
        }).eq("id", source_id).execute()

        sync_summary = {
            "sync_successful": True, # True if the process ran, success of fetching depends on articles count
            "source_name": source_name,
            "articles_fetched": len(synced_articles),
            "documents_created": len(documents_to_insert),
            "message": f"Sync completed for '{source_name}'. Fetched {len(synced_articles)} items, created {len(documents_to_insert)} new documents.",
            "timestamp": current_timestamp
        }
        # --- END REAL SYNC IMPLEMENTATION ---
        return sync_summary
        
    except HTTPException:
        # If HTTPException was raised by our code, re-raise it
        # Also ensure source status is updated to 'error' if not already handled
        supabase.table("data_sources").update({
            "status": "error",
            "updated_at": datetime.now(dt_timezone.utc).isoformat()
        }).eq("id", source_id).execute()
        raise
    except Exception as e:
        logger.error(f"Data source sync error for source ID {source_id}: {str(e)}\n{traceback.format_exc()}")
        supabase.table("data_sources").update({
            "status": "error",
            "last_test_error": f"Sync failed: {str(e)[:200]}", # Use last_test_error to store sync error too
            "updated_at": datetime.now(dt_timezone.utc).isoformat()
        }).eq("id", source_id).execute()
        raise HTTPException(status_code=500, detail=f"Data source sync failed: {str(e)}")


@app.get("/api/documents/search")
async def search_documents(
    query: str, 
    user=Depends(get_current_user),
    limit: int = 10,
    offset: int = 0
):
    """Search documents by content"""
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Database not configured")
        
        # Search in document text and analysis
        result = supabase.table("documents").select("*").eq("uploader_id", user.id).execute()
        
        matching_docs = []
        query_lower = query.lower()
        
        for doc in result.data:
            # Search in text content
            text_content = doc.get("text", "").lower()
            analysis = doc.get("analysis", {})
            
            # Simple text matching - in production, use full-text search
            if (query_lower in text_content or 
                query_lower in doc.get("original_filename", "").lower() or
                any(query_lower in str(v).lower() for v in analysis.values() if isinstance(v, (str, dict)))):
                
                matching_docs.append({
                    "document_id": doc["id"],
                    "filename": doc["original_filename"],
                    "file_type": doc["file_extension"],
                    "upload_time": doc["upload_time"],
                    "word_count": doc.get("word_count"),
                    "text_preview": doc.get("text_preview"),
                    "analysis_summary": analysis.get("keyword_analysis", {}).get("analysis_summary"),
                    "relevance_score": text_content.count(query_lower)  # Simple relevance
                })
        
        # Sort by relevance
        matching_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Apply pagination
        paginated_docs = matching_docs[offset:offset + limit]
        
        return {
            "query": query,
            "total_matches": len(matching_docs),
            "documents": paginated_docs,
            "page_info": {
                "limit": limit,
                "offset": offset,
                "has_more": len(matching_docs) > offset + limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error updating user profile for {user_id_str}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to update profile: {str(e)}")


# User Preferences Endpoints
@app.get("/api/users/me/preferences", response_model=UserPreferences)
async def get_user_preferences(user=Depends(get_current_user)):
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    user_id_str = str(user.id)
    try:
        result = supabase.table("user_preferences").select("*").eq("user_id", user_id_str).maybe_single().execute()
        if result.data:
            return result.data
        return UserPreferences()  # Return default empty preferences if not found
    except Exception as e:
        logger.error(f"Error fetching preferences for user {user_id_str}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to fetch user preferences")


@app.put("/api/users/me/preferences", response_model=UserPreferences)
async def update_user_preferences(preferences: UserPreferences, user=Depends(get_current_user)):
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    user_id_str = str(user.id)

    update_data = preferences.model_dump(exclude_unset=True)  # Get only fields that were set
    update_data["user_id"] = user_id_str  # Ensure user_id is set for upsert
    update_data["updated_at"] = datetime.now().isoformat()

    if not update_data:
        raise HTTPException(status_code=400, detail="No preference data provided")

    try:
        # Upsert ensures that if a record for the user doesn't exist, it's created.
        # If it exists, it's updated.
        result = supabase.table("user_preferences").upsert(update_data).execute()
        if result.data:
            return result.data[0]
        # Supabase upsert with returning="minimal" (default) might not return data on conflict/update.
        # Fetch after update if necessary, or rely on client to assume success.
        # For now, let's fetch to be sure.
        fetch_result = supabase.table("user_preferences").select("*").eq("user_id", user_id_str).single().execute()
        return fetch_result.data

    except Exception as e:
        logger.error(f"Error updating preferences for user {user_id_str}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to update user preferences")

# User Profile Endpoints (Added for Phase 2.4.1)
@app.get("/api/users/me/profile", response_model=Dict[str, Any])
async def get_user_profile(user=Depends(get_current_user)):
    if not user: # get_current_user should raise HTTPException if no valid user
        logger.error("get_user_profile: Unlikely scenario - user is None after Depends(get_current_user).")
        raise HTTPException(status_code=401, detail="Not authenticated or user data unavailable.")

    # Safely access user_metadata and its properties
    user_metadata = user.user_metadata if hasattr(user, 'user_metadata') else {}

    profile_data = {
        "id": str(user.id),
        "email": user.email,
        "full_name": user_metadata.get("full_name"),
        "avatar_url": user_metadata.get("avatar_url"),
        "created_at": user.created_at.isoformat() if hasattr(user, 'created_at') and user.created_at else None,
        "last_sign_in_at": user.last_sign_in_at.isoformat() if hasattr(user, 'last_sign_in_at') and user.last_sign_in_at else None,
        "onboarding_complete": user_metadata.get("onboarding_complete", False), # Default to False if not set
        "dashboard_widgets": user_metadata.get("dashboard_widgets", []) # Default to empty list
    }
    return profile_data

@app.put("/api/users/me/profile", response_model=Dict[str, Any])
async def update_user_profile(profile_update: UserProfileUpdateRequest, user=Depends(get_current_user)):
    if not supabase:
        logger.error("update_user_profile: Supabase client not available.")
        raise HTTPException(status_code=500, detail="Service misconfiguration: Auth service unavailable.")
    if not user: # Should be caught by Depends(get_current_user)
        raise HTTPException(status_code=401, detail="Not authenticated.")

    user_id_str = str(user.id)
    metadata_to_update = {} # Only include fields that are actually being changed

    if profile_update.full_name is not None:
        metadata_to_update["full_name"] = profile_update.full_name
    if profile_update.avatar_url is not None:
        metadata_to_update["avatar_url"] = profile_update.avatar_url

    if not metadata_to_update:
        # It's better to return the current profile than an error if no actual changes are requested.
        # Or, a 304 Not Modified, but that's more complex.
        # For now, let's return current profile data fetched freshly.
        logger.info(f"update_user_profile called for user {user_id_str} with no actual data fields to update.")
        # Re-fetch to ensure we return the latest state, though user object from Depends should be fresh.
        return await get_user_profile(user)


    try:
        # Merge with existing metadata to only update provided fields
        # Ensure user.user_metadata is not None (it can be if never set)
        existing_metadata = user.user_metadata if user.user_metadata is not None else {}
        new_metadata_payload = {**existing_metadata, **metadata_to_update}

        # Use supabase.auth.update_user for the currently authenticated user
        update_response = await supabase.auth.update_user(
            {"data": new_metadata_payload} # 'data' is the key for user_metadata updates
        )

        if update_response.user:
            logger.info(f"User profile updated successfully for user_id: {user_id_str}")
            refreshed_user = update_response.user
            # Construct response from the refreshed_user object
            return {
                "id": str(refreshed_user.id),
                "email": refreshed_user.email,
                "full_name": refreshed_user.user_metadata.get("full_name"),
                "avatar_url": refreshed_user.user_metadata.get("avatar_url"),
                "onboarding_complete": refreshed_user.user_metadata.get("onboarding_complete", False),
                "dashboard_widgets": refreshed_user.user_metadata.get("dashboard_widgets", [])
            }
        elif update_response.error:
            error_detail = f"Failed to update profile: {update_response.error.message}"
            logger.error(f"Supabase auth.update_user error for {user_id_str}: {update_response.error.message}")
            raise HTTPException(status_code=400, detail=error_detail) # 400 for bad request often from Supabase
        else:
            logger.error(f"User profile update for {user_id_str} had no user object and no error in response. This is unexpected.")
            raise HTTPException(status_code=500, detail="Failed to update profile due to an unexpected auth server response.")

    except HTTPException: # Re-raise HTTPExceptions from Supabase client or our own logic
        raise
    except Exception as e:
        logger.error(f"Exception during user profile update for {user_id_str}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to update profile due to an internal error: {str(e)}")


# Application Settings Endpoints
@app.get("/api/app-settings", response_model=List[AppSettingItem])
async def get_app_settings(user=Depends(get_current_user)):  # Added auth for now, can be removed if settings are public
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    try:
        result = supabase.table("app_settings").select("setting_key, setting_value, description").execute()
        return result.data if result.data else []
    except Exception as e:
        logger.error(f"Error fetching app settings: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to fetch application settings")


@app.put("/api/app-settings", response_model=List[AppSettingItem])
async def update_app_settings(
    app_settings_update: AppSettingsUpdateRequest, user=Depends(get_current_user)  # Require auth for updates
):
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")

    updated_settings = []
    # For simplicity, this endpoint will update existing keys.
    # A more robust version might handle creation or require admin roles.
    try:
        for setting_item in app_settings_update.settings:
            update_payload = {
                "setting_value": setting_item.setting_value,
                "description": setting_item.description,
                "updated_at": datetime.now().isoformat(),
            }
            # Upsert based on setting_key
            result = (
                supabase.table("app_settings")
                .upsert(
                    {
                        "setting_key": setting_item.setting_key,
                        "setting_value": setting_item.setting_value,
                        "description": setting_item.description,
                        "updated_at": datetime.now().isoformat(),
                    },
                    on_conflict="setting_key",
                )
                .execute()
            )  # type: ignore

            if result.data:
                updated_settings.append(result.data[0])
            else:  # If upsert doesn't return data on conflict, fetch it
                fetch_res = (
                    supabase.table("app_settings")
                    .select("setting_key, setting_value, description")
                    .eq("setting_key", setting_item.setting_key)
                    .single()
                    .execute()
                )
                if fetch_res.data:
                    updated_settings.append(fetch_res.data)

        return updated_settings

    except Exception as e:
        logger.error(f"Document insights extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Insight extraction failed: {str(e)}")


        
        # Mock sync process - in production, implement actual data syncing
        sync_result = {
            "sync_successful": True,
            "records_synced": 150,
            "message": f"Successfully synced data from {data_source['name']}",
            "timestamp": datetime.now().isoformat()
        }
        
        # Update last sync timestamp
        supabase.table("data_sources").update({
            "last_sync": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }).eq("id", source_id).execute()
        
        return sync_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data source sync error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data source sync failed: {str(e)}")

@app.get("/api/kpi")
async def get_kpi(timeframe: str = "latest", category: str = "all", user=Depends(get_current_user) ): # Added user dependency
    logger.info(f"KPI request: timeframe={timeframe}, category={category}, user_id={user.id if user else 'None'}")
    if not supabase:
        logger.error("KPI GET: Supabase client not available.")
        raise HTTPException(status_code=500, detail="Database connection not available.")

    try:
        # Fetch the latest global cumulative KPIs recorded by the cron job
        # For "total_analyses_run_cumulative"
        analyses_res = supabase.table("kpi_metrics") \
            .select("metric_value, metric_unit, period_end, details") \
            .eq("metric_name", "total_analyses_run_cumulative") \
            .is_("user_id", None) \
            .order("period_end", desc=True) \
            .limit(1) \
            .maybe_single() \
            .execute()

        # For "total_documents_processed_cumulative"
        docs_res = supabase.table("kpi_metrics") \
            .select("metric_value, metric_unit, period_end, details") \
            .eq("metric_name", "total_documents_processed_cumulative") \
            .is_("user_id", None) \
            .order("period_end", desc=True) \
            .limit(1) \
            .maybe_single() \
            .execute()

        # Placeholder for average trends and opportunities.
        # This would ideally query `kpi_metrics` if the cron job was populating these averages,
        # or calculate them on-the-fly here if it had access to all agent states.
        # For now, we'll return placeholders or default values for these.

        num_analyses = analyses_res.data["metric_value"] if analyses_res.data and analyses_res.data.get("metric_value") is not None else 0

        # Fetch total trends and opportunities (if cron stores them, or calculate if possible)
        # This is a simplified example; a real system might need more complex aggregation.
        # Let's assume the cron job *could* store these as cumulative totals as well.

        # Fetch all required latest global KPI metrics in one go if possible, or separate calls
        metric_names_to_fetch = [
            "total_analyses_run_cumulative",
            "total_documents_processed_cumulative",
            "cumulative_total_trends",
            "cumulative_total_opportunities",
            "analyses_count_for_trend_opportunity_avg" # Fetched to use as denominator for averages
        ]

        fetched_kpis = {}
        latest_date_recorded = None

        for name in metric_names_to_fetch:
            res = supabase.table("kpi_metrics") \
                .select("metric_value, metric_unit, period_end, details") \
                .eq("metric_name", name) \
                .is_("user_id", None) \
                .order("period_end", desc=True) \
                .limit(1) \
                .maybe_single() \
                .execute()
            if res.data:
                fetched_kpis[name] = res.data
                if latest_date_recorded is None or datetime.fromisoformat(res.data["period_end"]) > datetime.fromisoformat(latest_date_recorded):
                    latest_date_recorded = res.data["period_end"]
            else:
                fetched_kpis[name] = None # Metric not found or no data

        num_analyses = fetched_kpis.get("total_analyses_run_cumulative", {}).get("metric_value", 0)
        num_docs_processed = fetched_kpis.get("total_documents_processed_cumulative", {}).get("metric_value", 0)

        cumulative_trends = fetched_kpis.get("cumulative_total_trends", {}).get("metric_value", 0)
        cumulative_ops = fetched_kpis.get("cumulative_total_opportunities", {}).get("metric_value", 0)
        # Use analyses_count_for_trend_opportunity_avg as the denominator for averages
        # This count represents analyses that actually had trend/opportunity data processed by the cron.
        # If this specific metric isn't populated by cron, fallback to num_analyses (total completed analyses).
        denominator_for_avg = fetched_kpis.get("analyses_count_for_trend_opportunity_avg", {}).get("metric_value", num_analyses)
        if denominator_for_avg == 0 : # Avoid division by zero if no analyses were processed for these averages
            denominator_for_avg = num_analyses # Fallback to total analyses, if still 0, avg will be 0.


        avg_trends_per_analysis = (cumulative_trends / denominator_for_avg) if denominator_for_avg > 0 else 0
        avg_ops_per_analysis = (cumulative_ops / denominator_for_avg) if denominator_for_avg > 0 else 0

        # Format date_recorded to YYYY-MM-DD
        date_recorded_str = datetime.fromisoformat(latest_date_recorded).strftime('%Y-%m-%d') if latest_date_recorded else datetime.now(dt_timezone.utc).strftime('%Y-%m-%d')

        kpi_response_data = {
            "total_analyses_run": num_analyses,
            "documents_processed": num_docs_processed,
            "avg_trends_identified": round(avg_trends_per_analysis, 1),
            "avg_opportunities_identified": round(avg_ops_per_analysis, 1),
            "date_recorded": date_recorded_str
        }

        # The frontend dashboard page expects a list of kpi objects in response.data.
        # The new structure is a single object in response.data. This needs frontend adaptation.
        # For now, I will adapt the backend to return the old list structure but with new data.
        # This is temporary to avoid breaking the frontend immediately.
        # TODO: Coordinate with frontend to accept the new simpler object structure for KPIs.

        kpi_data_transformed_for_frontend_list = [
            {
                "metric_name": "Total Analyses Run",
                "metric_value": kpi_response_data["total_analyses_run"],
                "metric_unit": fetched_kpis.get("total_analyses_run_cumulative", {}).get("metric_unit", "count"),
                "change_percentage": 0, # Placeholder
            },
            {
                "metric_name": "Total Documents Processed",
                "metric_value": kpi_response_data["documents_processed"],
                "metric_unit": fetched_kpis.get("total_documents_processed_cumulative", {}).get("metric_unit", "count"),
                "change_percentage": 0, # Placeholder
            },
            {
                "metric_name": "Avg. Trends per Analysis",
                "metric_value": kpi_response_data["avg_trends_identified"],
                "metric_unit": "trends/analysis",
                "change_percentage": 0, # Placeholder
            },
            {
                "metric_name": "Avg. Opportunities per Analysis",
                "metric_value": kpi_response_data["avg_opportunities_identified"],
                "metric_unit": "ops/analysis",
                "change_percentage": 0, # Placeholder
            }
        ]

        return {
            "data": kpi_data_transformed_for_frontend_list, # Returning list for now
            # "data_object": kpi_response_data, # This is the target structure
            "metadata": {
                "timeframe": timeframe,
                "category": category,
                "last_updated": datetime.now(dt_timezone.utc).isoformat(),
                "date_recorded_for_kpis": date_recorded_str
            }
        }

    except Exception as e:
        logger.error(f"KPI GET error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"KPI fetch failed: {str(e)}")


@app.post("/api/kpi", dependencies=[Depends(get_current_user)]) # Added auth
async def store_kpi(request: KPIRequest, user=Depends(get_current_user)):
    logger.info(f"Storing KPI: Metric={request.metric}, Value={request.value}, UserID={user.id}")
    if not supabase:
        logger.error("KPI POST: Supabase client not available.")
        raise HTTPException(status_code=500, detail="Database connection not available.")

    try:
        current_time = datetime.now(dt_timezone.utc)
        # Assuming request.timestamp is the period_end for this KPI record
        period_end_ts = datetime.fromisoformat(request.timestamp) if request.timestamp else current_time

        # For manually stored KPIs, period_start might be the same as period_end or not directly relevant
        # unless the request provides it.
        # We'll make period_start nullable or set it same as period_end if not provided.

        kpi_record = {
            "user_id": user.id, # Associate with the user making the request
            "metric_name": request.metric,
            "metric_value": request.value,
            # "metric_unit": provided by request or inferred? For now, assume not part of basic KPIRequest
            "period_end": period_end_ts.isoformat(),
            "recorded_at": current_time.isoformat(),
            # "details": {} # Optional: if request included more details
        }

        # Upsert logic: if a KPI with the same name for this user for this exact period_end timestamp exists, update it.
        # Otherwise, insert. The unique constraints on (metric_name, period_end, user_id) will handle this.
        # The Supabase Python client's .upsert() by default works on the primary key `id`.
        # To achieve semantic upsert on our defined keys, we'd typically:
        # 1. Try to select an existing record based on metric_name, user_id, and period_end.
        # 2. If exists, update it.
        # 3. If not, insert a new one.
        # OR rely on database's ON CONFLICT UPDATE if table is set up for it and client supports it easily.
        # For simplicity here, we'll insert. If a duplicate for (metric_name, period_end, user_id) is attempted,
        # the DB unique constraint `idx_kpi_metrics_name_period_user` should raise an error.
        # A true "upsert" on these columns would require more complex logic or ensuring `id` is known and stable.

        # Let's try a simple insert and let the DB handle conflicts if an exact match is inserted again.
        # Or, if this endpoint is meant to *always* create a new historical entry, then insert is fine.
        # If it's to update the "latest" for a user for a metric, different logic is needed.
        # Given the schema, it seems designed for multiple records over time.

        response = supabase.table("kpi_metrics").insert(kpi_record).execute()

        if response.data:
            logger.info(f"KPI POST: Successfully stored KPI '{request.metric}' for user {user.id}.")
            return response.data[0] # Return the created record
        else:
            logger.error(f"KPI POST: Failed to store KPI. Error: {response.error}")
            raise HTTPException(status_code=500, detail=f"Failed to store KPI: {response.error.message if response.error else 'Unknown error'}")

    except Exception as e:
        logger.error(f"KPI POST storage error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"KPI storage failed: {str(e)}")


@app.post("/api/agent/sync")
async def agent_sync(request: AgentSyncRequest):
    try:
        logger.info(f"Agent sync: action={request.action}")

        # Enhanced agent sync with more actions
        sync_result = {
            "success": True,
            "action": request.action,
            "data": request.data,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "sync_id": f"sync_{hash(request.action)}_{int(datetime.now().timestamp())}",
        }

        return sync_result
    except Exception as e:
        logger.error(f"Agent sync error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent sync failed: {str(e)}")


@app.get("/api/agent/status")
async def agent_status():
    return {
        "status": "online",
        "version": "1.0.0",
        "uptime": "running",
        "capabilities": [
            "market_analysis",
            "chat_interface",
            "kpi_tracking",
            "data_sync",
            "file_processing",
            "rag_search",
            "ai_insights",
        ],
        "ai_models": ["google-gemini"],
        "file_types_supported": [".csv", ".xlsx", ".pdf", ".txt"],
        "timestamp": datetime.now().isoformat(),
    }


# Report Generation Endpoints
@app.post("/api/reports/generate")
async def generate_report(report_type: str = "comprehensive", format: str = "json", user=Depends(get_current_user)):
    """Generate various types of reports"""
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Database not configured")

        # Get user's data for report generation
        analyses = supabase.table("market_analyses").select("*").eq("user_id", user.id).limit(10).execute()
        files = supabase.table("uploaded_files").select("*").eq("user_id", user.id).limit(10).execute()

        report_data = {
            "report_id": str(uuid.uuid4()),
            "report_type": report_type,
            "user_id": user.id,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_analyses": len(analyses.data),
                "total_files": len(files.data),
                "most_recent_analysis": analyses.data[0]["created_at"] if analyses.data else None,
            },
            "analyses": analyses.data[:5],  # Latest 5 analyses
            "file_summary": [
                {"filename": f["filename"], "type": f["file_type"], "uploaded": f["uploaded_at"]}
                for f in files.data[:5]
            ],
        }

        if format == "csv":
            # For CSV format, return structured data
            return {
                "report_id": report_data["report_id"],
                "download_url": f"/api/reports/{report_data['report_id']}/download",
                "format": "csv",
                "generated_at": report_data["generated_at"],
            }

        return report_data

    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get("/api/reports/{report_id}/download")
async def download_report(report_id: str, format: str = "json"):
    """Download a generated report"""
    try:
        # In production, retrieve actual report data
        mock_report = {
            "report_id": report_id,
            "title": "Market Intelligence Report",
            "generated_at": datetime.now().isoformat(),
            "sections": [
                {
                    "title": "Executive Summary",
                    "content": "Market analysis shows positive trends in digital transformation sectors."
                },
                {
                    "title": "Key Insights",
                    "content": "3 major opportunities identified in emerging markets."
                }
            ]
        }
        
        if format == "csv":
            import io

            output = io.StringIO()
            output.write("Section,Content\n")
            for section in mock_report["sections"]:
                output.write(f'"{section["title"]}","{section["content"]}"\n')
            
            return JSONResponse(
                content={"csv_data": output.getvalue()},
                headers={"Content-Disposition": f"attachment; filename=report_{report_id}.csv"},
            )
        
        return mock_report
        
    except Exception as e:
        logger.error(f"Report download error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report download failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"

    logger.info(f"Starting Enhanced Market Intelligence Agent API on {host}:{port}")
    logger.info("Features: File Upload, RAG Chat, AI Analysis, Data Integration")
    uvicorn.run(app, host=host, port=port, log_level="info")


# --- KPI Update Cron Endpoint ---
# Import calculator and other necessary things
from api.kpi_calculator import (
    get_total_analyses_run,
    get_total_documents_processed,
    extract_kpis_from_analysis_state
    # We'll need a way to get all relevant agent states to calculate avg trends/opportunities
)
from fastapi import Header, Depends
from datetime import datetime, timedelta, timezone as dt_timezone # Renamed to avoid conflict

CRON_SECRET_ENV_VAR = os.getenv("CRON_SECRET")

async def verify_cron_secret(x_cron_secret: Optional[str] = Header(None)):
    if not CRON_SECRET_ENV_VAR:
        logger.error("CRON_SECRET is not set in environment variables. KPI update endpoint is effectively disabled.")
        raise HTTPException(status_code=500, detail="Endpoint misconfiguration.")
    if not x_cron_secret or x_cron_secret != CRON_SECRET_ENV_VAR:
        logger.warning("Unauthorized attempt to access KPI update endpoint.")
        raise HTTPException(status_code=403, detail="Forbidden.")
    return True

@app.post("/api/internal/cron/update-global-kpis", dependencies=[Depends(verify_cron_secret)])
async def cron_update_global_kpis():
    """
    Scheduled endpoint to calculate and update global KPIs.
    Secured by X-Cron-Secret header.
    """
    logger.info("CRON: Starting global KPI update process.")
    if not supabase:
        logger.error("CRON: Supabase client not available. Cannot update KPIs.")
        raise HTTPException(status_code=500, detail="Database connection not available.")

    current_time = datetime.now(dt_timezone.utc)
    period_end_ts = current_time
    # For daily KPIs, period_start could be the beginning of the current day
    period_start_ts = current_time.replace(hour=0, minute=0, second=0, microsecond=0)

    kpi_updates = []
    errors = []

    # 1. Total Analyses Run (Cumulative)
    total_analyses = get_total_analyses_run()
    if total_analyses is not None:
        kpi_updates.append({
            "metric_name": "total_analyses_run_cumulative",
            "metric_value": total_analyses,
            "metric_unit": "count",
            "period_start": None, # Cumulative, so no specific start for this record
            "period_end": period_end_ts.isoformat(),
            "recorded_at": current_time.isoformat(),
            "user_id": None, # Global KPI
            "details": {"description": "Cumulative total of all successfully completed analyses."}
        })
    else:
        errors.append("Failed to retrieve total_analyses_run.")

    # 2. Total Documents Processed (Cumulative)
    total_docs = get_total_documents_processed()
    if total_docs is not None:
        kpi_updates.append({
            "metric_name": "total_documents_processed_cumulative",
            "metric_value": total_docs,
            "metric_unit": "count",
            "period_start": None, # Cumulative
            "period_end": period_end_ts.isoformat(),
            "recorded_at": current_time.isoformat(),
            "user_id": None, # Global KPI
            "details": {"description": "Cumulative total of all successfully analyzed documents."}
        })
    else:
        errors.append("Failed to retrieve total_documents_processed.")

    # For Average Trends/Opportunities:
    # This requires iterating through all agent states or a summary table.
    # For simplicity in this step, we'll placeholder this.
    # A more robust solution would query the agent's SQLite DB for all states,
    # extract num_trends/num_opportunities, sum them up, and also count total states with these metrics.
    # Then calculate overall averages.

    # Placeholder for cumulative total trends and opportunities identified
    # In a real scenario, you'd query agent_logic's DB or a summary table.
    # For now, let's imagine we have these totals.
    # This part needs a robust way to get all agent states from agent_logic.py's DB.
    # Let's assume for now these are not implemented by this cron directly to avoid complexity
    # of accessing SQLite from here, and will be handled by a different mechanism or
    # the /api/kpi GET endpoint will calculate averages on the fly.

    # Upserting into Supabase
    if kpi_updates:
        try:
            # Note: Supabase Python client upsert needs a list of dicts.
            # `on_conflict` should match the unique index. For global KPIs, it's (metric_name, period_end) where user_id IS NULL.
            # The python client might need a specific way to define this, or we ensure period_end is unique enough for global.
            # A simpler unique key for global daily KPIs: (metric_name, DATE(period_end))
            # For this example, we assume `period_end` (timestamp) + `metric_name` is sufficiently unique for global daily snapshot.
            # If user_id is part of the upsert and is None, it should match the global unique index.

            # We need to ensure the `on_conflict` strategy works with nullable user_id.
            # One way is to have separate upserts or ensure the unique index handles NULLs as distinct if that's the DB behavior,
            # or always include user_id and for global ones, use a fixed placeholder UUID if the DB doesn't like NULL in unique constraints with other non-nulls.
            # Given our schema with two conditional unique indexes, we should be fine with user_id=None.

            upsert_result = supabase.table("kpi_metrics").upsert(kpi_updates).execute() # default on_conflict is primary key `id`
                                                                                       # We need conflict on metric_name, period_end, user_id

            # To correctly use on_conflict with Supabase upsert for our unique constraints:
            # The table needs a primary key (id). The upsert can use `on_conflict` with other columns.
            # However, the Python client's upsert has a default behavior. If `id` is provided and exists, it updates.
            # If `id` is not provided, it inserts.
            # To achieve true upsert on our semantic key (metric_name, period_end, user_id),
            # we might need to query first then update/insert, or rely on DB-level ON CONFLICT rules if the Python client doesn't expose it directly.
            # For now, let's assume we are inserting new records for each `period_end` for simplicity of this example,
            # and `id` is auto-generated. If we want to update the *same record* for a given day, more complex logic is needed client-side
            # or a stored procedure.
            # A common pattern for upsert without direct on_conflict on specific columns in client:
            # Try to update where metric_name, period_end (date part), user_id match. If rows affected = 0, then insert.
            # Or, if we are okay with multiple records per day for a metric and take the latest, then just insert.
            # For now, let's just insert. The unique constraint will prevent duplicates for the *exact same timestamp* `period_end`.
            # For true daily upsert, `period_end` should be truncated to the day.

            # Corrected approach for daily upsert (assuming period_end is just the date for daily KPIs):
            # We will store `period_end` as the specific timestamp of calculation for cumulative values.
            # For "average trends today", period_start and period_end would be start/end of day.

            # For this iteration, let's just insert. The GET /api/kpi will fetch the latest for a given metric_name.
            insert_result = supabase.table("kpi_metrics").insert(kpi_updates).execute()

            if insert_result.data:
                logger.info(f"CRON: Successfully inserted/updated {len(insert_result.data)} KPI records.")
            else: # Handle potential errors from Supabase
                logger.error(f"CRON: KPI upsert failed or returned no data. Error: {insert_result.error}")
                errors.append(f"KPI upsert failed: {insert_result.error.message if insert_result.error else 'Unknown error'}")

        except Exception as e:
            logger.error(f"CRON: Error during Supabase KPI upsert: {e}")
            errors.append(f"Supabase KPI upsert exception: {str(e)}")

    if errors:
        logger.error(f"CRON: KPI update process completed with errors: {errors}")
        # Depending on severity, could raise HTTPException or just log
        return {"status": "completed_with_errors", "updated_kpis": len(kpi_updates), "errors": errors}

    logger.info("CRON: Global KPI update process completed successfully.")
    return {"status": "success", "updated_kpis": len(kpi_updates), "message": "Global KPIs updated."}

# Make sure to import kpi_calculator at the top of main.py
# from api import kpi_calculator # If it's structured like this
# from . import kpi_calculator # Or relative if main.py is a module
# For now, assuming it's available as `kpi_calculator.func_name`

# Also, ensure CRON_SECRET is set in your .env file for this to work.
# Example: CRON_SECRET="your_strong_random_secret_here"

# --- Dashboard Chart Data Endpoints ---
from api.agent_logic import load_state as load_agent_state # Renamed to avoid conflict with any FastAPI state

@app.get("/api/trends")
async def get_trends_data(analysis_id: Optional[str] = None, user=Depends(get_current_user)):
    logger.info(f"GET /api/trends: analysis_id={analysis_id}, user_id={user.id}")
    if not supabase: # Although load_agent_state uses SQLite, supabase client check is good for consistency
        raise HTTPException(status_code=500, detail="Service not fully configured.")

    state_to_load = None
    if analysis_id:
        # TODO: Add check to ensure user owns this analysis_id if loading directly by ID
        # For now, assuming direct load is admin/specific or analysis_id is validated elsewhere
        loaded_s = load_agent_state(analysis_id)
        if loaded_s and loaded_s.user_id == str(user.id): # Check ownership
            state_to_load = loaded_s
        elif loaded_s:
            logger.warning(f"User {user.id} attempted to load state {analysis_id} owned by {loaded_s.user_id}")
            raise HTTPException(status_code=403, detail="Access denied to this analysis.")
        else:
            raise HTTPException(status_code=404, detail=f"Analysis with ID '{analysis_id}' not found.")
    else:
        # Fetch the most recent completed analysis for the current user from agent's SQLite DB
        # This requires a new function in agent_logic.py to get most recent state_id for user
        # For now, let's return a placeholder if no analysis_id is given,
        # as implementing "most recent" query on SQLite from here is complex.
        # Or, use the Supabase 'reports' table if it stores state_id and completion status.

        # Let's assume for now the frontend *should* provide an analysis_id for specific charts.
        # If not, we return a sample or error.
        # For the current dashboard, it doesn't pass an ID. So we need a default.
        # Let's try to get the latest from Supabase 'reports' table, assuming it has 'state_id' populated.

        # This part is complex as 'reports' table might not have state_id.
        # The agent's SQLite DB is the primary source of states.
        # We need a robust way to list states for a user from agent_logic.py
        # For now, if no analysis_id, return default/mock data or an error.
        # The dashboard's current call pattern (no ID) for /trends needs to be addressed.
        # A simpler default: Use a globally pre-defined "sample" analysis ID if one exists for demos.

        # Fallback: if agent_logic.py's init_db() creates sample data, use that.
        # Or, we can query the agent's SQLite DB for the latest state for the user.
        # This requires a new function in agent_logic.py: get_latest_user_state_id(user_id: str)
        from api.agent_logic import get_latest_user_state_id # Assuming this function will be created
        latest_state_id = get_latest_user_state_id(str(user.id))
        if latest_state_id:
            state_to_load = load_agent_state(latest_state_id)
            if not (state_to_load and state_to_load.user_id == str(user.id)): # Double check ownership
                state_to_load = None # Should not happen if get_latest_user_state_id is correct

        if not state_to_load:
            logger.warning(f"No analysis ID provided and no suitable recent analysis found for user {user.id} for /trends.")
            # Return sample data to match frontend expectation for now
            return {"data": [
                {"name": "Sample Trend 1", "impact": 3},
                {"name": "Sample Trend 2", "impact": 2},
                {"name": "Sample Trend 3", "impact": 1}
            ]}


    if not state_to_load or not state_to_load.market_trends:
        logger.warning(f"No market trends found in loaded state for analysis_id={analysis_id or 'latest'} for user {user.id}.")
        return {"data": []} # Return empty list if no trends

    def impact_to_value(impact_str: Optional[str]) -> int:
        if not impact_str: return 0
        val = str(impact_str).lower()
        if val == 'high': return 3
        if val == 'medium': return 2
        if val == 'low': return 1
        return 0

    transformed_trends = [
        {
            "name": trend.get("trend_name", "Unknown Trend"),
            "impact": impact_to_value(trend.get("estimated_impact"))
        }
        for trend in state_to_load.market_trends
    ]

    return {"data": transformed_trends}

# TODO: Create get_latest_user_state_id(user_id: str) in agent_logic.py
# This function would query the agent's SQLite 'states' table:
# SELECT id FROM states WHERE user_id = ? ORDER BY created_at DESC LIMIT 1;
# (Need to ensure created_at is stored and indexed properly for sorting)

@app.get("/api/competitors")
async def get_competitors_data(analysis_id: Optional[str] = None, user=Depends(get_current_user)):
    logger.info(f"GET /api/competitors: analysis_id={analysis_id}, user_id={user.id}")
    # Basic structure similar to /api/trends
    state_to_load = None
    if analysis_id:
        loaded_s = load_agent_state(analysis_id)
        if loaded_s and loaded_s.user_id == str(user.id):
            state_to_load = loaded_s
        elif loaded_s:
            raise HTTPException(status_code=403, detail="Access denied.")
        else:
            raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found.")
    else:
        from api.agent_logic import get_latest_user_state_id
        latest_state_id = get_latest_user_state_id(str(user.id))
        if latest_state_id:
            state_to_load = load_agent_state(latest_state_id)
            if not (state_to_load and state_to_load.user_id == str(user.id)):
                state_to_load = None

        if not state_to_load:
            logger.warning(f"No analysis ID provided and no suitable recent analysis found for user {user.id} for /competitors. Returning sample data.")
            # This sample data should match the structure expected by app/competitors/page.tsx's transform function
            return {"data": [
                {"company_name": "Mock Comp A", "market_share": "25.5", "activity_score": "75", "growth_rate": "5.2", "revenue": "$100M", "employees": "500", "strengths": ["Innovation"], "weaknesses": ["High Price"], "recentActivity": "Launched new product X.", "threat_level": "high"},
                {"company_name": "Mock Comp B", "market_share": "15.0", "activity_score": "60", "growth_rate": "-2.1", "revenue": "$50M", "employees": "200", "strengths": ["Price"], "weaknesses": ["Marketing"], "recentActivity": "Acquired smaller company Y.", "threat_level": "medium"},
            ]}

    if not state_to_load or not state_to_load.raw_news_data: # Using raw_news_data as placeholder
        logger.warning(f"No competitor data (raw_news_data) found in loaded state for analysis_id={analysis_id or 'latest'} for user {user.id}.")
        return {"data": []}

    # --- MOCK TRANSFORMATION of raw_news_data to competitor-like structure ---
    # This is highly artificial and needs replacement with a real competitor analysis module.
    logger.warning("Serving MOCK/DERIVED competitor data from raw_news_data for /api/competitors. Needs proper competitor analysis module.")

    mock_competitors = []
    for i, item in enumerate(state_to_load.raw_news_data[:5]): # Take first 5 articles as mock competitors
        source_name = item.get("source", f"Source {i+1}")
        # Try to parse a domain from source if it's a URL, otherwise use source name
        try:
            parsed_url = urlparse(source_name)
            display_name = parsed_url.netloc if parsed_url.netloc else source_name
        except:
            display_name = source_name

        mock_competitors.append({
            "id": item.get("url", f"comp-{i}"), # Use URL as ID or generate one
            "company_name": item.get("title", display_name)[:50], # Use article title or source
            "name": item.get("title", display_name)[:50], # For compatibility with some frontend chart expectations
            "title": item.get("title", display_name)[:50], # ibid
            "market_share": str(10 + (hash(display_name) % 15)),  # Dummy market share btw 10-24%
            "revenue_string": f"${50 + (hash(display_name) % 100)}M", # Dummy revenue
            "revenue": f"${50 + (hash(display_name) % 100)}M", # For compatibility
            "employees": str(100 + (hash(display_name) % 900)), # Dummy employees
            "growth_rate": str(round(-5 + (hash(display_name) % 100) / 10, 1)), # Dummy growth rate -5 to 4.9%
            "activity_score": str(50 + (hash(display_name) % 50)), # Dummy activity score
            "activity": str(50 + (hash(display_name) % 50)), # For compatibility
            "growth": str(round(-5 + (hash(display_name) % 100) / 10, 1)), # For compatibility
            "strengths": ["Strong Brand", "Innovation"] if i % 2 == 0 else ["Large Customer Base"],
            "weaknesses": ["High Prices"] if i % 2 == 0 else ["Slow Adaptability"],
            "recentActivity": item.get("summary", "Recent news summary placeholder.")[:150],
            "summary": item.get("summary", "Recent news summary placeholder.")[:150], # For compatibility
            "threat_level": ["low", "medium", "high"][i % 3],
        })

    return {"data": mock_competitors}


@app.get("/api/customer-insights")
async def get_customer_insights_data(analysis_id: Optional[str] = None, user=Depends(get_current_user)):
    logger.info(f"GET /api/customer-insights: analysis_id={analysis_id}, user_id={user.id}")
    state_to_load = None
    if analysis_id:
        loaded_s = load_agent_state(analysis_id)
        if loaded_s and loaded_s.user_id == str(user.id):
            state_to_load = loaded_s
        elif loaded_s:
            raise HTTPException(status_code=403, detail="Access denied.")
        else:
            raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found.")
    else:
        from api.agent_logic import get_latest_user_state_id
        latest_state_id = get_latest_user_state_id(str(user.id))
        if latest_state_id:
            state_to_load = load_agent_state(latest_state_id)
            if not (state_to_load and state_to_load.user_id == str(user.id)):
                state_to_load = None

        if not state_to_load:
            logger.warning(f"No analysis ID provided and no suitable recent analysis found for user {user.id} for /customer-insights. Returning sample data.")
            # Sample data structure based on app/customer-insights/page.tsx
            return {"data": [
                {"segment_name": "Sample Segment Alpha", "description": "Description for Alpha.", "percentage": 60, "key_characteristics": ["Char1", "Char2"], "pain_points": ["Pain1", "Pain2"], "growth_potential": "High", "satisfaction_score": 8.5, "retention_rate": 75, "acquisition_cost": "Medium", "lifetime_value": "High"},
                {"segment_name": "Sample Segment Beta", "description": "Description for Beta.", "percentage": 40, "key_characteristics": ["Char3", "Char4"], "pain_points": ["Pain3", "Pain4"], "growth_potential": "Medium", "satisfaction_score": 7.0, "retention_rate": 60, "acquisition_cost": "High", "lifetime_value": "Medium"},
            ]}

    if not state_to_load or not state_to_load.customer_insights:
        logger.warning(f"No customer insights found in loaded state for analysis_id={analysis_id or 'latest'} for user {user.id}.")
        return {"data": []}

    # The customer_insights in agent state should already be in the correct format
    # as generated by customer_insights_generator node.
    return {"data": state_to_load.customer_insights}

# Need to import urlparse for the mock competitor data generation
from urllib.parse import urlparse
from fastapi.responses import FileResponse # For serving files

# Agent State related endpoints (for downloads page)
@app.get("/api/analysis-states/{state_id}/downloads-info") # Renamed slightly for clarity from just /downloads
async def get_analysis_state_downloads_info(state_id: str, user=Depends(get_current_user)):
    logger.info(f"GET /api/analysis-states/{state_id}/downloads-info: UserID='{user.id}'")
    if not state_id:
        raise HTTPException(status_code=400, detail="State ID is required.")

    try:
        # Assuming get_state_download_info from agent_logic handles user_id check internally for ownership
        # or returns None if state_id doesn't belong to user or doesn't exist.
        download_info = agent_logic.get_state_download_info(state_id=state_id, user_id=str(user.id))
        if not download_info:
            # This could be 404 if state not found, or 403 if not owned.
            # get_state_download_info logs details, so a generic 404 is fine here.
            raise HTTPException(status_code=404, detail="Analysis state not found or not accessible.")

        return download_info # Structure should be like {"state_id": ..., "files": [{"category": ..., "filename": ...}]}
    except HTTPException:
        raise # Re-raise known HTTP exceptions
    except Exception as e:
        logger.error(f"Error getting download info for state {state_id}, user {user.id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to retrieve download information.")

@app.get("/api/analysis-states/{state_id}/download-file/{file_identifier}")
async def download_analysis_file(state_id: str, file_identifier: str, user=Depends(get_current_user)):
    logger.info(f"GET /api/analysis-states/{state_id}/download-file/{file_identifier}: UserID='{user.id}'")
    if not state_id or not file_identifier:
        raise HTTPException(status_code=400, detail="State ID and file identifier are required.")

    try:
        # get_download_file_path in agent_logic should handle ownership check and path validation
        file_path_str = agent_logic.get_download_file_path(state_id=state_id, user_id=str(user.id), file_identifier=file_identifier)

        if not file_path_str:
            raise HTTPException(status_code=404, detail="File not found or not accessible.")

        # Check if file exists, just in case get_download_file_path didn't fully validate existence (it should)
        if not os.path.exists(file_path_str) or not os.path.isfile(file_path_str):
            logger.error(f"File path resolved but file does not exist or is not a file: {file_path_str}")
            raise HTTPException(status_code=404, detail="File not found on server.")

        # Use original filename for download if available, otherwise use file_identifier
        # This might require get_download_file_path to return more than just the path,
        # or we derive it from file_identifier if it's the actual filename.
        # For now, using file_identifier as the download name.
        # A better approach: get_state_download_info could provide the original filename for each identifier.
        # The `file_identifier` from the frontend IS the `filename` from `DownloadableFile` interface.

        return FileResponse(path=file_path_str, filename=file_identifier, media_type='application/octet-stream')

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file {file_identifier} for state {state_id}, user {user.id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to download file.")
