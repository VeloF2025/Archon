"""
Gemini Multimodal API Routes

This module provides API endpoints for Gemini CLI's multimodal processing capabilities,
including image-to-code, PDF processing, codebase analysis, and more.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ..config.logfire_config import api_logger, safe_set_attribute, safe_span
from ..services.gemini_cli_service import (
    TaskType,
    TaskPriority,
    get_gemini_cli_service
)
from ..services.llm_provider_service import (
    TaskCharacteristics,
    route_llm_task,
    execute_with_gemini_cli
)

router = APIRouter(prefix="/api/gemini", tags=["gemini_multimodal"])


class ImageToCodeRequest(BaseModel):
    """Request for converting images to code"""
    image_type: str = "ui_mockup"  # ui_mockup, architecture_diagram, database_schema
    output_language: str = "typescript"  # typescript, python, react, etc.
    additional_instructions: Optional[str] = None


class PDFToCodeRequest(BaseModel):
    """Request for converting PDFs to code"""
    pdf_type: str = "specification"  # specification, api_docs, requirements
    output_type: str = "implementation"  # implementation, tests, client_library
    additional_context: Optional[str] = None


class CodebaseAnalysisRequest(BaseModel):
    """Request for analyzing a codebase"""
    path: str
    analysis_type: str = "general"  # general, security, performance, architecture
    specific_questions: Optional[List[str]] = None
    include_patterns: Optional[List[str]] = None  # File patterns to include
    exclude_patterns: Optional[List[str]] = None  # File patterns to exclude


class MultimodalProcessRequest(BaseModel):
    """Generic multimodal processing request"""
    prompt: str
    task_type: str = "general"
    priority: str = "normal"


@router.post("/image-to-code")
async def image_to_code(
    file: UploadFile = File(...),
    image_type: str = Form("ui_mockup"),
    output_language: str = Form("typescript"),
    additional_instructions: Optional[str] = Form(None)
):
    """Convert an image (UI mockup, diagram, etc.) to code
    
    This endpoint leverages Gemini CLI's multimodal capabilities to generate
    code from visual inputs like UI sketches, architecture diagrams, or database schemas.
    """
    with safe_span("api_gemini_image_to_code") as span:
        safe_set_attribute(span, "image_type", image_type)
        safe_set_attribute(span, "output_language", output_language)
        safe_set_attribute(span, "file_size", file.size)
        
        try:
            # Validate file type
            allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"]
            if file.content_type not in allowed_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
                )
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                # Build prompt based on image type
                prompts = {
                    "ui_mockup": f"Convert this UI mockup into a {output_language} component with proper styling, state management, and responsive design.",
                    "architecture_diagram": f"Generate {output_language} code structure based on this architecture diagram, including interfaces, classes, and relationships.",
                    "database_schema": f"Create {output_language} models/entities based on this database schema diagram, including relationships and validations."
                }
                
                base_prompt = prompts.get(image_type, "Generate code from this image")
                
                if additional_instructions:
                    base_prompt += f"\n\nAdditional requirements: {additional_instructions}"
                
                # Execute with Gemini CLI
                result = await execute_with_gemini_cli(
                    prompt=base_prompt,
                    files=[tmp_file_path],
                    task_type=TaskType.MULTIMODAL,
                    priority=TaskPriority.HIGH
                )
                
                api_logger.info(f"Image to code conversion completed: {image_type} -> {output_language}")
                safe_set_attribute(span, "success", True)
                
                return {
                    "success": True,
                    "generated_code": result.get("content", ""),
                    "image_type": image_type,
                    "output_language": output_language,
                    "execution_time": result.get("execution_time", 0),
                    "model": result.get("model", "gemini-2.5-pro")
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
        except HTTPException:
            raise
        except Exception as e:
            api_logger.error(f"Image to code conversion failed: {e}")
            safe_set_attribute(span, "error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/pdf-to-code")
async def pdf_to_code(
    file: UploadFile = File(...),
    pdf_type: str = Form("specification"),
    output_type: str = Form("implementation"),
    additional_context: Optional[str] = Form(None)
):
    """Convert a PDF document to code
    
    Process technical PDFs (specs, API docs, requirements) and generate
    corresponding code implementations, tests, or client libraries.
    """
    with safe_span("api_gemini_pdf_to_code") as span:
        safe_set_attribute(span, "pdf_type", pdf_type)
        safe_set_attribute(span, "output_type", output_type)
        safe_set_attribute(span, "file_size", file.size)
        
        try:
            # Validate file type
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail="File must be a PDF document"
                )
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                # Build prompt based on PDF type and output type
                prompt_templates = {
                    ("specification", "implementation"): "Generate a complete implementation based on this specification document. Include all required features, error handling, and documentation.",
                    ("specification", "tests"): "Generate comprehensive test cases based on this specification. Include unit tests, integration tests, and edge cases.",
                    ("api_docs", "client_library"): "Generate a TypeScript client library for this API documentation. Include all endpoints, types, error handling, and usage examples.",
                    ("requirements", "implementation"): "Generate code that fulfills all requirements in this document. Ensure all acceptance criteria are met."
                }
                
                prompt_key = (pdf_type, output_type)
                base_prompt = prompt_templates.get(prompt_key, "Generate code based on this PDF document")
                
                if additional_context:
                    base_prompt += f"\n\nAdditional context: {additional_context}"
                
                # Execute with Gemini CLI
                result = await execute_with_gemini_cli(
                    prompt=base_prompt,
                    files=[tmp_file_path],
                    task_type=TaskType.MULTIMODAL,
                    priority=TaskPriority.HIGH
                )
                
                api_logger.info(f"PDF to code conversion completed: {pdf_type} -> {output_type}")
                safe_set_attribute(span, "success", True)
                
                return {
                    "success": True,
                    "generated_code": result.get("content", ""),
                    "pdf_type": pdf_type,
                    "output_type": output_type,
                    "execution_time": result.get("execution_time", 0),
                    "model": result.get("model", "gemini-2.5-pro")
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
        except HTTPException:
            raise
        except Exception as e:
            api_logger.error(f"PDF to code conversion failed: {e}")
            safe_set_attribute(span, "error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-codebase")
async def analyze_codebase(request: CodebaseAnalysisRequest):
    """Analyze an entire codebase using Gemini's 1M token context
    
    Perform comprehensive analysis of large codebases including architecture review,
    security audits, performance analysis, and more.
    """
    with safe_span("api_gemini_analyze_codebase") as span:
        safe_set_attribute(span, "path", request.path)
        safe_set_attribute(span, "analysis_type", request.analysis_type)
        
        try:
            # Verify path exists
            if not os.path.exists(request.path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Path not found: {request.path}"
                )
            
            # Get Gemini CLI service
            gemini_cli = await get_gemini_cli_service()
            
            # Build analysis prompt
            analysis_prompts = {
                "general": "Provide a comprehensive analysis of this codebase including architecture, patterns, potential improvements, and technical debt.",
                "security": "Perform a security audit of this codebase. Identify vulnerabilities, insecure patterns, and provide remediation recommendations.",
                "performance": "Analyze performance bottlenecks, inefficient algorithms, and optimization opportunities in this codebase.",
                "architecture": "Review the architecture of this codebase. Evaluate design patterns, modularity, scalability, and provide improvement suggestions."
            }
            
            prompt = analysis_prompts.get(request.analysis_type, "Analyze this codebase")
            
            if request.specific_questions:
                prompt += "\n\nSpecifically answer these questions:\n"
                for i, question in enumerate(request.specific_questions, 1):
                    prompt += f"{i}. {question}\n"
            
            # Perform analysis
            result = await gemini_cli.analyze_codebase(
                path=request.path,
                prompt=prompt
            )
            
            api_logger.info(f"Codebase analysis completed: {request.path}")
            safe_set_attribute(span, "success", True)
            
            return {
                "success": True,
                "analysis": result.get("content", ""),
                "analysis_type": request.analysis_type,
                "path": request.path,
                "execution_time": result.get("execution_time", 0),
                "model": result.get("model", "gemini-2.5-pro"),
                "context_window": result.get("context_window", "1M tokens")
            }
            
        except HTTPException:
            raise
        except Exception as e:
            api_logger.error(f"Codebase analysis failed: {e}")
            safe_set_attribute(span, "error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-multimodal")
async def process_multimodal(
    request: MultimodalProcessRequest,
    files: Optional[List[UploadFile]] = File(None)
):
    """Generic endpoint for multimodal processing with Gemini CLI
    
    Process any combination of text prompts and files (images, PDFs, etc.)
    using Gemini's multimodal capabilities.
    """
    with safe_span("api_gemini_process_multimodal") as span:
        safe_set_attribute(span, "task_type", request.task_type)
        safe_set_attribute(span, "priority", request.priority)
        safe_set_attribute(span, "num_files", len(files) if files else 0)
        
        try:
            # Save uploaded files temporarily
            temp_files = []
            
            if files:
                for file in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                        content = await file.read()
                        tmp_file.write(content)
                        temp_files.append(tmp_file.name)
            
            try:
                # Map string types to enums
                task_type_map = {
                    "multimodal": TaskType.MULTIMODAL,
                    "large_context": TaskType.LARGE_CONTEXT,
                    "code_generation": TaskType.CODE_GENERATION,
                    "documentation": TaskType.DOCUMENTATION,
                    "analysis": TaskType.ANALYSIS,
                    "general": TaskType.GENERAL
                }
                
                priority_map = {
                    "high": TaskPriority.HIGH,
                    "normal": TaskPriority.NORMAL,
                    "low": TaskPriority.LOW
                }
                
                # Execute with Gemini CLI
                result = await execute_with_gemini_cli(
                    prompt=request.prompt,
                    files=temp_files,
                    task_type=task_type_map.get(request.task_type, TaskType.GENERAL),
                    priority=priority_map.get(request.priority, TaskPriority.NORMAL)
                )
                
                api_logger.info(f"Multimodal processing completed")
                safe_set_attribute(span, "success", True)
                
                return {
                    "success": True,
                    "result": result.get("content", ""),
                    "task_type": request.task_type,
                    "execution_time": result.get("execution_time", 0),
                    "model": result.get("model", "gemini-2.5-pro"),
                    "status": result.get("status", "completed")
                }
                
            finally:
                # Clean up temporary files
                for tmp_file in temp_files:
                    if os.path.exists(tmp_file):
                        os.unlink(tmp_file)
                        
        except Exception as e:
            api_logger.error(f"Multimodal processing failed: {e}")
            safe_set_attribute(span, "error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/usage-stats")
async def get_usage_stats():
    """Get Gemini CLI usage statistics
    
    Returns current rate limit status, daily allocations, and queue information.
    """
    with safe_span("api_gemini_usage_stats") as span:
        try:
            gemini_cli = await get_gemini_cli_service()
            
            if not gemini_cli._initialized:
                return {
                    "available": False,
                    "message": "Gemini CLI is not initialized"
                }
            
            stats = await gemini_cli.get_usage_stats()
            
            api_logger.info("Retrieved Gemini CLI usage stats")
            safe_set_attribute(span, "success", True)
            
            return {
                "available": True,
                "stats": stats
            }
            
        except Exception as e:
            api_logger.error(f"Failed to get usage stats: {e}")
            safe_set_attribute(span, "error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-queue")
async def process_queue():
    """Process queued Gemini CLI tasks
    
    Manually trigger processing of queued tasks when rate limits allow.
    Returns results of processed tasks.
    """
    with safe_span("api_gemini_process_queue") as span:
        try:
            gemini_cli = await get_gemini_cli_service()
            
            if not gemini_cli._initialized:
                raise HTTPException(
                    status_code=503,
                    detail="Gemini CLI is not initialized"
                )
            
            results = await gemini_cli.process_queue()
            
            api_logger.info(f"Processed {len(results)} queued tasks")
            safe_set_attribute(span, "tasks_processed", len(results))
            safe_set_attribute(span, "success", True)
            
            return {
                "success": True,
                "tasks_processed": len(results),
                "results": results
            }
            
        except HTTPException:
            raise
        except Exception as e:
            api_logger.error(f"Queue processing failed: {e}")
            safe_set_attribute(span, "error", str(e))
            raise HTTPException(status_code=500, detail=str(e))