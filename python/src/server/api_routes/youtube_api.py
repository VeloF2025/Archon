"""
YouTube API Routes for Archon Knowledge Engine

Provides endpoints for:
- Processing YouTube videos
- Extracting transcripts
- Searching videos
- Adding videos to knowledge base
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field, HttpUrl

from ..services.youtube_service import get_youtube_service
# TODO: Fix knowledge_service import - service doesn't exist yet
# from ..services.knowledge_service import get_knowledge_service

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/youtube", tags=["youtube"])


class YouTubeVideoRequest(BaseModel):
    """Request model for processing YouTube video"""
    url: HttpUrl = Field(..., description="YouTube video URL")
    add_to_knowledge_base: bool = Field(False, description="Add to knowledge base after processing")
    download_video: bool = Field(False, description="Download video file")
    audio_only: bool = Field(False, description="Download audio only")
    quality: str = Field("720p", description="Video quality for download")


class YouTubeSearchRequest(BaseModel):
    """Request model for searching YouTube videos"""
    query: str = Field(..., description="Search query")
    max_results: int = Field(10, ge=1, le=50, description="Maximum results to return")


class YouTubeChannelRequest(BaseModel):
    """Request model for getting channel videos"""
    channel_id: str = Field(..., description="YouTube channel ID")
    max_results: int = Field(50, ge=1, le=100, description="Maximum videos to retrieve")


class YouTubeTranscriptRequest(BaseModel):
    """Request model for extracting transcript"""
    url: HttpUrl = Field(..., description="YouTube video URL")
    languages: List[str] = Field(["en"], description="Preferred transcript languages")


@router.get("/health")
async def get_youtube_health():
    """Health check for YouTube service"""
    try:
        youtube_service = get_youtube_service()
        return {
            "status": "healthy",
            "api_configured": youtube_service.api_key is not None,
            "service": "youtube"
        }
    except Exception as e:
        logger.error(f"YouTube health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.post("/process")
async def process_youtube_video(
    request: YouTubeVideoRequest,
    background_tasks: BackgroundTasks
):
    """
    Process a YouTube video and optionally add to knowledge base
    """
    try:
        youtube_service = get_youtube_service()
        
        # Extract video ID
        video_id = youtube_service.extract_video_id(str(request.url))
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        
        # Process video for knowledge base
        processed_data = await youtube_service.process_video_for_knowledge_base(str(request.url))
        
        if "error" in processed_data:
            raise HTTPException(status_code=400, detail=processed_data["error"])
        
        # Optionally download video
        download_path = None
        if request.download_video:
            download_path = await youtube_service.download_video(
                video_id, 
                quality=request.quality,
                audio_only=request.audio_only
            )
            processed_data["download_path"] = download_path
        
        # Optionally add to knowledge base
        if request.add_to_knowledge_base:
            # TODO: Fix knowledge_service integration
            # knowledge_service = get_knowledge_service()
            
            # # Create knowledge base entry
            # source_data = {
            #     "url": str(request.url),
            #     "title": processed_data["title"],
            #     "description": processed_data["description"],
            #     "content": processed_data.get("content", ""),
            #     "metadata": {
            #         "video_id": video_id,
            #         "channel": processed_data["channel"],
            #         "channel_id": processed_data["channel_id"],
            #         "published_at": processed_data["published_at"],
            #         "duration": processed_data["duration"],
            #         "transcript": processed_data.get("transcript"),
            #         **processed_data.get("metadata", {})
            #     }
            # }
            
            # # Add to knowledge base in background
            # background_tasks.add_task(
            #     knowledge_service.add_youtube_to_knowledge_base,
            #     source_data
            # )
            
            processed_data["added_to_knowledge_base"] = False  # Temporarily disabled
        
        return {
            "success": True,
            "video_id": video_id,
            "data": processed_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing YouTube video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcript")
async def extract_transcript(request: YouTubeTranscriptRequest):
    """
    Extract transcript from a YouTube video
    """
    try:
        youtube_service = get_youtube_service()
        
        # Extract video ID
        video_id = youtube_service.extract_video_id(str(request.url))
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        
        # Get transcript
        transcript = await youtube_service.get_transcript(video_id, request.languages)
        
        if not transcript:
            raise HTTPException(
                status_code=404, 
                detail="No transcript available for this video"
            )
        
        return {
            "success": True,
            "video_id": video_id,
            "transcript": transcript
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_youtube_videos(request: YouTubeSearchRequest):
    """
    Search for YouTube videos
    """
    try:
        youtube_service = get_youtube_service()
        
        if not youtube_service.api_key:
            raise HTTPException(
                status_code=400,
                detail="YouTube API key not configured. Search requires API key."
            )
        
        # Search videos
        videos = await youtube_service.search_videos(request.query, request.max_results)
        
        return {
            "success": True,
            "query": request.query,
            "results": videos,
            "count": len(videos)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching YouTube: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/channel/videos")
async def get_channel_videos(request: YouTubeChannelRequest):
    """
    Get videos from a YouTube channel
    """
    try:
        youtube_service = get_youtube_service()
        
        if not youtube_service.api_key:
            raise HTTPException(
                status_code=400,
                detail="YouTube API key not configured. Channel videos requires API key."
            )
        
        # Get channel videos
        videos = await youtube_service.get_channel_videos(
            request.channel_id, 
            request.max_results
        )
        
        return {
            "success": True,
            "channel_id": request.channel_id,
            "videos": videos,
            "count": len(videos)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting channel videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/video/{video_id}")
async def get_video_metadata(video_id: str):
    """
    Get metadata for a specific YouTube video
    """
    try:
        youtube_service = get_youtube_service()
        
        # Get video metadata
        metadata = await youtube_service.get_video_metadata(video_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail="Video not found")
        
        return {
            "success": True,
            "video_id": video_id,
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/download/{video_id}")
async def download_youtube_video(
    video_id: str,
    quality: str = Query("720p", description="Video quality"),
    audio_only: bool = Query(False, description="Download audio only")
):
    """
    Download a YouTube video or audio
    """
    try:
        youtube_service = get_youtube_service()
        
        # Download video
        download_path = await youtube_service.download_video(
            video_id,
            quality=quality,
            audio_only=audio_only
        )
        
        if not download_path:
            raise HTTPException(status_code=500, detail="Download failed")
        
        return {
            "success": True,
            "video_id": video_id,
            "download_path": download_path,
            "audio_only": audio_only,
            "quality": quality
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch/process")
async def batch_process_videos(
    urls: List[HttpUrl],
    add_to_knowledge_base: bool = False,
    background_tasks: BackgroundTasks = None
):
    """
    Process multiple YouTube videos in batch
    """
    try:
        youtube_service = get_youtube_service()
        results = []
        
        for url in urls:
            try:
                video_id = youtube_service.extract_video_id(str(url))
                if not video_id:
                    results.append({
                        "url": str(url),
                        "success": False,
                        "error": "Invalid YouTube URL"
                    })
                    continue
                
                # Process video
                processed_data = await youtube_service.process_video_for_knowledge_base(str(url))
                
                if "error" in processed_data:
                    results.append({
                        "url": str(url),
                        "success": False,
                        "error": processed_data["error"]
                    })
                else:
                    results.append({
                        "url": str(url),
                        "success": True,
                        "video_id": video_id,
                        "title": processed_data["title"],
                        "has_transcript": processed_data.get("transcript") is not None
                    })
                    
                    # Add to knowledge base if requested
                    if add_to_knowledge_base and background_tasks:
                        # TODO: Fix knowledge_service integration
                        # knowledge_service = get_knowledge_service()
                        # background_tasks.add_task(
                        #     knowledge_service.add_youtube_to_knowledge_base,
                        #     processed_data
                        # )
                        pass  # Temporarily disabled
                        
            except Exception as e:
                results.append({
                    "url": str(url),
                    "success": False,
                    "error": str(e)
                })
        
        successful = sum(1 for r in results if r["success"])
        
        return {
            "success": True,
            "total": len(urls),
            "successful": successful,
            "failed": len(urls) - successful,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Export router for main app
__all__ = ["router"]