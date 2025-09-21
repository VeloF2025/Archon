"""
YouTube Service for Archon Knowledge Engine

Provides functionality for:
- Fetching video metadata
- Extracting transcripts
- Downloading videos
- Processing YouTube content for knowledge base
"""

import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import yt_dlp

logger = logging.getLogger(__name__)


class YouTubeService:
    """Service for interacting with YouTube API and processing video content"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize YouTube service
        
        Args:
            api_key: YouTube Data API v3 key (optional, can use env var)
        """
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        self.youtube = None
        
        if self.api_key:
            try:
                self.youtube = build("youtube", "v3", developerKey=self.api_key)
                logger.info("YouTube API client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize YouTube API client: {e}")
        else:
            logger.info("YouTube API key not provided - using limited functionality")
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract video ID from YouTube URL
        
        Args:
            url: YouTube video URL
            
        Returns:
            Video ID or None if invalid URL
        """
        import re
        
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    async def get_video_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch video metadata from YouTube API
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Video metadata dictionary or None if error
        """
        if not self.youtube:
            logger.warning("YouTube API client not initialized")
            return await self._get_metadata_with_ytdlp(video_id)
        
        try:
            request = self.youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=video_id
            )
            response = request.execute()
            
            if response["items"]:
                item = response["items"][0]
                return {
                    "id": video_id,
                    "title": item["snippet"]["title"],
                    "description": item["snippet"]["description"],
                    "channel": item["snippet"]["channelTitle"],
                    "channel_id": item["snippet"]["channelId"],
                    "published_at": item["snippet"]["publishedAt"],
                    "duration": item["contentDetails"]["duration"],
                    "view_count": int(item["statistics"].get("viewCount", 0)),
                    "like_count": int(item["statistics"].get("likeCount", 0)),
                    "comment_count": int(item["statistics"].get("commentCount", 0)),
                    "tags": item["snippet"].get("tags", []),
                    "thumbnail": item["snippet"]["thumbnails"]["high"]["url"]
                }
            
            return None
            
        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            return await self._get_metadata_with_ytdlp(video_id)
        except Exception as e:
            logger.error(f"Error fetching video metadata: {e}")
            return None
    
    async def _get_metadata_with_ytdlp(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Fallback method to get metadata using yt-dlp
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Video metadata dictionary or None if error
        """
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://youtube.com/watch?v={video_id}", download=False)
                
                return {
                    "id": video_id,
                    "title": info.get("title"),
                    "description": info.get("description"),
                    "channel": info.get("uploader"),
                    "channel_id": info.get("channel_id"),
                    "published_at": datetime.fromtimestamp(info.get("upload_date", 0)).isoformat() if info.get("upload_date") else None,
                    "duration": info.get("duration"),
                    "view_count": info.get("view_count", 0),
                    "like_count": info.get("like_count", 0),
                    "comment_count": info.get("comment_count", 0),
                    "tags": info.get("tags", []),
                    "thumbnail": info.get("thumbnail")
                }
                
        except Exception as e:
            logger.error(f"Error fetching metadata with yt-dlp: {e}")
            return None
    
    async def get_transcript(self, video_id: str, languages: List[str] = ["en"]) -> Optional[Dict[str, Any]]:
        """
        Extract transcript from YouTube video
        
        Args:
            video_id: YouTube video ID
            languages: Preferred languages for transcript
            
        Returns:
            Transcript data or None if not available
        """
        try:
            # Try to get transcript in preferred languages
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try manual transcripts first
            try:
                transcript = transcript_list.find_manually_created_transcript(languages)
            except:
                # Fall back to auto-generated
                try:
                    transcript = transcript_list.find_generated_transcript(languages)
                except:
                    # Get any available transcript
                    transcript = next(iter(transcript_list))
            
            # Fetch the actual transcript data
            transcript_data = transcript.fetch()
            
            # Combine all text
            full_text = " ".join([entry["text"] for entry in transcript_data])
            
            # Create timestamped segments for reference
            segments = [
                {
                    "start": entry["start"],
                    "duration": entry["duration"],
                    "text": entry["text"]
                }
                for entry in transcript_data
            ]
            
            return {
                "language": transcript.language,
                "language_code": transcript.language_code,
                "is_generated": transcript.is_generated,
                "full_text": full_text,
                "segments": segments,
                "word_count": len(full_text.split())
            }
            
        except TranscriptsDisabled:
            logger.warning(f"Transcripts are disabled for video {video_id}")
            return None
        except NoTranscriptFound:
            logger.warning(f"No transcript found for video {video_id}")
            return None
        except Exception as e:
            logger.error(f"Error extracting transcript: {e}")
            return None
    
    async def download_video(self, video_id: str, output_path: str = "./downloads", 
                           quality: str = "720p", audio_only: bool = False) -> Optional[str]:
        """
        Download YouTube video or audio
        
        Args:
            video_id: YouTube video ID
            output_path: Directory to save the download
            quality: Video quality (e.g., "720p", "1080p", "best")
            audio_only: Download only audio if True
            
        Returns:
            Path to downloaded file or None if error
        """
        try:
            os.makedirs(output_path, exist_ok=True)
            
            if audio_only:
                format_string = "bestaudio/best"
                ext = "mp3"
            else:
                if quality == "best":
                    format_string = "best"
                else:
                    format_string = f"best[height<={quality[:-1]}]"
                ext = "mp4"
            
            output_template = os.path.join(output_path, f"%(title)s.%(ext)s")
            
            ydl_opts = {
                'format': format_string,
                'outtmpl': output_template,
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False
            }
            
            if audio_only:
                ydl_opts['postprocessors'] = [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }]
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://youtube.com/watch?v={video_id}", download=True)
                filename = ydl.prepare_filename(info)
                
                if audio_only:
                    # Replace extension for audio files
                    filename = filename.rsplit(".", 1)[0] + ".mp3"
                
                logger.info(f"Downloaded video {video_id} to {filename}")
                return filename
                
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return None
    
    async def search_videos(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for YouTube videos
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of video metadata dictionaries
        """
        if not self.youtube:
            logger.warning("YouTube API client not initialized - search requires API key")
            return []
        
        try:
            request = self.youtube.search().list(
                q=query,
                part="snippet",
                type="video",
                maxResults=max_results
            )
            response = request.execute()
            
            videos = []
            for item in response.get("items", []):
                videos.append({
                    "id": item["id"]["videoId"],
                    "title": item["snippet"]["title"],
                    "description": item["snippet"]["description"],
                    "channel": item["snippet"]["channelTitle"],
                    "published_at": item["snippet"]["publishedAt"],
                    "thumbnail": item["snippet"]["thumbnails"]["high"]["url"]
                })
            
            return videos
            
        except HttpError as e:
            logger.error(f"YouTube API error during search: {e}")
            return []
        except Exception as e:
            logger.error(f"Error searching videos: {e}")
            return []
    
    async def get_channel_videos(self, channel_id: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Get videos from a YouTube channel
        
        Args:
            channel_id: YouTube channel ID
            max_results: Maximum number of videos to retrieve
            
        Returns:
            List of video metadata dictionaries
        """
        if not self.youtube:
            logger.warning("YouTube API client not initialized - channel videos requires API key")
            return []
        
        try:
            # Get the uploads playlist ID
            channel_request = self.youtube.channels().list(
                id=channel_id,
                part="contentDetails"
            )
            channel_response = channel_request.execute()
            
            if not channel_response["items"]:
                logger.warning(f"Channel {channel_id} not found")
                return []
            
            uploads_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
            
            # Get videos from the uploads playlist
            videos = []
            next_page_token = None
            
            while len(videos) < max_results:
                playlist_request = self.youtube.playlistItems().list(
                    playlistId=uploads_playlist_id,
                    part="snippet",
                    maxResults=min(50, max_results - len(videos)),
                    pageToken=next_page_token
                )
                playlist_response = playlist_request.execute()
                
                for item in playlist_response.get("items", []):
                    videos.append({
                        "id": item["snippet"]["resourceId"]["videoId"],
                        "title": item["snippet"]["title"],
                        "description": item["snippet"]["description"],
                        "published_at": item["snippet"]["publishedAt"],
                        "thumbnail": item["snippet"]["thumbnails"].get("high", {}).get("url")
                    })
                
                next_page_token = playlist_response.get("nextPageToken")
                if not next_page_token:
                    break
            
            return videos[:max_results]
            
        except HttpError as e:
            logger.error(f"YouTube API error getting channel videos: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting channel videos: {e}")
            return []
    
    async def process_video_for_knowledge_base(self, video_url: str) -> Dict[str, Any]:
        """
        Process a YouTube video for inclusion in the knowledge base
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Processed video data ready for knowledge base
        """
        video_id = self.extract_video_id(video_url)
        if not video_id:
            return {"error": "Invalid YouTube URL"}
        
        # Get video metadata
        metadata = await self.get_video_metadata(video_id)
        if not metadata:
            return {"error": "Failed to fetch video metadata"}
        
        # Get transcript
        transcript = await self.get_transcript(video_id)
        
        # Prepare knowledge base entry
        knowledge_entry = {
            "source_type": "youtube_video",
            "source_url": video_url,
            "video_id": video_id,
            "title": metadata["title"],
            "description": metadata["description"],
            "channel": metadata["channel"],
            "channel_id": metadata["channel_id"],
            "published_at": metadata["published_at"],
            "duration": metadata["duration"],
            "metadata": {
                "view_count": metadata["view_count"],
                "like_count": metadata["like_count"],
                "tags": metadata["tags"],
                "thumbnail": metadata["thumbnail"]
            }
        }
        
        if transcript:
            knowledge_entry["content"] = transcript["full_text"]
            knowledge_entry["transcript"] = {
                "language": transcript["language"],
                "is_generated": transcript["is_generated"],
                "word_count": transcript["word_count"],
                "segments": transcript["segments"][:10]  # Store first 10 segments as sample
            }
        else:
            knowledge_entry["content"] = metadata["description"]
            knowledge_entry["transcript"] = None
        
        return knowledge_entry


# Singleton instance
_youtube_service: Optional[YouTubeService] = None


def get_youtube_service() -> YouTubeService:
    """Get or create YouTube service singleton"""
    global _youtube_service
    if _youtube_service is None:
        _youtube_service = YouTubeService()
    return _youtube_service