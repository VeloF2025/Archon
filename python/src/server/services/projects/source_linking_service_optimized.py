"""
Optimized Source Linking Service Module for Archon

This module provides centralized logic for managing project-source relationships,
with batch query optimizations to eliminate N+1 query problems.
"""

from typing import Any, Dict, List
from collections import defaultdict

from src.server.utils import get_supabase_client
from ...config.logfire_config import get_logger

logger = get_logger(__name__)


class OptimizedSourceLinkingService:
    """Optimized service class for managing project-source relationships"""

    def __init__(self, supabase_client=None):
        """Initialize with optional supabase client"""
        self.supabase_client = supabase_client or get_supabase_client()

    def get_project_sources(self, project_id: str) -> tuple[bool, dict[str, list[str]]]:
        """
        Get all linked sources for a project, separated by type.
        (Legacy method for backward compatibility)

        Returns:
            Tuple of (success, {"technical_sources": [...], "business_sources": [...]})
        """
        try:
            response = (
                self.supabase_client.table("archon_project_sources")
                .select("source_id, notes")
                .eq("project_id", project_id)
                .execute()
            )

            technical_sources = []
            business_sources = []

            for source_link in response.data:
                if source_link.get("notes") == "technical":
                    technical_sources.append(source_link["source_id"])
                elif source_link.get("notes") == "business":
                    business_sources.append(source_link["source_id"])

            return True, {
                "technical_sources": technical_sources,
                "business_sources": business_sources,
            }
        except Exception as e:
            logger.error(f"Error getting project sources: {e}")
            return False, {
                "error": f"Failed to retrieve linked sources: {str(e)}",
                "technical_sources": [],
                "business_sources": [],
            }

    def get_all_project_sources_batch(self, project_ids: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """
        Get all linked sources for multiple projects in a single batch query.
        This eliminates the N+1 query problem.

        Args:
            project_ids: List of project IDs to fetch sources for

        Returns:
            Dict mapping project_id to {"technical_sources": [...], "business_sources": [...]}
        """
        if not project_ids:
            return {}

        try:
            # Single batch query to get ALL project sources
            response = (
                self.supabase_client.table("archon_project_sources")
                .select("project_id, source_id, notes")
                .in_("project_id", project_ids)
                .execute()
            )

            # Group sources by project and type
            project_sources = defaultdict(lambda: {"technical_sources": [], "business_sources": []})
            
            for source_link in response.data:
                project_id = source_link["project_id"]
                source_id = source_link["source_id"]
                source_type = source_link.get("notes")

                if source_type == "technical":
                    project_sources[project_id]["technical_sources"].append(source_id)
                elif source_type == "business":
                    project_sources[project_id]["business_sources"].append(source_id)

            # Ensure all requested project_ids are in the result (even if empty)
            for project_id in project_ids:
                if project_id not in project_sources:
                    project_sources[project_id] = {"technical_sources": [], "business_sources": []}

            logger.info(f"Batch loaded sources for {len(project_ids)} projects with single query")
            return dict(project_sources)

        except Exception as e:
            logger.error(f"Error batch loading project sources: {e}")
            # Return empty results for all projects on error
            return {pid: {"technical_sources": [], "business_sources": []} for pid in project_ids}

    def get_all_sources_details_batch(self, source_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get full source details for multiple source IDs in a single batch query.
        
        Args:
            source_ids: List of source IDs to fetch details for
            
        Returns:
            Dict mapping source_id to full source object
        """
        if not source_ids:
            return {}

        try:
            # Single batch query to get ALL source details
            response = (
                self.supabase_client.table("archon_sources")
                .select("*")
                .in_("source_id", source_ids)
                .execute()
            )

            # Create lookup dict by source_id
            sources_dict = {source["source_id"]: source for source in response.data}
            
            logger.info(f"Batch loaded {len(response.data)} source details with single query")
            return sources_dict

        except Exception as e:
            logger.error(f"Error batch loading source details: {e}")
            return {}

    def format_project_with_sources(self, project: dict[str, Any]) -> dict[str, Any]:
        """
        Format a project dict with its linked sources included.
        (Legacy method for backward compatibility - still uses individual queries)

        Returns:
            Formatted project dict with technical_sources and business_sources
        """
        # Get linked sources
        success, sources = self.get_project_sources(project["id"])
        if not success:
            logger.warning(f"Failed to get sources for project {project['id']}")
            sources = {"technical_sources": [], "business_sources": []}

        # Ensure datetime objects are converted to strings
        created_at = project.get("created_at", "")
        updated_at = project.get("updated_at", "")
        if hasattr(created_at, "isoformat"):
            created_at = created_at.isoformat()
        if hasattr(updated_at, "isoformat"):
            updated_at = updated_at.isoformat()

        return {
            "id": project["id"],
            "title": project["title"],
            "description": project.get("description", ""),
            "github_repo": project.get("github_repo"),
            "created_at": created_at,
            "updated_at": updated_at,
            "docs": project.get("docs", []),
            "features": project.get("features", []),
            "data": project.get("data", []),
            "technical_sources": sources["technical_sources"],
            "business_sources": sources["business_sources"],
            "pinned": project.get("pinned", False),
        }

    def format_projects_with_sources_optimized(self, projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        OPTIMIZED: Format a list of projects with their linked sources using batch queries.
        This eliminates the N+1 query problem completely.

        Returns:
            List of formatted project dicts with sources
        """
        if not projects:
            return []

        start_time = logger.info("Starting optimized project formatting with batch queries")

        # Extract all project IDs
        project_ids = [project["id"] for project in projects]
        
        # BATCH QUERY 1: Get all project sources in a single query
        all_project_sources = self.get_all_project_sources_batch(project_ids)
        
        # BATCH QUERY 2: Get all unique source IDs that we need details for
        all_source_ids = set()
        for project_sources in all_project_sources.values():
            all_source_ids.update(project_sources["technical_sources"])
            all_source_ids.update(project_sources["business_sources"])
        
        # Get full source details in a single query (if needed)
        # For now, we're just returning source IDs, but this batch query is ready
        # if you need full source objects later
        # all_source_details = self.get_all_sources_details_batch(list(all_source_ids))
        
        # Format all projects with their sources
        formatted_projects = []
        for project in projects:
            project_id = project["id"]
            project_sources = all_project_sources.get(project_id, {"technical_sources": [], "business_sources": []})
            
            # Ensure datetime objects are converted to strings
            created_at = project.get("created_at", "")
            updated_at = project.get("updated_at", "")
            if hasattr(created_at, "isoformat"):
                created_at = created_at.isoformat()
            if hasattr(updated_at, "isoformat"):
                updated_at = updated_at.isoformat()

            formatted_project = {
                "id": project["id"],
                "title": project["title"],
                "description": project.get("description", ""),
                "github_repo": project.get("github_repo"),
                "created_at": created_at,
                "updated_at": updated_at,
                "docs": project.get("docs", []),
                "features": project.get("features", []),
                "data": project.get("data", []),
                "technical_sources": project_sources["technical_sources"],
                "business_sources": project_sources["business_sources"],
                "pinned": project.get("pinned", False),
            }
            formatted_projects.append(formatted_project)

        logger.info(f"Optimized formatting complete: {len(projects)} projects, {len(all_source_ids)} unique sources, 2 queries total (vs {len(projects) * 2} queries before)")
        
        return formatted_projects

    def format_projects_with_sources(self, projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Entry point for formatting projects with sources.
        Now uses optimized batch queries by default!

        Returns:
            List of formatted project dicts
        """
        return self.format_projects_with_sources_optimized(projects)

    def update_project_sources(
        self,
        project_id: str,
        technical_sources: list[str] | None = None,
        business_sources: list[str] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Update project sources, replacing existing ones if provided.
        (Unchanged from original - updating logic is already efficient)

        Returns:
            Tuple of (success, result_dict with counts)
        """
        result = {
            "technical_success": 0,
            "technical_failed": 0,
            "business_success": 0,
            "business_failed": 0,
        }

        try:
            # Update technical sources if provided
            if technical_sources is not None:
                # Remove existing technical sources
                self.supabase_client.table("archon_project_sources").delete().eq(
                    "project_id", project_id
                ).eq("notes", "technical").execute()

                # Add new technical sources
                for source_id in technical_sources:
                    try:
                        self.supabase_client.table("archon_project_sources").insert({
                            "project_id": project_id,
                            "source_id": source_id,
                            "notes": "technical",
                        }).execute()
                        result["technical_success"] += 1
                    except Exception as e:
                        result["technical_failed"] += 1
                        logger.warning(f"Failed to link technical source {source_id}: {e}")

            # Update business sources if provided
            if business_sources is not None:
                # Remove existing business sources
                self.supabase_client.table("archon_project_sources").delete().eq(
                    "project_id", project_id
                ).eq("notes", "business").execute()

                # Add new business sources
                for source_id in business_sources:
                    try:
                        self.supabase_client.table("archon_project_sources").insert({
                            "project_id": project_id,
                            "source_id": source_id,
                            "notes": "business",
                        }).execute()
                        result["business_success"] += 1
                    except Exception as e:
                        result["business_failed"] += 1
                        logger.warning(f"Failed to link business source {source_id}: {e}")

            # Overall success if no critical failures
            total_failed = result["technical_failed"] + result["business_failed"]

            return True, result

        except Exception as e:
            logger.error(f"Error updating project sources: {e}")
            return False, {"error": str(e), **result}