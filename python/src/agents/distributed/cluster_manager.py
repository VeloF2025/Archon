"""
Advanced Cluster Manager
Manages distributed computing clusters with auto-scaling, health monitoring, and resource allocation
"""

import asyncio
import logging
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import psutil
from collections import defaultdict, deque
import docker
import kubernetes
from kubernetes import client, config

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node status in cluster"""
    PENDING = "pending"
    READY = "ready"
    NOT_READY = "not_ready"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    TERMINATED = "terminated"


class NodeType(Enum):
    """Types of cluster nodes"""
    MASTER = "master"
    WORKER = "worker"
    ETCD = "etcd"
    INGRESS = "ingress"
    STORAGE = "storage"


class ClusterStatus(Enum):
    """Overall cluster status"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SCALING = "scaling"
    MAINTENANCE = "maintenance"


class ResourceType(Enum):
    """Resource types for allocation"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"


@dataclass
class NodeResource:
    """Node resource specification"""
    resource_type: ResourceType
    total: float
    allocated: float = 0.0
    available: float = 0.0
    unit: str = ""
    
    def __post_init__(self):
        self.available = self.total - self.allocated
    
    def allocate(self, amount: float) -> bool:
        """Allocate resources"""
        if self.available >= amount:
            self.allocated += amount
            self.available = self.total - self.allocated
            return True
        return False
    
    def deallocate(self, amount: float) -> None:
        """Deallocate resources"""
        self.allocated = max(0.0, self.allocated - amount)
        self.available = self.total - self.allocated
    
    def utilization_percent(self) -> float:
        """Get resource utilization percentage"""
        return (self.allocated / self.total) * 100 if self.total > 0 else 0.0


@dataclass
class ClusterNode:
    """Individual cluster node"""
    node_id: str
    hostname: str
    ip_address: str
    node_type: NodeType
    status: NodeStatus = NodeStatus.PENDING
    resources: Dict[ResourceType, NodeResource] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: Optional[datetime] = None
    health_checks: List[Dict[str, Any]] = field(default_factory=list)
    running_pods: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if node is healthy"""
        if self.status != NodeStatus.READY:
            return False
        if self.last_heartbeat:
            return datetime.now() - self.last_heartbeat < timedelta(minutes=2)
        return False
    
    def can_schedule(self, required_resources: Dict[ResourceType, float]) -> bool:
        """Check if node can schedule workload with required resources"""
        if not self.is_healthy():
            return False
        
        for resource_type, required_amount in required_resources.items():
            if resource_type not in self.resources:
                return False
            if self.resources[resource_type].available < required_amount:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        resources_dict = {}
        for res_type, resource in self.resources.items():
            resources_dict[res_type.value] = {
                "total": resource.total,
                "allocated": resource.allocated,
                "available": resource.available,
                "unit": resource.unit,
                "utilization": resource.utilization_percent()
            }
        
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "node_type": self.node_type.value,
            "status": self.status.value,
            "resources": resources_dict,
            "labels": self.labels,
            "created_at": self.created_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "is_healthy": self.is_healthy(),
            "running_pods": self.running_pods
        }


@dataclass
class WorkloadSpec:
    """Specification for cluster workload"""
    workload_id: str
    name: str
    image: str
    resource_requirements: Dict[ResourceType, float]
    replicas: int = 1
    labels: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    ports: List[int] = field(default_factory=list)
    volumes: List[Dict[str, str]] = field(default_factory=list)
    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, Any]] = field(default_factory=list)
    affinity_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterWorkload:
    """Running workload in cluster"""
    workload_id: str
    spec: WorkloadSpec
    status: str = "pending"
    scheduled_nodes: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    health_status: str = "unknown"
    restart_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "name": self.spec.name,
            "image": self.spec.image,
            "replicas": self.spec.replicas,
            "status": self.status,
            "scheduled_nodes": self.scheduled_nodes,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "health_status": self.health_status,
            "restart_count": self.restart_count,
            "resource_requirements": {rt.value: req for rt, req in self.spec.resource_requirements.items()}
        }


class AutoScaler:
    """Automatic cluster scaling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_nodes = config.get("min_nodes", 3)
        self.max_nodes = config.get("max_nodes", 100)
        self.scale_up_threshold = config.get("scale_up_threshold", 80.0)  # CPU %
        self.scale_down_threshold = config.get("scale_down_threshold", 20.0)  # CPU %
        self.scale_up_cooldown = timedelta(minutes=config.get("scale_up_cooldown_minutes", 5))
        self.scale_down_cooldown = timedelta(minutes=config.get("scale_down_cooldown_minutes", 10))
        self.last_scale_action: Optional[datetime] = None
        
    def should_scale_up(self, cluster_metrics: Dict[str, float], node_count: int) -> bool:
        """Check if cluster should scale up"""
        if node_count >= self.max_nodes:
            return False
        
        if self.last_scale_action and datetime.now() - self.last_scale_action < self.scale_up_cooldown:
            return False
        
        avg_cpu = cluster_metrics.get("average_cpu_utilization", 0.0)
        return avg_cpu > self.scale_up_threshold
    
    def should_scale_down(self, cluster_metrics: Dict[str, float], node_count: int) -> bool:
        """Check if cluster should scale down"""
        if node_count <= self.min_nodes:
            return False
        
        if self.last_scale_action and datetime.now() - self.last_scale_action < self.scale_down_cooldown:
            return False
        
        avg_cpu = cluster_metrics.get("average_cpu_utilization", 0.0)
        return avg_cpu < self.scale_down_threshold
    
    def record_scale_action(self) -> None:
        """Record that a scaling action occurred"""
        self.last_scale_action = datetime.now()


class ResourceScheduler:
    """Schedules workloads on cluster nodes"""
    
    def __init__(self):
        self.placement_strategies = {
            "round_robin": self._round_robin_placement,
            "least_utilized": self._least_utilized_placement,
            "best_fit": self._best_fit_placement,
            "affinity_based": self._affinity_based_placement
        }
        self.default_strategy = "best_fit"
    
    def schedule_workload(self, workload: WorkloadSpec, available_nodes: List[ClusterNode],
                         strategy: str = None) -> List[str]:
        """Schedule workload on appropriate nodes"""
        strategy = strategy or self.default_strategy
        
        if strategy not in self.placement_strategies:
            strategy = self.default_strategy
        
        placement_func = self.placement_strategies[strategy]
        return placement_func(workload, available_nodes)
    
    def _round_robin_placement(self, workload: WorkloadSpec, nodes: List[ClusterNode]) -> List[str]:
        """Round-robin placement strategy"""
        suitable_nodes = [node for node in nodes 
                         if node.can_schedule(workload.resource_requirements)]
        
        if not suitable_nodes:
            return []
        
        scheduled_nodes = []
        for i in range(workload.replicas):
            node = suitable_nodes[i % len(suitable_nodes)]
            scheduled_nodes.append(node.node_id)
        
        return scheduled_nodes
    
    def _least_utilized_placement(self, workload: WorkloadSpec, nodes: List[ClusterNode]) -> List[str]:
        """Place on least utilized nodes"""
        suitable_nodes = [node for node in nodes 
                         if node.can_schedule(workload.resource_requirements)]
        
        if not suitable_nodes:
            return []
        
        # Sort by CPU utilization
        suitable_nodes.sort(key=lambda n: n.resources.get(ResourceType.CPU, 
                                                         NodeResource(ResourceType.CPU, 0)).utilization_percent())
        
        scheduled_nodes = []
        for i in range(min(workload.replicas, len(suitable_nodes))):
            scheduled_nodes.append(suitable_nodes[i].node_id)
        
        return scheduled_nodes
    
    def _best_fit_placement(self, workload: WorkloadSpec, nodes: List[ClusterNode]) -> List[str]:
        """Best fit placement strategy"""
        suitable_nodes = [node for node in nodes 
                         if node.can_schedule(workload.resource_requirements)]
        
        if not suitable_nodes:
            return []
        
        def fit_score(node: ClusterNode) -> float:
            """Calculate how well workload fits on node"""
            score = 0.0
            for res_type, required in workload.resource_requirements.items():
                if res_type in node.resources:
                    available = node.resources[res_type].available
                    if available >= required:
                        # Prefer nodes where resource usage will be high but not wasteful
                        utilization = required / available
                        score += min(utilization, 1.0)
            return score
        
        # Sort by best fit score
        suitable_nodes.sort(key=fit_score, reverse=True)
        
        scheduled_nodes = []
        for i in range(min(workload.replicas, len(suitable_nodes))):
            scheduled_nodes.append(suitable_nodes[i].node_id)
        
        return scheduled_nodes
    
    def _affinity_based_placement(self, workload: WorkloadSpec, nodes: List[ClusterNode]) -> List[str]:
        """Placement based on node affinity rules"""
        suitable_nodes = [node for node in nodes 
                         if node.can_schedule(workload.resource_requirements)]
        
        if not suitable_nodes:
            return []
        
        # Filter by node selector
        if workload.node_selector:
            filtered_nodes = []
            for node in suitable_nodes:
                matches = True
                for key, value in workload.node_selector.items():
                    if key not in node.labels or node.labels[key] != value:
                        matches = False
                        break
                if matches:
                    filtered_nodes.append(node)
            suitable_nodes = filtered_nodes
        
        if not suitable_nodes:
            return []
        
        # Apply affinity rules (simplified)
        # In production, would implement full Kubernetes affinity logic
        
        scheduled_nodes = []
        for i in range(min(workload.replicas, len(suitable_nodes))):
            scheduled_nodes.append(suitable_nodes[i].node_id)
        
        return scheduled_nodes


class ClusterManager:
    """Main cluster management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.cluster_id = self.config.get("cluster_id", str(uuid.uuid4()))
        self.cluster_name = self.config.get("cluster_name", "archon-cluster")
        
        # Cluster state
        self.nodes: Dict[str, ClusterNode] = {}
        self.workloads: Dict[str, ClusterWorkload] = {}
        self.cluster_status = ClusterStatus.INITIALIZING
        
        # Components
        self.auto_scaler = AutoScaler(self.config.get("auto_scaling", {}))
        self.scheduler = ResourceScheduler()
        
        # Metrics and monitoring
        self.metrics: Dict[str, float] = {}
        self.events: deque = deque(maxlen=1000)
        
        # External integrations
        self.docker_client = None
        self.k8s_client = None
        
        # Initialize integrations
        self._initialize_integrations()
        
    def _initialize_integrations(self) -> None:
        """Initialize Docker and Kubernetes integrations"""
        try:
            # Docker integration
            if self.config.get("enable_docker", True):
                self.docker_client = docker.from_env()
                logger.info("Docker integration initialized")
        except Exception as e:
            logger.warning(f"Docker integration failed: {str(e)}")
        
        try:
            # Kubernetes integration
            if self.config.get("enable_kubernetes", False):
                if self.config.get("k8s_config_file"):
                    config.load_kube_config(config_file=self.config["k8s_config_file"])
                else:
                    config.load_incluster_config()  # For in-cluster execution
                
                self.k8s_client = client.CoreV1Api()
                logger.info("Kubernetes integration initialized")
        except Exception as e:
            logger.warning(f"Kubernetes integration failed: {str(e)}")
    
    async def add_node(self, hostname: str, ip_address: str, node_type: NodeType = NodeType.WORKER,
                      labels: Dict[str, str] = None) -> ClusterNode:
        """Add new node to cluster"""
        node_id = f"node-{hashlib.md5(f'{hostname}{ip_address}'.encode()).hexdigest()[:8]}"
        
        # Detect node resources
        resources = await self._detect_node_resources(ip_address)
        
        node = ClusterNode(
            node_id=node_id,
            hostname=hostname,
            ip_address=ip_address,
            node_type=node_type,
            resources=resources,
            labels=labels or {}
        )
        
        self.nodes[node_id] = node
        
        # Start health monitoring
        asyncio.create_task(self._monitor_node_health(node_id))
        
        self._log_event("node_added", f"Added node {hostname} ({node_id})")
        logger.info(f"Added node {hostname} to cluster")
        
        return node
    
    async def _detect_node_resources(self, ip_address: str) -> Dict[ResourceType, NodeResource]:
        """Detect node resources"""
        # In production, would query actual node via SSH/agent
        # For now, return default resources
        resources = {
            ResourceType.CPU: NodeResource(ResourceType.CPU, 4.0, unit="cores"),
            ResourceType.MEMORY: NodeResource(ResourceType.MEMORY, 16.0, unit="GB"),
            ResourceType.DISK: NodeResource(ResourceType.DISK, 100.0, unit="GB"),
            ResourceType.NETWORK: NodeResource(ResourceType.NETWORK, 1.0, unit="Gbps")
        }
        
        return resources
    
    async def remove_node(self, node_id: str, drain: bool = True) -> bool:
        """Remove node from cluster"""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        
        if drain:
            # Drain workloads from node
            await self._drain_node(node_id)
        
        # Update node status
        node.status = NodeStatus.TERMINATED
        
        # Remove from cluster after grace period
        await asyncio.sleep(30)  # 30 second grace period
        del self.nodes[node_id]
        
        self._log_event("node_removed", f"Removed node {node.hostname} ({node_id})")
        logger.info(f"Removed node {node.hostname} from cluster")
        
        return True
    
    async def _drain_node(self, node_id: str) -> None:
        """Drain all workloads from node"""
        node = self.nodes.get(node_id)
        if not node:
            return
        
        node.status = NodeStatus.DRAINING
        
        # Find workloads running on this node
        workloads_to_reschedule = []
        for workload in self.workloads.values():
            if node_id in workload.scheduled_nodes:
                workloads_to_reschedule.append(workload)
        
        # Reschedule workloads to other nodes
        for workload in workloads_to_reschedule:
            await self._reschedule_workload(workload.workload_id, exclude_nodes=[node_id])
        
        logger.info(f"Drained {len(workloads_to_reschedule)} workloads from node {node_id}")
    
    async def schedule_workload(self, workload_spec: WorkloadSpec, 
                               scheduling_strategy: str = "best_fit") -> ClusterWorkload:
        """Schedule new workload on cluster"""
        # Get available nodes
        available_nodes = [node for node in self.nodes.values() 
                          if node.status == NodeStatus.READY]
        
        if not available_nodes:
            raise RuntimeError("No available nodes for scheduling")
        
        # Schedule workload
        scheduled_node_ids = self.scheduler.schedule_workload(
            workload_spec, available_nodes, scheduling_strategy
        )
        
        if not scheduled_node_ids:
            raise RuntimeError("No suitable nodes found for workload")
        
        # Create workload
        workload = ClusterWorkload(
            workload_id=workload_spec.workload_id,
            spec=workload_spec,
            scheduled_nodes=scheduled_node_ids
        )
        
        self.workloads[workload_spec.workload_id] = workload
        
        # Allocate resources on scheduled nodes
        for node_id in scheduled_node_ids:
            node = self.nodes[node_id]
            for resource_type, amount in workload_spec.resource_requirements.items():
                if resource_type in node.resources:
                    node.resources[resource_type].allocate(amount)
            node.running_pods += 1
        
        # Deploy workload
        await self._deploy_workload(workload)
        
        self._log_event("workload_scheduled", 
                       f"Scheduled workload {workload_spec.name} on {len(scheduled_node_ids)} nodes")
        
        return workload
    
    async def _deploy_workload(self, workload: ClusterWorkload) -> None:
        """Deploy workload to scheduled nodes"""
        workload.status = "deploying"
        
        try:
            if self.k8s_client:
                await self._deploy_to_kubernetes(workload)
            elif self.docker_client:
                await self._deploy_to_docker(workload)
            else:
                # Simulate deployment
                await asyncio.sleep(2)
                workload.status = "running"
                workload.started_at = datetime.now()
                workload.health_status = "healthy"
                
        except Exception as e:
            logger.error(f"Failed to deploy workload {workload.workload_id}: {str(e)}")
            workload.status = "failed"
            workload.health_status = "unhealthy"
    
    async def _deploy_to_kubernetes(self, workload: ClusterWorkload) -> None:
        """Deploy workload to Kubernetes"""
        # Would implement Kubernetes deployment creation
        # This is a simplified placeholder
        workload.status = "running"
        workload.started_at = datetime.now()
        workload.health_status = "healthy"
    
    async def _deploy_to_docker(self, workload: ClusterWorkload) -> None:
        """Deploy workload to Docker"""
        try:
            # Create Docker containers
            for node_id in workload.scheduled_nodes:
                container = self.docker_client.containers.run(
                    image=workload.spec.image,
                    name=f"{workload.spec.name}-{node_id}",
                    environment=workload.spec.environment,
                    detach=True,
                    labels=workload.spec.labels
                )
                
                workload.metadata[f"container_{node_id}"] = container.id
            
            workload.status = "running"
            workload.started_at = datetime.now()
            workload.health_status = "healthy"
            
        except Exception as e:
            logger.error(f"Docker deployment failed: {str(e)}")
            raise
    
    async def _reschedule_workload(self, workload_id: str, exclude_nodes: List[str] = None) -> bool:
        """Reschedule workload to different nodes"""
        workload = self.workloads.get(workload_id)
        if not workload:
            return False
        
        exclude_nodes = exclude_nodes or []
        
        # Get available nodes (excluding specified nodes)
        available_nodes = [node for node in self.nodes.values() 
                          if (node.status == NodeStatus.READY and 
                              node.node_id not in exclude_nodes)]
        
        if not available_nodes:
            logger.error(f"No available nodes for rescheduling workload {workload_id}")
            return False
        
        # Deallocate resources from old nodes
        for node_id in workload.scheduled_nodes:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                for resource_type, amount in workload.spec.resource_requirements.items():
                    if resource_type in node.resources:
                        node.resources[resource_type].deallocate(amount)
                node.running_pods = max(0, node.running_pods - 1)
        
        # Schedule on new nodes
        new_scheduled_nodes = self.scheduler.schedule_workload(
            workload.spec, available_nodes
        )
        
        if new_scheduled_nodes:
            workload.scheduled_nodes = new_scheduled_nodes
            
            # Allocate resources on new nodes
            for node_id in new_scheduled_nodes:
                node = self.nodes[node_id]
                for resource_type, amount in workload.spec.resource_requirements.items():
                    if resource_type in node.resources:
                        node.resources[resource_type].allocate(amount)
                node.running_pods += 1
            
            # Redeploy workload
            await self._deploy_workload(workload)
            
            logger.info(f"Rescheduled workload {workload_id} to new nodes")
            return True
        
        return False
    
    async def scale_workload(self, workload_id: str, target_replicas: int) -> bool:
        """Scale workload to target replica count"""
        workload = self.workloads.get(workload_id)
        if not workload:
            return False
        
        current_replicas = workload.spec.replicas
        
        if target_replicas == current_replicas:
            return True
        
        workload.spec.replicas = target_replicas
        
        if target_replicas > current_replicas:
            # Scale up - schedule additional replicas
            additional_replicas = target_replicas - current_replicas
            
            # Create temporary spec for additional replicas
            scale_spec = WorkloadSpec(
                workload_id=f"{workload_id}-scale",
                name=workload.spec.name,
                image=workload.spec.image,
                resource_requirements=workload.spec.resource_requirements,
                replicas=additional_replicas,
                labels=workload.spec.labels,
                environment=workload.spec.environment
            )
            
            available_nodes = [node for node in self.nodes.values() 
                             if node.status == NodeStatus.READY]
            
            additional_nodes = self.scheduler.schedule_workload(scale_spec, available_nodes)
            
            if additional_nodes:
                workload.scheduled_nodes.extend(additional_nodes)
                
                # Allocate resources
                for node_id in additional_nodes:
                    node = self.nodes[node_id]
                    for resource_type, amount in workload.spec.resource_requirements.items():
                        if resource_type in node.resources:
                            node.resources[resource_type].allocate(amount)
                    node.running_pods += 1
        
        else:
            # Scale down - remove excess replicas
            excess_replicas = current_replicas - target_replicas
            nodes_to_remove = workload.scheduled_nodes[:excess_replicas]
            
            # Deallocate resources
            for node_id in nodes_to_remove:
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    for resource_type, amount in workload.spec.resource_requirements.items():
                        if resource_type in node.resources:
                            node.resources[resource_type].deallocate(amount)
                    node.running_pods = max(0, node.running_pods - 1)
            
            workload.scheduled_nodes = workload.scheduled_nodes[excess_replicas:]
        
        # Redeploy workload with new replica count
        await self._deploy_workload(workload)
        
        self._log_event("workload_scaled", 
                       f"Scaled workload {workload_id} from {current_replicas} to {target_replicas} replicas")
        
        return True
    
    async def _monitor_node_health(self, node_id: str) -> None:
        """Monitor node health continuously"""
        while node_id in self.nodes:
            try:
                node = self.nodes[node_id]
                
                # Perform health check
                health_result = await self._perform_health_check(node)
                
                # Update node status based on health check
                if health_result["healthy"]:
                    if node.status == NodeStatus.PENDING:
                        node.status = NodeStatus.READY
                    node.last_heartbeat = datetime.now()
                else:
                    if node.status == NodeStatus.READY:
                        node.status = NodeStatus.NOT_READY
                
                # Store health check result
                node.health_checks.append({
                    "timestamp": datetime.now().isoformat(),
                    "healthy": health_result["healthy"],
                    "details": health_result.get("details", {})
                })
                
                # Keep only last 10 health checks
                if len(node.health_checks) > 10:
                    node.health_checks = node.health_checks[-10:]
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitoring error for node {node_id}: {str(e)}")
                await asyncio.sleep(60)
    
    async def _perform_health_check(self, node: ClusterNode) -> Dict[str, Any]:
        """Perform health check on node"""
        try:
            # In production, would perform actual health checks:
            # - Network connectivity
            # - Resource availability  
            # - Service status
            # - Disk space
            
            # Simulate health check
            import random
            healthy = random.random() > 0.05  # 95% success rate
            
            return {
                "healthy": healthy,
                "details": {
                    "network": "ok" if healthy else "timeout",
                    "disk_space": 85.0,
                    "load_average": 0.5,
                    "memory_usage": 70.0
                }
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "details": {"error": str(e)}
            }
    
    async def auto_scale(self) -> None:
        """Perform automatic scaling based on cluster metrics"""
        try:
            # Calculate cluster metrics
            cluster_metrics = await self._calculate_cluster_metrics()
            
            current_node_count = len([n for n in self.nodes.values() 
                                    if n.status == NodeStatus.READY])
            
            # Check if scaling is needed
            if self.auto_scaler.should_scale_up(cluster_metrics, current_node_count):
                await self._scale_cluster_up()
                self.auto_scaler.record_scale_action()
                
            elif self.auto_scaler.should_scale_down(cluster_metrics, current_node_count):
                await self._scale_cluster_down()
                self.auto_scaler.record_scale_action()
                
        except Exception as e:
            logger.error(f"Auto-scaling error: {str(e)}")
    
    async def _calculate_cluster_metrics(self) -> Dict[str, float]:
        """Calculate cluster-wide metrics"""
        if not self.nodes:
            return {}
        
        ready_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.READY]
        
        if not ready_nodes:
            return {}
        
        # Calculate average resource utilization
        total_cpu_utilization = 0.0
        total_memory_utilization = 0.0
        
        for node in ready_nodes:
            if ResourceType.CPU in node.resources:
                total_cpu_utilization += node.resources[ResourceType.CPU].utilization_percent()
            if ResourceType.MEMORY in node.resources:
                total_memory_utilization += node.resources[ResourceType.MEMORY].utilization_percent()
        
        node_count = len(ready_nodes)
        
        metrics = {
            "average_cpu_utilization": total_cpu_utilization / node_count,
            "average_memory_utilization": total_memory_utilization / node_count,
            "total_nodes": len(self.nodes),
            "ready_nodes": node_count,
            "total_workloads": len(self.workloads),
            "running_workloads": len([w for w in self.workloads.values() if w.status == "running"])
        }
        
        self.metrics.update(metrics)
        return metrics
    
    async def _scale_cluster_up(self) -> None:
        """Scale cluster up by adding nodes"""
        # In production, would integrate with cloud provider APIs
        # to provision new nodes
        
        new_node_id = f"auto-node-{int(time.time())}"
        await self.add_node(
            hostname=f"auto-{new_node_id}",
            ip_address=f"10.0.0.{len(self.nodes) + 100}",
            node_type=NodeType.WORKER
        )
        
        self.cluster_status = ClusterStatus.SCALING
        self._log_event("cluster_scaled_up", f"Added node {new_node_id} via auto-scaling")
        
        # Reset status after scaling
        await asyncio.sleep(60)
        self.cluster_status = ClusterStatus.HEALTHY
    
    async def _scale_cluster_down(self) -> None:
        """Scale cluster down by removing nodes"""
        # Find least utilized node
        ready_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.READY]
        
        if len(ready_nodes) <= self.auto_scaler.min_nodes:
            return
        
        # Sort by utilization (lowest first)
        ready_nodes.sort(key=lambda n: n.resources.get(ResourceType.CPU, 
                                                      NodeResource(ResourceType.CPU, 0)).utilization_percent())
        
        node_to_remove = ready_nodes[0]
        
        self.cluster_status = ClusterStatus.SCALING
        await self.remove_node(node_to_remove.node_id, drain=True)
        
        self._log_event("cluster_scaled_down", f"Removed node {node_to_remove.node_id} via auto-scaling")
        
        # Reset status after scaling
        await asyncio.sleep(60)
        self.cluster_status = ClusterStatus.HEALTHY
    
    def _log_event(self, event_type: str, message: str) -> None:
        """Log cluster event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "message": message,
            "cluster_id": self.cluster_id
        }
        self.events.append(event)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        ready_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.READY])
        unhealthy_nodes = len([n for n in self.nodes.values() if not n.is_healthy()])
        
        # Determine overall cluster health
        if ready_nodes == 0:
            status = ClusterStatus.UNHEALTHY
        elif unhealthy_nodes > len(self.nodes) * 0.3:  # More than 30% unhealthy
            status = ClusterStatus.DEGRADED
        else:
            status = ClusterStatus.HEALTHY
            
        self.cluster_status = status
        
        return {
            "cluster_id": self.cluster_id,
            "cluster_name": self.cluster_name,
            "status": self.cluster_status.value,
            "total_nodes": len(self.nodes),
            "ready_nodes": ready_nodes,
            "unhealthy_nodes": unhealthy_nodes,
            "total_workloads": len(self.workloads),
            "running_workloads": len([w for w in self.workloads.values() if w.status == "running"]),
            "cluster_metrics": self.metrics,
            "auto_scaling_enabled": True,
            "last_updated": datetime.now().isoformat()
        }
    
    def list_nodes(self) -> List[Dict[str, Any]]:
        """List all cluster nodes"""
        return [node.to_dict() for node in self.nodes.values()]
    
    def list_workloads(self) -> List[Dict[str, Any]]:
        """List all cluster workloads"""
        return [workload.to_dict() for workload in self.workloads.values()]
    
    def get_cluster_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent cluster events"""
        return list(self.events)[-limit:]
    
    async def start_monitoring(self) -> None:
        """Start cluster monitoring and auto-scaling"""
        logger.info("Starting cluster monitoring and auto-scaling")
        
        while True:
            try:
                # Update cluster metrics
                await self._calculate_cluster_metrics()
                
                # Perform auto-scaling
                await self.auto_scale()
                
                # Sleep for monitoring interval
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                await asyncio.sleep(30)
                
    async def stop_monitoring(self) -> None:
        """Stop cluster monitoring"""
        logger.info("Stopping cluster monitoring")
        # Would cancel monitoring tasks in production