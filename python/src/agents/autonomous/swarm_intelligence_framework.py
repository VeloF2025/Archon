#!/usr/bin/env python3
"""
Swarm Intelligence Framework Module

This module provides comprehensive swarm intelligence capabilities for autonomous AI agents.
It implements collective intelligence algorithms, emergent behavior patterns, distributed
decision-making, and swarm coordination mechanisms to enable sophisticated group behaviors.

Created: 2025-01-09
Author: Archon Enhancement System
Version: 7.0.0
"""

import asyncio
import json
import uuid
import math
import random
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SwarmAlgorithm(Enum):
    """Types of swarm intelligence algorithms"""
    PARTICLE_SWARM_OPTIMIZATION = auto()
    ANT_COLONY_OPTIMIZATION = auto()
    BEE_COLONY_OPTIMIZATION = auto()
    FIREFLY_ALGORITHM = auto()
    WOLF_PACK_OPTIMIZATION = auto()
    FISH_SCHOOL_SEARCH = auto()
    BIRD_FLOCKING = auto()
    BACTERIAL_FORAGING = auto()
    CUCKOO_SEARCH = auto()
    BAT_ALGORITHM = auto()


class SwarmBehavior(Enum):
    """Types of swarm behaviors"""
    FORAGING = auto()               # Resource gathering
    EXPLORATION = auto()            # Environment exploration
    CONSENSUS = auto()             # Collective decision making
    FORMATION = auto()             # Spatial organization
    MIGRATION = auto()             # Coordinated movement
    DEFENSE = auto()               # Collective protection
    OPTIMIZATION = auto()          # Problem solving
    LEARNING = auto()              # Collective learning
    ADAPTATION = auto()            # Environmental adaptation
    COOPERATION = auto()           # Task cooperation


class SwarmRole(Enum):
    """Roles within a swarm"""
    LEADER = auto()                # Leadership role
    SCOUT = auto()                 # Exploration role
    WORKER = auto()                # Task execution role
    GUARD = auto()                 # Protection role
    FOLLOWER = auto()              # Following role
    COORDINATOR = auto()           # Coordination role
    SPECIALIST = auto()            # Specialized role
    GENERALIST = auto()            # General purpose role


class SwarmState(Enum):
    """States of swarm operation"""
    INITIALIZING = auto()
    ACTIVE = auto()
    CONVERGING = auto()
    EXPLORING = auto()
    EXPLOITING = auto()
    MIGRATING = auto()
    DORMANT = auto()
    TERMINATED = auto()


@dataclass
class SwarmPosition:
    """Represents position in swarm space"""
    coordinates: List[float]
    dimensions: int
    bounds: Optional[Tuple[List[float], List[float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def distance_to(self, other: 'SwarmPosition') -> float:
        """Calculate distance to another position"""
        if self.dimensions != other.dimensions:
            raise ValueError("Position dimensions must match")
        
        return math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(self.coordinates, other.coordinates)
        ))
    
    def move_towards(self, target: 'SwarmPosition', step_size: float) -> 'SwarmPosition':
        """Move towards target position"""
        direction = [
            target.coordinates[i] - self.coordinates[i]
            for i in range(self.dimensions)
        ]
        
        distance = math.sqrt(sum(d ** 2 for d in direction))
        if distance == 0:
            return SwarmPosition(
                coordinates=self.coordinates.copy(),
                dimensions=self.dimensions,
                bounds=self.bounds
            )
        
        # Normalize direction and apply step size
        normalized_direction = [d / distance for d in direction]
        new_coordinates = [
            self.coordinates[i] + normalized_direction[i] * step_size
            for i in range(self.dimensions)
        ]
        
        # Apply bounds if specified
        if self.bounds:
            lower_bounds, upper_bounds = self.bounds
            new_coordinates = [
                max(lower_bounds[i], min(upper_bounds[i], coord))
                for i, coord in enumerate(new_coordinates)
            ]
        
        return SwarmPosition(
            coordinates=new_coordinates,
            dimensions=self.dimensions,
            bounds=self.bounds
        )


@dataclass
class SwarmAgent:
    """Represents an agent within a swarm"""
    agent_id: str
    swarm_id: str
    role: SwarmRole
    position: SwarmPosition
    velocity: List[float] = field(default_factory=list)
    fitness: float = 0.0
    best_position: Optional[SwarmPosition] = None
    best_fitness: float = float('-inf')
    
    # Agent properties
    capabilities: Set[str] = field(default_factory=set)
    energy: float = 100.0
    communication_range: float = 10.0
    influence_radius: float = 5.0
    learning_rate: float = 0.1
    
    # Behavioral parameters
    exploration_factor: float = 0.5
    exploitation_factor: float = 0.5
    social_factor: float = 0.5
    cognitive_factor: float = 0.5
    
    # State tracking
    state: Dict[str, Any] = field(default_factory=dict)
    neighbors: Set[str] = field(default_factory=set)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)
    
    # Performance metrics
    decisions_made: int = 0
    tasks_completed: int = 0
    resources_gathered: int = 0
    distance_traveled: float = 0.0
    
    def update_fitness(self, new_fitness: float) -> None:
        """Update agent fitness and best position"""
        self.fitness = new_fitness
        if new_fitness > self.best_fitness:
            self.best_fitness = new_fitness
            self.best_position = SwarmPosition(
                coordinates=self.position.coordinates.copy(),
                dimensions=self.position.dimensions,
                bounds=self.position.bounds
            )
    
    def add_neighbor(self, neighbor_id: str) -> None:
        """Add a neighbor agent"""
        self.neighbors.add(neighbor_id)
    
    def remove_neighbor(self, neighbor_id: str) -> None:
        """Remove a neighbor agent"""
        self.neighbors.discard(neighbor_id)
    
    def send_message(self, message: Dict[str, Any]) -> None:
        """Send a message to the swarm"""
        message['sender_id'] = self.agent_id
        message['timestamp'] = datetime.now()
        # Message would be distributed by swarm coordinator
    
    def receive_message(self, message: Dict[str, Any]) -> None:
        """Receive a message from another agent"""
        self.messages.append(message)
        # Limit message history
        if len(self.messages) > 100:
            self.messages = self.messages[-100:]


@dataclass
class SwarmObjective:
    """Represents an objective for the swarm"""
    objective_id: str
    name: str
    description: str
    objective_type: str
    target_value: Optional[float] = None
    maximization: bool = True
    weight: float = 1.0
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_function: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmEnvironment:
    """Represents the environment in which the swarm operates"""
    environment_id: str
    dimensions: int
    bounds: Tuple[List[float], List[float]]
    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    resources: List[Dict[str, Any]] = field(default_factory=list)
    hazards: List[Dict[str, Any]] = field(default_factory=list)
    dynamic_features: List[Dict[str, Any]] = field(default_factory=list)
    
    # Environment properties
    temperature: float = 25.0
    humidity: float = 0.5
    wind_speed: float = 0.0
    wind_direction: float = 0.0
    visibility: float = 1.0
    
    # Environmental dynamics
    time_step: int = 0
    change_rate: float = 0.1
    noise_level: float = 0.1
    
    def get_fitness_at_position(self, position: SwarmPosition) -> float:
        """Get fitness value at a position"""
        # Simple fitness landscape example
        x, y = position.coordinates[0], position.coordinates[1] if len(position.coordinates) > 1 else 0
        
        # Multi-modal fitness function
        fitness = (
            10 * math.exp(-((x - 2) ** 2 + (y - 2) ** 2) / 10) +
            8 * math.exp(-((x + 2) ** 2 + (y + 2) ** 2) / 8) +
            6 * math.exp(-((x - 1) ** 2 + (y + 1) ** 2) / 6)
        )
        
        # Add noise
        fitness += random.gauss(0, self.noise_level)
        
        return fitness
    
    def is_valid_position(self, position: SwarmPosition) -> bool:
        """Check if position is valid (not in obstacle)"""
        # Check bounds
        lower_bounds, upper_bounds = self.bounds
        for i, coord in enumerate(position.coordinates):
            if coord < lower_bounds[i] or coord > upper_bounds[i]:
                return False
        
        # Check obstacles
        for obstacle in self.obstacles:
            obs_pos = obstacle.get('position', [0, 0])
            obs_radius = obstacle.get('radius', 1.0)
            
            if position.distance_to(SwarmPosition(
                coordinates=obs_pos,
                dimensions=self.dimensions
            )) < obs_radius:
                return False
        
        return True
    
    def update_environment(self) -> None:
        """Update dynamic environment features"""
        self.time_step += 1
        
        # Update dynamic features
        for feature in self.dynamic_features:
            if feature.get('type') == 'moving_resource':
                # Move resources randomly
                position = feature.get('position', [0, 0])
                for i in range(len(position)):
                    position[i] += random.gauss(0, 0.1)
                    # Keep within bounds
                    position[i] = max(self.bounds[0][i], min(self.bounds[1][i], position[i]))


@dataclass
class SwarmMetrics:
    """Metrics for swarm performance"""
    total_agents: int = 0
    active_agents: int = 0
    best_fitness: float = float('-inf')
    average_fitness: float = 0.0
    diversity_measure: float = 0.0
    convergence_rate: float = 0.0
    exploration_coverage: float = 0.0
    communication_efficiency: float = 0.0
    energy_consumption: float = 0.0
    task_completion_rate: float = 0.0
    collective_intelligence_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class BaseSwarmAlgorithm(ABC):
    """Abstract base class for swarm algorithms"""
    
    @abstractmethod
    async def initialize_swarm(self, swarm_size: int, environment: SwarmEnvironment) -> List[SwarmAgent]:
        """Initialize swarm agents"""
        pass
    
    @abstractmethod
    async def update_agents(self, agents: List[SwarmAgent], environment: SwarmEnvironment) -> None:
        """Update agent positions and states"""
        pass
    
    @abstractmethod
    async def evaluate_fitness(self, agents: List[SwarmAgent], environment: SwarmEnvironment) -> None:
        """Evaluate fitness of agents"""
        pass
    
    @abstractmethod
    def get_best_solution(self, agents: List[SwarmAgent]) -> Tuple[SwarmPosition, float]:
        """Get the best solution found by the swarm"""
        pass


class ParticleSwarmOptimization(BaseSwarmAlgorithm):
    """Particle Swarm Optimization algorithm implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.inertia_weight = config.get('inertia_weight', 0.9)
        self.cognitive_factor = config.get('cognitive_factor', 2.0)
        self.social_factor = config.get('social_factor', 2.0)
        self.max_velocity = config.get('max_velocity', 1.0)
        self.global_best_position: Optional[SwarmPosition] = None
        self.global_best_fitness = float('-inf')
    
    async def initialize_swarm(self, swarm_size: int, environment: SwarmEnvironment) -> List[SwarmAgent]:
        """Initialize PSO swarm"""
        agents = []
        
        for i in range(swarm_size):
            # Random initial position within bounds
            position_coords = [
                random.uniform(environment.bounds[0][j], environment.bounds[1][j])
                for j in range(environment.dimensions)
            ]
            
            position = SwarmPosition(
                coordinates=position_coords,
                dimensions=environment.dimensions,
                bounds=environment.bounds
            )
            
            # Random initial velocity
            velocity = [
                random.uniform(-self.max_velocity, self.max_velocity)
                for _ in range(environment.dimensions)
            ]
            
            agent = SwarmAgent(
                agent_id=f"pso_agent_{i}",
                swarm_id="pso_swarm",
                role=SwarmRole.WORKER,
                position=position,
                velocity=velocity,
                cognitive_factor=self.cognitive_factor,
                social_factor=self.social_factor
            )
            
            agents.append(agent)
        
        return agents
    
    async def update_agents(self, agents: List[SwarmAgent], environment: SwarmEnvironment) -> None:
        """Update PSO agent positions"""
        for agent in agents:
            # Update velocity
            new_velocity = []
            
            for i in range(environment.dimensions):
                # Inertia component
                inertia = self.inertia_weight * agent.velocity[i]
                
                # Cognitive component (personal best)
                cognitive = 0.0
                if agent.best_position:
                    cognitive = (
                        self.cognitive_factor * random.random() *
                        (agent.best_position.coordinates[i] - agent.position.coordinates[i])
                    )
                
                # Social component (global best)
                social = 0.0
                if self.global_best_position:
                    social = (
                        self.social_factor * random.random() *
                        (self.global_best_position.coordinates[i] - agent.position.coordinates[i])
                    )
                
                # New velocity
                v = inertia + cognitive + social
                
                # Clamp velocity
                v = max(-self.max_velocity, min(self.max_velocity, v))
                new_velocity.append(v)
            
            agent.velocity = new_velocity
            
            # Update position
            new_coordinates = [
                agent.position.coordinates[i] + agent.velocity[i]
                for i in range(environment.dimensions)
            ]
            
            # Apply bounds
            lower_bounds, upper_bounds = environment.bounds
            new_coordinates = [
                max(lower_bounds[i], min(upper_bounds[i], coord))
                for i, coord in enumerate(new_coordinates)
            ]
            
            # Update distance traveled
            old_position = SwarmPosition(
                coordinates=agent.position.coordinates,
                dimensions=environment.dimensions
            )
            
            agent.position = SwarmPosition(
                coordinates=new_coordinates,
                dimensions=environment.dimensions,
                bounds=environment.bounds
            )
            
            agent.distance_traveled += old_position.distance_to(agent.position)
    
    async def evaluate_fitness(self, agents: List[SwarmAgent], environment: SwarmEnvironment) -> None:
        """Evaluate fitness for PSO agents"""
        for agent in agents:
            fitness = environment.get_fitness_at_position(agent.position)
            agent.update_fitness(fitness)
            
            # Update global best
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = SwarmPosition(
                    coordinates=agent.position.coordinates.copy(),
                    dimensions=agent.position.dimensions,
                    bounds=agent.position.bounds
                )
    
    def get_best_solution(self, agents: List[SwarmAgent]) -> Tuple[SwarmPosition, float]:
        """Get best PSO solution"""
        if self.global_best_position:
            return self.global_best_position, self.global_best_fitness
        
        # Find best among current agents
        best_agent = max(agents, key=lambda a: a.fitness)
        return best_agent.position, best_agent.fitness


class AntColonyOptimization(BaseSwarmAlgorithm):
    """Ant Colony Optimization algorithm implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alpha = config.get('alpha', 1.0)  # Pheromone importance
        self.beta = config.get('beta', 2.0)    # Heuristic importance
        self.evaporation_rate = config.get('evaporation_rate', 0.1)
        self.pheromone_deposit = config.get('pheromone_deposit', 1.0)
        self.pheromone_matrix: Dict[Tuple, float] = defaultdict(float)
        self.best_path: List[SwarmPosition] = []
        self.best_fitness = float('-inf')
    
    async def initialize_swarm(self, swarm_size: int, environment: SwarmEnvironment) -> List[SwarmAgent]:
        """Initialize ACO swarm"""
        agents = []
        
        for i in range(swarm_size):
            # Start at random position
            position_coords = [
                random.uniform(environment.bounds[0][j], environment.bounds[1][j])
                for j in range(environment.dimensions)
            ]
            
            position = SwarmPosition(
                coordinates=position_coords,
                dimensions=environment.dimensions,
                bounds=environment.bounds
            )
            
            agent = SwarmAgent(
                agent_id=f"ant_{i}",
                swarm_id="aco_swarm",
                role=SwarmRole.SCOUT,
                position=position,
                state={'path': [position], 'visited': set()}
            )
            
            agents.append(agent)
        
        return agents
    
    async def update_agents(self, agents: List[SwarmAgent], environment: SwarmEnvironment) -> None:
        """Update ACO agent positions"""
        for agent in agents:
            # Choose next position based on pheromone and heuristic
            current_pos = agent.position
            
            # Generate candidate positions
            candidates = []
            for _ in range(8):  # 8 directions
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0.1, 1.0)
                
                new_coords = [
                    current_pos.coordinates[0] + distance * math.cos(angle),
                    current_pos.coordinates[1] + distance * math.sin(angle)
                    if len(current_pos.coordinates) > 1 else 0
                ]
                
                new_pos = SwarmPosition(
                    coordinates=new_coords,
                    dimensions=environment.dimensions,
                    bounds=environment.bounds
                )
                
                if environment.is_valid_position(new_pos):
                    candidates.append(new_pos)
            
            if candidates:
                # Select position based on probability
                probabilities = []
                for candidate in candidates:
                    pos_key = tuple(candidate.coordinates)
                    pheromone = self.pheromone_matrix[pos_key]
                    heuristic = environment.get_fitness_at_position(candidate)
                    
                    probability = (pheromone ** self.alpha) * (heuristic ** self.beta)
                    probabilities.append(probability)
                
                # Roulette wheel selection
                total_prob = sum(probabilities)
                if total_prob > 0:
                    rand_val = random.uniform(0, total_prob)
                    cumulative = 0
                    
                    for i, prob in enumerate(probabilities):
                        cumulative += prob
                        if rand_val <= cumulative:
                            old_position = SwarmPosition(
                                coordinates=agent.position.coordinates,
                                dimensions=environment.dimensions
                            )
                            
                            agent.position = candidates[i]
                            agent.state['path'].append(candidates[i])
                            agent.distance_traveled += old_position.distance_to(agent.position)
                            break
    
    async def evaluate_fitness(self, agents: List[SwarmAgent], environment: SwarmEnvironment) -> None:
        """Evaluate fitness for ACO agents"""
        for agent in agents:
            fitness = environment.get_fitness_at_position(agent.position)
            agent.update_fitness(fitness)
            
            # Update best path
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_path = agent.state['path'].copy()
        
        # Update pheromones
        await self._update_pheromones(agents)
    
    async def _update_pheromones(self, agents: List[SwarmAgent]) -> None:
        """Update pheromone matrix"""
        # Evaporation
        for key in list(self.pheromone_matrix.keys()):
            self.pheromone_matrix[key] *= (1 - self.evaporation_rate)
            if self.pheromone_matrix[key] < 0.01:
                del self.pheromone_matrix[key]
        
        # Deposit pheromones
        for agent in agents:
            if agent.fitness > 0:
                deposit_amount = self.pheromone_deposit * agent.fitness
                for position in agent.state['path']:
                    pos_key = tuple(position.coordinates)
                    self.pheromone_matrix[pos_key] += deposit_amount
    
    def get_best_solution(self, agents: List[SwarmAgent]) -> Tuple[SwarmPosition, float]:
        """Get best ACO solution"""
        if self.best_path:
            return self.best_path[-1], self.best_fitness
        
        best_agent = max(agents, key=lambda a: a.fitness)
        return best_agent.position, best_agent.fitness


class BeeColonyOptimization(BaseSwarmAlgorithm):
    """Bee Colony Optimization algorithm implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.employed_ratio = config.get('employed_ratio', 0.5)
        self.scout_ratio = config.get('scout_ratio', 0.1)
        self.max_trials = config.get('max_trials', 10)
        self.food_sources: List[Dict[str, Any]] = []
    
    async def initialize_swarm(self, swarm_size: int, environment: SwarmEnvironment) -> List[SwarmAgent]:
        """Initialize BCO swarm"""
        agents = []
        employed_count = int(swarm_size * self.employed_ratio)
        scout_count = int(swarm_size * self.scout_ratio)
        onlooker_count = swarm_size - employed_count - scout_count
        
        # Initialize food sources
        self.food_sources = []
        for i in range(employed_count):
            position_coords = [
                random.uniform(environment.bounds[0][j], environment.bounds[1][j])
                for j in range(environment.dimensions)
            ]
            
            self.food_sources.append({
                'position': SwarmPosition(
                    coordinates=position_coords,
                    dimensions=environment.dimensions,
                    bounds=environment.bounds
                ),
                'fitness': 0.0,
                'trials': 0
            })
        
        # Create agents
        for i in range(swarm_size):
            if i < employed_count:
                role = SwarmRole.WORKER
            elif i < employed_count + scout_count:
                role = SwarmRole.SCOUT
            else:
                role = SwarmRole.FOLLOWER
            
            position_coords = [
                random.uniform(environment.bounds[0][j], environment.bounds[1][j])
                for j in range(environment.dimensions)
            ]
            
            agent = SwarmAgent(
                agent_id=f"bee_{i}",
                swarm_id="bco_swarm",
                role=role,
                position=SwarmPosition(
                    coordinates=position_coords,
                    dimensions=environment.dimensions,
                    bounds=environment.bounds
                ),
                state={'food_source_index': i if i < employed_count else -1}
            )
            
            agents.append(agent)
        
        return agents
    
    async def update_agents(self, agents: List[SwarmAgent], environment: SwarmEnvironment) -> None:
        """Update BCO agent positions"""
        employed_agents = [a for a in agents if a.role == SwarmRole.WORKER]
        onlooker_agents = [a for a in agents if a.role == SwarmRole.FOLLOWER]
        scout_agents = [a for a in agents if a.role == SwarmRole.SCOUT]
        
        # Employed bee phase
        for agent in employed_agents:
            food_index = agent.state.get('food_source_index', -1)
            if food_index < len(self.food_sources):
                await self._explore_food_source(agent, food_index, environment)
        
        # Onlooker bee phase
        for agent in onlooker_agents:
            # Select food source probabilistically
            food_index = self._select_food_source()
            if food_index >= 0:
                await self._explore_food_source(agent, food_index, environment)
        
        # Scout bee phase
        for agent in scout_agents:
            await self._scout_new_area(agent, environment)
        
        # Replace abandoned food sources
        await self._replace_abandoned_sources(environment)
    
    async def _explore_food_source(self, agent: SwarmAgent, food_index: int, environment: SwarmEnvironment) -> None:
        """Explore around a food source"""
        if food_index >= len(self.food_sources):
            return
        
        food_source = self.food_sources[food_index]
        current_pos = food_source['position']
        
        # Generate neighbor position
        neighbor_coords = []
        for i in range(environment.dimensions):
            phi = random.uniform(-1, 1)
            # Select random dimension to modify
            if random.random() < 0.5:
                neighbor_coords.append(
                    current_pos.coordinates[i] + phi * (current_pos.coordinates[i] - random.choice([
                        fs['position'].coordinates[i] for fs in self.food_sources if fs != food_source
                    ] or [current_pos.coordinates[i]]))
                )
            else:
                neighbor_coords.append(current_pos.coordinates[i])
        
        # Apply bounds
        lower_bounds, upper_bounds = environment.bounds
        neighbor_coords = [
            max(lower_bounds[i], min(upper_bounds[i], coord))
            for i, coord in enumerate(neighbor_coords)
        ]
        
        neighbor_pos = SwarmPosition(
            coordinates=neighbor_coords,
            dimensions=environment.dimensions,
            bounds=environment.bounds
        )
        
        # Evaluate neighbor
        neighbor_fitness = environment.get_fitness_at_position(neighbor_pos)
        
        # Greedy selection
        if neighbor_fitness > food_source['fitness']:
            old_position = SwarmPosition(
                coordinates=agent.position.coordinates,
                dimensions=environment.dimensions
            )
            
            agent.position = neighbor_pos
            food_source['position'] = neighbor_pos
            food_source['fitness'] = neighbor_fitness
            food_source['trials'] = 0
            agent.distance_traveled += old_position.distance_to(agent.position)
        else:
            food_source['trials'] += 1
    
    async def _scout_new_area(self, agent: SwarmAgent, environment: SwarmEnvironment) -> None:
        """Scout for new areas"""
        # Random exploration
        position_coords = [
            random.uniform(environment.bounds[0][j], environment.bounds[1][j])
            for j in range(environment.dimensions)
        ]
        
        old_position = SwarmPosition(
            coordinates=agent.position.coordinates,
            dimensions=environment.dimensions
        )
        
        agent.position = SwarmPosition(
            coordinates=position_coords,
            dimensions=environment.dimensions,
            bounds=environment.bounds
        )
        
        agent.distance_traveled += old_position.distance_to(agent.position)
    
    def _select_food_source(self) -> int:
        """Select food source probabilistically"""
        if not self.food_sources:
            return -1
        
        # Calculate probabilities based on fitness
        fitness_values = [max(fs['fitness'], 0.1) for fs in self.food_sources]
        total_fitness = sum(fitness_values)
        
        if total_fitness == 0:
            return random.randint(0, len(self.food_sources) - 1)
        
        # Roulette wheel selection
        rand_val = random.uniform(0, total_fitness)
        cumulative = 0
        
        for i, fitness in enumerate(fitness_values):
            cumulative += fitness
            if rand_val <= cumulative:
                return i
        
        return len(self.food_sources) - 1
    
    async def _replace_abandoned_sources(self, environment: SwarmEnvironment) -> None:
        """Replace abandoned food sources"""
        for i, food_source in enumerate(self.food_sources):
            if food_source['trials'] > self.max_trials:
                # Create new random food source
                position_coords = [
                    random.uniform(environment.bounds[0][j], environment.bounds[1][j])
                    for j in range(environment.dimensions)
                ]
                
                food_source['position'] = SwarmPosition(
                    coordinates=position_coords,
                    dimensions=environment.dimensions,
                    bounds=environment.bounds
                )
                food_source['fitness'] = environment.get_fitness_at_position(food_source['position'])
                food_source['trials'] = 0
    
    async def evaluate_fitness(self, agents: List[SwarmAgent], environment: SwarmEnvironment) -> None:
        """Evaluate fitness for BCO agents"""
        for agent in agents:
            fitness = environment.get_fitness_at_position(agent.position)
            agent.update_fitness(fitness)
        
        # Update food source fitness
        for food_source in self.food_sources:
            food_source['fitness'] = environment.get_fitness_at_position(food_source['position'])
    
    def get_best_solution(self, agents: List[SwarmAgent]) -> Tuple[SwarmPosition, float]:
        """Get best BCO solution"""
        if self.food_sources:
            best_food_source = max(self.food_sources, key=lambda fs: fs['fitness'])
            return best_food_source['position'], best_food_source['fitness']
        
        best_agent = max(agents, key=lambda a: a.fitness)
        return best_agent.position, best_agent.fitness


class SwarmIntelligenceFramework:
    """Main swarm intelligence framework"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.framework_id = f"swarm_{uuid.uuid4().hex[:8]}"
        
        # Initialize algorithms
        self.algorithms: Dict[SwarmAlgorithm, BaseSwarmAlgorithm] = {}
        self._initialize_algorithms()
        
        # Swarm management
        self.active_swarms: Dict[str, Dict[str, Any]] = {}
        self.environments: Dict[str, SwarmEnvironment] = {}
        self.objectives: Dict[str, SwarmObjective] = {}
        self.metrics = SwarmMetrics()
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.is_running = False
        
        # Global coordination
        self.global_best_solutions: Dict[str, Tuple[SwarmPosition, float]] = {}
        self.swarm_interactions: Dict[str, List[str]] = {}
    
    def _initialize_algorithms(self) -> None:
        """Initialize swarm algorithms"""
        try:
            self.algorithms[SwarmAlgorithm.PARTICLE_SWARM_OPTIMIZATION] = ParticleSwarmOptimization(
                self.config.get('pso', {})
            )
            
            self.algorithms[SwarmAlgorithm.ANT_COLONY_OPTIMIZATION] = AntColonyOptimization(
                self.config.get('aco', {})
            )
            
            self.algorithms[SwarmAlgorithm.BEE_COLONY_OPTIMIZATION] = BeeColonyOptimization(
                self.config.get('bco', {})
            )
            
            logger.info(f"Initialized {len(self.algorithms)} swarm algorithms")
            
        except Exception as e:
            logger.error(f"Algorithm initialization failed: {e}")
            raise
    
    async def start(self) -> None:
        """Start the swarm intelligence framework"""
        try:
            self.is_running = True
            
            # Start background tasks
            self.background_tasks.add(
                asyncio.create_task(self._swarm_monitor())
            )
            
            self.background_tasks.add(
                asyncio.create_task(self._environment_updater())
            )
            
            logger.info(f"Swarm intelligence framework {self.framework_id} started")
            
        except Exception as e:
            logger.error(f"Framework start failed: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the swarm intelligence framework"""
        try:
            self.is_running = False
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            self.background_tasks.clear()
            
            logger.info(f"Swarm intelligence framework {self.framework_id} stopped")
            
        except Exception as e:
            logger.error(f"Framework stop failed: {e}")
    
    async def create_swarm(self, swarm_id: str, algorithm: SwarmAlgorithm, 
                          swarm_size: int, environment_config: Dict[str, Any],
                          objectives: List[Dict[str, Any]] = None) -> bool:
        """Create a new swarm"""
        try:
            if algorithm not in self.algorithms:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Create environment
            environment = SwarmEnvironment(
                environment_id=f"env_{swarm_id}",
                dimensions=environment_config.get('dimensions', 2),
                bounds=(
                    environment_config.get('lower_bounds', [-10.0, -10.0]),
                    environment_config.get('upper_bounds', [10.0, 10.0])
                ),
                obstacles=environment_config.get('obstacles', []),
                resources=environment_config.get('resources', []),
                hazards=environment_config.get('hazards', [])
            )
            
            self.environments[swarm_id] = environment
            
            # Create objectives
            swarm_objectives = []
            for obj_config in (objectives or []):
                objective = SwarmObjective(
                    objective_id=f"obj_{uuid.uuid4().hex[:8]}",
                    name=obj_config.get('name', 'Optimization'),
                    description=obj_config.get('description', ''),
                    objective_type=obj_config.get('type', 'maximize'),
                    target_value=obj_config.get('target_value'),
                    maximization=obj_config.get('maximization', True),
                    weight=obj_config.get('weight', 1.0)
                )
                swarm_objectives.append(objective)
                self.objectives[objective.objective_id] = objective
            
            # Initialize swarm agents
            algorithm_instance = self.algorithms[algorithm]
            agents = await algorithm_instance.initialize_swarm(swarm_size, environment)
            
            # Store swarm information
            self.active_swarms[swarm_id] = {
                'algorithm': algorithm,
                'algorithm_instance': algorithm_instance,
                'agents': agents,
                'environment': environment,
                'objectives': swarm_objectives,
                'state': SwarmState.ACTIVE,
                'iteration': 0,
                'created_at': datetime.now(),
                'last_update': datetime.now()
            }
            
            self.metrics.total_agents += len(agents)
            self.metrics.active_agents += len(agents)
            
            logger.info(f"Created swarm {swarm_id} with {len(agents)} agents using {algorithm.name}")
            return True
            
        except Exception as e:
            logger.error(f"Swarm creation failed: {e}")
            return False
    
    async def run_swarm_iteration(self, swarm_id: str) -> bool:
        """Run one iteration of swarm optimization"""
        try:
            if swarm_id not in self.active_swarms:
                logger.error(f"Swarm {swarm_id} not found")
                return False
            
            swarm_info = self.active_swarms[swarm_id]
            algorithm_instance = swarm_info['algorithm_instance']
            agents = swarm_info['agents']
            environment = swarm_info['environment']
            
            # Update environment
            environment.update_environment()
            
            # Update agents
            await algorithm_instance.update_agents(agents, environment)
            
            # Evaluate fitness
            await algorithm_instance.evaluate_fitness(agents, environment)
            
            # Update swarm state
            swarm_info['iteration'] += 1
            swarm_info['last_update'] = datetime.now()
            
            # Update best solution
            best_position, best_fitness = algorithm_instance.get_best_solution(agents)
            self.global_best_solutions[swarm_id] = (best_position, best_fitness)
            
            # Update metrics
            self._update_swarm_metrics(swarm_id, agents)
            
            return True
            
        except Exception as e:
            logger.error(f"Swarm iteration failed for {swarm_id}: {e}")
            return False
    
    async def run_swarm_optimization(self, swarm_id: str, max_iterations: int = 100,
                                   convergence_threshold: float = 1e-6) -> Dict[str, Any]:
        """Run complete swarm optimization"""
        try:
            if swarm_id not in self.active_swarms:
                raise ValueError(f"Swarm {swarm_id} not found")
            
            start_time = datetime.now()
            best_fitness_history = []
            convergence_count = 0
            
            for iteration in range(max_iterations):
                success = await self.run_swarm_iteration(swarm_id)
                if not success:
                    break
                
                # Check convergence
                if swarm_id in self.global_best_solutions:
                    _, current_best_fitness = self.global_best_solutions[swarm_id]
                    best_fitness_history.append(current_best_fitness)
                    
                    if len(best_fitness_history) > 10:
                        recent_improvement = (
                            best_fitness_history[-1] - best_fitness_history[-11]
                        )
                        
                        if abs(recent_improvement) < convergence_threshold:
                            convergence_count += 1
                            if convergence_count >= 5:
                                logger.info(f"Swarm {swarm_id} converged after {iteration + 1} iterations")
                                break
                        else:
                            convergence_count = 0
                
                # Small delay for cooperative multitasking
                await asyncio.sleep(0.001)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Get final results
            final_position, final_fitness = self.global_best_solutions.get(
                swarm_id, (None, float('-inf'))
            )
            
            results = {
                'swarm_id': swarm_id,
                'iterations': iteration + 1,
                'execution_time': execution_time,
                'best_position': final_position.coordinates if final_position else None,
                'best_fitness': final_fitness,
                'fitness_history': best_fitness_history,
                'converged': convergence_count >= 5,
                'final_metrics': self.get_swarm_metrics(swarm_id)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Swarm optimization failed for {swarm_id}: {e}")
            raise
    
    def _update_swarm_metrics(self, swarm_id: str, agents: List[SwarmAgent]) -> None:
        """Update swarm metrics"""
        if not agents:
            return
        
        # Basic metrics
        fitness_values = [agent.fitness for agent in agents]
        self.metrics.best_fitness = max(self.metrics.best_fitness, max(fitness_values))
        self.metrics.average_fitness = sum(fitness_values) / len(fitness_values)
        
        # Diversity measure (average pairwise distance)
        if len(agents) > 1:
            total_distance = 0
            pairs = 0
            
            for i, agent1 in enumerate(agents):
                for j, agent2 in enumerate(agents[i+1:], i+1):
                    total_distance += agent1.position.distance_to(agent2.position)
                    pairs += 1
            
            self.metrics.diversity_measure = total_distance / pairs if pairs > 0 else 0
        
        # Energy consumption
        total_energy = sum(100.0 - agent.energy for agent in agents)
        self.metrics.energy_consumption = total_energy
        
        # Task completion rate
        total_tasks = sum(agent.tasks_completed for agent in agents)
        self.metrics.task_completion_rate = total_tasks / len(agents) if agents else 0
        
        self.metrics.last_updated = datetime.now()
    
    def get_swarm_status(self, swarm_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a swarm"""
        if swarm_id not in self.active_swarms:
            return None
        
        swarm_info = self.active_swarms[swarm_id]
        best_solution = self.global_best_solutions.get(swarm_id)
        
        return {
            'swarm_id': swarm_id,
            'algorithm': swarm_info['algorithm'].name,
            'agent_count': len(swarm_info['agents']),
            'state': swarm_info['state'].name,
            'iteration': swarm_info['iteration'],
            'best_position': best_solution[0].coordinates if best_solution else None,
            'best_fitness': best_solution[1] if best_solution else None,
            'created_at': swarm_info['created_at'],
            'last_update': swarm_info['last_update']
        }
    
    def get_swarm_metrics(self, swarm_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific swarm"""
        if swarm_id not in self.active_swarms:
            return None
        
        swarm_info = self.active_swarms[swarm_id]
        agents = swarm_info['agents']
        
        # Calculate swarm-specific metrics
        fitness_values = [agent.fitness for agent in agents]
        distances_traveled = [agent.distance_traveled for agent in agents]
        
        return {
            'swarm_id': swarm_id,
            'agent_count': len(agents),
            'best_fitness': max(fitness_values) if fitness_values else 0,
            'average_fitness': sum(fitness_values) / len(fitness_values) if fitness_values else 0,
            'worst_fitness': min(fitness_values) if fitness_values else 0,
            'total_distance_traveled': sum(distances_traveled),
            'average_distance_per_agent': sum(distances_traveled) / len(distances_traveled) if distances_traveled else 0,
            'iteration': swarm_info['iteration'],
            'algorithm': swarm_info['algorithm'].name
        }
    
    def get_global_metrics(self) -> SwarmMetrics:
        """Get global framework metrics"""
        return self.metrics
    
    def list_active_swarms(self) -> List[Dict[str, Any]]:
        """List all active swarms"""
        return [
            {
                'swarm_id': swarm_id,
                'algorithm': info['algorithm'].name,
                'agent_count': len(info['agents']),
                'state': info['state'].name,
                'iteration': info['iteration']
            }
            for swarm_id, info in self.active_swarms.items()
        ]
    
    async def _swarm_monitor(self) -> None:
        """Background task for monitoring swarms"""
        while self.is_running:
            try:
                # Update global metrics
                active_count = 0
                for swarm_info in self.active_swarms.values():
                    if swarm_info['state'] == SwarmState.ACTIVE:
                        active_count += len(swarm_info['agents'])
                
                self.metrics.active_agents = active_count
                
                # Check for inactive swarms
                current_time = datetime.now()
                for swarm_id, swarm_info in list(self.active_swarms.items()):
                    time_since_update = (current_time - swarm_info['last_update']).total_seconds()
                    
                    if time_since_update > 300:  # 5 minutes
                        logger.warning(f"Swarm {swarm_id} inactive for {time_since_update}s")
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Swarm monitoring failed: {e}")
                await asyncio.sleep(30)
    
    async def _environment_updater(self) -> None:
        """Background task for updating environments"""
        while self.is_running:
            try:
                # Update all environments
                for environment in self.environments.values():
                    environment.update_environment()
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Environment update failed: {e}")
                await asyncio.sleep(1)


async def example_swarm_intelligence_usage():
    """Comprehensive example of swarm intelligence framework usage"""
    
    print("\n🐝 Swarm Intelligence Framework Example")
    print("=" * 60)
    
    # Configuration
    config = {
        'pso': {
            'inertia_weight': 0.9,
            'cognitive_factor': 2.0,
            'social_factor': 2.0,
            'max_velocity': 2.0
        },
        'aco': {
            'alpha': 1.0,
            'beta': 2.0,
            'evaporation_rate': 0.1,
            'pheromone_deposit': 1.0
        },
        'bco': {
            'employed_ratio': 0.5,
            'scout_ratio': 0.1,
            'max_trials': 10
        }
    }
    
    # Initialize framework
    swarm_framework = SwarmIntelligenceFramework(config)
    await swarm_framework.start()
    
    print(f"✅ Swarm framework {swarm_framework.framework_id} started")
    
    try:
        # Example 1: Particle Swarm Optimization
        print("\n1. Particle Swarm Optimization")
        print("-" * 40)
        
        environment_config = {
            'dimensions': 2,
            'lower_bounds': [-5.0, -5.0],
            'upper_bounds': [5.0, 5.0],
            'obstacles': [
                {'position': [0, 0], 'radius': 0.5}
            ],
            'resources': [
                {'position': [2, 2], 'value': 10},
                {'position': [-2, -2], 'value': 8}
            ]
        }
        
        objectives = [
            {
                'name': 'Maximize Fitness',
                'type': 'maximize',
                'maximization': True,
                'weight': 1.0
            }
        ]
        
        success = await swarm_framework.create_swarm(
            swarm_id="pso_swarm_1",
            algorithm=SwarmAlgorithm.PARTICLE_SWARM_OPTIMIZATION,
            swarm_size=30,
            environment_config=environment_config,
            objectives=objectives
        )
        
        print(f"✅ PSO swarm created: {success}")
        
        # Run optimization
        pso_results = await swarm_framework.run_swarm_optimization(
            swarm_id="pso_swarm_1",
            max_iterations=50,
            convergence_threshold=1e-4
        )
        
        print(f"✅ PSO optimization completed:")
        print(f"   Iterations: {pso_results['iterations']}")
        print(f"   Best position: {pso_results['best_position']}")
        print(f"   Best fitness: {pso_results['best_fitness']:.4f}")
        print(f"   Execution time: {pso_results['execution_time']:.2f}s")
        print(f"   Converged: {pso_results['converged']}")
        
        # Example 2: Ant Colony Optimization
        print("\n2. Ant Colony Optimization")
        print("-" * 40)
        
        path_environment = {
            'dimensions': 2,
            'lower_bounds': [-3.0, -3.0],
            'upper_bounds': [3.0, 3.0],
            'obstacles': [
                {'position': [1, 0], 'radius': 0.3},
                {'position': [-1, 0], 'radius': 0.3}
            ]
        }
        
        success = await swarm_framework.create_swarm(
            swarm_id="aco_swarm_1",
            algorithm=SwarmAlgorithm.ANT_COLONY_OPTIMIZATION,
            swarm_size=25,
            environment_config=path_environment
        )
        
        print(f"✅ ACO swarm created: {success}")
        
        # Run optimization
        aco_results = await swarm_framework.run_swarm_optimization(
            swarm_id="aco_swarm_1",
            max_iterations=40
        )
        
        print(f"✅ ACO optimization completed:")
        print(f"   Iterations: {aco_results['iterations']}")
        print(f"   Best position: {aco_results['best_position']}")
        print(f"   Best fitness: {aco_results['best_fitness']:.4f}")
        print(f"   Execution time: {aco_results['execution_time']:.2f}s")
        
        # Example 3: Bee Colony Optimization
        print("\n3. Bee Colony Optimization")
        print("-" * 40)
        
        foraging_environment = {
            'dimensions': 2,
            'lower_bounds': [-4.0, -4.0],
            'upper_bounds': [4.0, 4.0],
            'resources': [
                {'position': [2.5, 2.5], 'value': 15},
                {'position': [-2.5, -2.5], 'value': 12},
                {'position': [0, 3], 'value': 8}
            ]
        }
        
        success = await swarm_framework.create_swarm(
            swarm_id="bco_swarm_1",
            algorithm=SwarmAlgorithm.BEE_COLONY_OPTIMIZATION,
            swarm_size=40,
            environment_config=foraging_environment
        )
        
        print(f"✅ BCO swarm created: {success}")
        
        # Run optimization
        bco_results = await swarm_framework.run_swarm_optimization(
            swarm_id="bco_swarm_1",
            max_iterations=35
        )
        
        print(f"✅ BCO optimization completed:")
        print(f"   Iterations: {bco_results['iterations']}")
        print(f"   Best position: {bco_results['best_position']}")
        print(f"   Best fitness: {bco_results['best_fitness']:.4f}")
        print(f"   Execution time: {bco_results['execution_time']:.2f}s")
        
        # Example 4: Swarm Status and Metrics
        print("\n4. Swarm Status and Metrics")
        print("-" * 40)
        
        active_swarms = swarm_framework.list_active_swarms()
        print(f"✅ Active swarms: {len(active_swarms)}")
        
        for swarm_info in active_swarms:
            print(f"   - {swarm_info['swarm_id']}: {swarm_info['algorithm']}")
            print(f"     Agents: {swarm_info['agent_count']}")
            print(f"     State: {swarm_info['state']}")
            print(f"     Iterations: {swarm_info['iteration']}")
            
            # Get detailed metrics
            metrics = swarm_framework.get_swarm_metrics(swarm_info['swarm_id'])
            if metrics:
                print(f"     Best fitness: {metrics['best_fitness']:.4f}")
                print(f"     Average fitness: {metrics['average_fitness']:.4f}")
                print(f"     Total distance: {metrics['total_distance_traveled']:.2f}")
        
        # Example 5: Global Metrics
        print("\n5. Global Framework Metrics")
        print("-" * 40)
        
        global_metrics = swarm_framework.get_global_metrics()
        print(f"✅ Total agents: {global_metrics.total_agents}")
        print(f"✅ Active agents: {global_metrics.active_agents}")
        print(f"✅ Best fitness achieved: {global_metrics.best_fitness:.4f}")
        print(f"✅ Average fitness: {global_metrics.average_fitness:.4f}")
        print(f"✅ Diversity measure: {global_metrics.diversity_measure:.4f}")
        print(f"✅ Energy consumption: {global_metrics.energy_consumption:.2f}")
        print(f"✅ Task completion rate: {global_metrics.task_completion_rate:.2f}")
        
        # Example 6: Algorithm Comparison
        print("\n6. Algorithm Performance Comparison")
        print("-" * 40)
        
        results_comparison = [
            ("PSO", pso_results),
            ("ACO", aco_results),
            ("BCO", bco_results)
        ]
        
        print("Algorithm | Best Fitness | Iterations | Time (s) | Converged")
        print("-" * 55)
        
        for name, results in results_comparison:
            print(f"{name:8} | {results['best_fitness']:11.4f} | {results['iterations']:10} | {results['execution_time']:7.2f} | {results.get('converged', False):9}")
        
        # Allow background tasks to run briefly
        await asyncio.sleep(2)
        
    finally:
        # Cleanup
        await swarm_framework.stop()
        print(f"\n✅ Swarm intelligence framework stopped successfully")


if __name__ == "__main__":
    asyncio.run(example_swarm_intelligence_usage())