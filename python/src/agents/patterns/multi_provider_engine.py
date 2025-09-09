"""
Multi-Provider Abstraction Engine

Enables patterns to be deployed across different cloud providers
with provider-specific optimizations and configurations.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .pattern_models import (
    Pattern, PatternProvider, PatternTechnology, PatternComponent,
    MultiProviderConfig
)
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProviderResource:
    """Represents a cloud provider resource."""
    
    name: str
    type: str  # e.g., 'compute', 'storage', 'database', 'network'
    provider_specific_name: str
    configuration: Dict[str, Any]
    dependencies: List[str] = None
    cost_estimate: Optional[float] = None  # Monthly USD
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class DeploymentPlan:
    """Deployment plan for a specific provider."""
    
    provider: PatternProvider
    resources: List[ProviderResource]
    deployment_scripts: List[str]
    configuration_files: Dict[str, str]  # filename -> content
    estimated_cost: float
    deployment_time_estimate: int  # minutes
    prerequisites: List[str]
    post_deployment_steps: List[str]


class BaseProviderAdapter(ABC):
    """Base class for provider-specific adapters."""
    
    def __init__(self, provider: PatternProvider):
        self.provider = provider
    
    @abstractmethod
    async def translate_component(self, component: PatternComponent) -> List[ProviderResource]:
        """Translate a pattern component to provider-specific resources."""
        pass
    
    @abstractmethod
    async def generate_deployment_scripts(self, resources: List[ProviderResource]) -> List[str]:
        """Generate deployment scripts for the provider."""
        pass
    
    @abstractmethod
    async def estimate_costs(self, resources: List[ProviderResource]) -> float:
        """Estimate monthly costs for the resources."""
        pass
    
    @abstractmethod
    async def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """Validate provider-specific configuration. Returns list of errors."""
        pass


class AWSAdapter(BaseProviderAdapter):
    """AWS-specific provider adapter."""
    
    # AWS service mappings
    SERVICE_MAPPINGS = {
        'compute': {
            'container': 'ecs',
            'serverless': 'lambda',
            'vm': 'ec2'
        },
        'database': {
            'postgresql': 'rds-postgresql',
            'mysql': 'rds-mysql',
            'mongodb': 'documentdb',
            'redis': 'elasticache-redis'
        },
        'storage': {
            'object': 's3',
            'file': 'efs',
            'block': 'ebs'
        },
        'network': {
            'loadbalancer': 'alb',
            'cdn': 'cloudfront',
            'dns': 'route53'
        }
    }
    
    # Cost estimates (monthly USD)
    COST_ESTIMATES = {
        'ecs': 50.0,  # Fargate container
        'lambda': 10.0,  # Based on typical usage
        'ec2': 25.0,  # t3.small
        'rds-postgresql': 30.0,  # db.t3.micro
        'rds-mysql': 30.0,
        'documentdb': 200.0,  # Minimum cluster
        'elasticache-redis': 45.0,
        's3': 5.0,
        'efs': 10.0,
        'alb': 20.0,
        'cloudfront': 15.0
    }
    
    def __init__(self):
        super().__init__(PatternProvider.AWS)
    
    async def translate_component(self, component: PatternComponent) -> List[ProviderResource]:
        """Translate component to AWS resources."""
        resources = []
        
        # Map component type to AWS services
        if component.type == 'frontend':
            resources.extend(await self._create_frontend_resources(component))
        elif component.type == 'backend':
            resources.extend(await self._create_backend_resources(component))
        elif component.type == 'database':
            resources.extend(await self._create_database_resources(component))
        elif component.type == 'storage':
            resources.extend(await self._create_storage_resources(component))
        
        return resources
    
    async def _create_frontend_resources(self, component: PatternComponent) -> List[ProviderResource]:
        """Create AWS resources for frontend components."""
        resources = []
        
        # S3 bucket for static hosting
        resources.append(ProviderResource(
            name=f"{component.name}-static-hosting",
            type='storage',
            provider_specific_name='s3',
            configuration={
                'bucket_name': f"{component.name}-static-{self.provider.value}",
                'website_configuration': True,
                'public_read_access': True
            },
            cost_estimate=self.COST_ESTIMATES.get('s3', 5.0)
        ))
        
        # CloudFront distribution
        resources.append(ProviderResource(
            name=f"{component.name}-cdn",
            type='network',
            provider_specific_name='cloudfront',
            configuration={
                'origin_domain': f"{component.name}-static-{self.provider.value}.s3-website-us-east-1.amazonaws.com",
                'cache_behavior': 'default',
                'price_class': 'PriceClass_100'
            },
            dependencies=[f"{component.name}-static-hosting"],
            cost_estimate=self.COST_ESTIMATES.get('cloudfront', 15.0)
        ))
        
        return resources
    
    async def _create_backend_resources(self, component: PatternComponent) -> List[ProviderResource]:
        """Create AWS resources for backend components."""
        resources = []
        
        # Check if component uses containers
        uses_containers = any(
            tech.name.lower() == 'docker' 
            for tech in component.technologies
        )
        
        if uses_containers:
            # ECS Fargate service
            resources.append(ProviderResource(
                name=f"{component.name}-service",
                type='compute',
                provider_specific_name='ecs',
                configuration={
                    'launch_type': 'FARGATE',
                    'cpu': '256',
                    'memory': '512',
                    'desired_count': 1,
                    'container_image': f"{component.name}:latest"
                },
                cost_estimate=self.COST_ESTIMATES.get('ecs', 50.0)
            ))
            
            # Application Load Balancer
            resources.append(ProviderResource(
                name=f"{component.name}-alb",
                type='network',
                provider_specific_name='alb',
                configuration={
                    'scheme': 'internet-facing',
                    'type': 'application',
                    'target_type': 'ip'
                },
                dependencies=[f"{component.name}-service"],
                cost_estimate=self.COST_ESTIMATES.get('alb', 20.0)
            ))
        else:
            # Lambda function for serverless
            resources.append(ProviderResource(
                name=f"{component.name}-function",
                type='compute',
                provider_specific_name='lambda',
                configuration={
                    'runtime': self._determine_runtime(component.technologies),
                    'memory': 256,
                    'timeout': 30,
                    'environment_variables': {}
                },
                cost_estimate=self.COST_ESTIMATES.get('lambda', 10.0)
            ))
        
        return resources
    
    async def _create_database_resources(self, component: PatternComponent) -> List[ProviderResource]:
        """Create AWS database resources."""
        resources = []
        
        # Determine database type from technologies
        db_tech = next(
            (tech for tech in component.technologies 
             if tech.name.lower() in ['postgresql', 'mysql', 'mongodb', 'redis']),
            None
        )
        
        if not db_tech:
            return resources
        
        db_name = db_tech.name.lower()
        
        if db_name in ['postgresql', 'mysql']:
            # RDS instance
            service_name = f'rds-{db_name}'
            resources.append(ProviderResource(
                name=f"{component.name}-db",
                type='database',
                provider_specific_name=service_name,
                configuration={
                    'engine': db_name,
                    'instance_class': 'db.t3.micro',
                    'allocated_storage': 20,
                    'multi_az': False,
                    'backup_retention': 7
                },
                cost_estimate=self.COST_ESTIMATES.get(service_name, 30.0)
            ))
        elif db_name == 'mongodb':
            # DocumentDB cluster
            resources.append(ProviderResource(
                name=f"{component.name}-docdb",
                type='database',
                provider_specific_name='documentdb',
                configuration={
                    'engine': 'docdb',
                    'instance_class': 'db.t3.medium',
                    'instance_count': 1,
                    'backup_retention': 7
                },
                cost_estimate=self.COST_ESTIMATES.get('documentdb', 200.0)
            ))
        elif db_name == 'redis':
            # ElastiCache Redis
            resources.append(ProviderResource(
                name=f"{component.name}-redis",
                type='database',
                provider_specific_name='elasticache-redis',
                configuration={
                    'engine': 'redis',
                    'node_type': 'cache.t3.micro',
                    'num_cache_nodes': 1,
                    'port': 6379
                },
                cost_estimate=self.COST_ESTIMATES.get('elasticache-redis', 45.0)
            ))
        
        return resources
    
    async def _create_storage_resources(self, component: PatternComponent) -> List[ProviderResource]:
        """Create AWS storage resources."""
        resources = []
        
        # S3 bucket for object storage
        resources.append(ProviderResource(
            name=f"{component.name}-storage",
            type='storage',
            provider_specific_name='s3',
            configuration={
                'bucket_name': f"{component.name}-storage-{self.provider.value}",
                'versioning': True,
                'lifecycle_policy': True
            },
            cost_estimate=self.COST_ESTIMATES.get('s3', 5.0)
        ))
        
        return resources
    
    def _determine_runtime(self, technologies: List[PatternTechnology]) -> str:
        """Determine Lambda runtime from technologies."""
        runtime_map = {
            'python': 'python3.11',
            'nodejs': 'nodejs18.x',
            'java': 'java11',
            'dotnet': 'dotnet6',
            'go': 'go1.x',
            'ruby': 'ruby3.2'
        }
        
        for tech in technologies:
            if tech.name.lower() in runtime_map:
                return runtime_map[tech.name.lower()]
        
        return 'python3.11'  # Default
    
    async def generate_deployment_scripts(self, resources: List[ProviderResource]) -> List[str]:
        """Generate AWS deployment scripts (CloudFormation/CDK)."""
        scripts = []
        
        # Generate CloudFormation template
        cf_template = await self._generate_cloudformation_template(resources)
        scripts.append(f"# CloudFormation Template\n{cf_template}")
        
        # Generate deployment script
        deploy_script = await self._generate_deploy_script()
        scripts.append(f"# Deployment Script\n{deploy_script}")
        
        return scripts
    
    async def _generate_cloudformation_template(self, resources: List[ProviderResource]) -> str:
        """Generate CloudFormation template."""
        template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": "Auto-generated CloudFormation template",
            "Resources": {}
        }
        
        for resource in resources:
            cf_resource = await self._resource_to_cloudformation(resource)
            template["Resources"][resource.name.replace("-", "")] = cf_resource
        
        return json.dumps(template, indent=2)
    
    async def _resource_to_cloudformation(self, resource: ProviderResource) -> Dict[str, Any]:
        """Convert resource to CloudFormation format."""
        cf_type_map = {
            's3': 'AWS::S3::Bucket',
            'ecs': 'AWS::ECS::Service',
            'lambda': 'AWS::Lambda::Function',
            'rds-postgresql': 'AWS::RDS::DBInstance',
            'rds-mysql': 'AWS::RDS::DBInstance',
            'alb': 'AWS::ElasticLoadBalancingV2::LoadBalancer',
            'cloudfront': 'AWS::CloudFront::Distribution'
        }
        
        cf_type = cf_type_map.get(resource.provider_specific_name, 'AWS::CloudFormation::CustomResource')
        
        return {
            "Type": cf_type,
            "Properties": resource.configuration
        }
    
    async def _generate_deploy_script(self) -> str:
        """Generate deployment script."""
        return """#!/bin/bash

# Deploy AWS resources
aws cloudformation deploy \\
  --template-file template.json \\
  --stack-name archon-pattern-stack \\
  --capabilities CAPABILITY_IAM \\
  --region us-east-1

echo "Deployment complete!"
"""
    
    async def estimate_costs(self, resources: List[ProviderResource]) -> float:
        """Estimate monthly costs for AWS resources."""
        total_cost = sum(
            resource.cost_estimate or self.COST_ESTIMATES.get(resource.provider_specific_name, 0.0)
            for resource in resources
        )
        return total_cost
    
    async def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """Validate AWS-specific configuration."""
        errors = []
        
        # Check for required AWS fields
        required_fields = ['region']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required AWS configuration: {field}")
        
        # Validate region
        if 'region' in config:
            valid_regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1']
            if config['region'] not in valid_regions:
                errors.append(f"Invalid AWS region: {config['region']}")
        
        return errors


class GCPAdapter(BaseProviderAdapter):
    """Google Cloud Platform adapter."""
    
    SERVICE_MAPPINGS = {
        'compute': {
            'container': 'cloud-run',
            'serverless': 'cloud-functions',
            'vm': 'compute-engine'
        },
        'database': {
            'postgresql': 'cloud-sql-postgresql',
            'mysql': 'cloud-sql-mysql',
            'mongodb': 'mongodb-atlas',
            'redis': 'memorystore-redis'
        }
    }
    
    COST_ESTIMATES = {
        'cloud-run': 25.0,
        'cloud-functions': 8.0,
        'compute-engine': 30.0,
        'cloud-sql-postgresql': 35.0,
        'cloud-sql-mysql': 35.0,
        'memorystore-redis': 40.0,
        'cloud-storage': 3.0
    }
    
    def __init__(self):
        super().__init__(PatternProvider.GCP)
    
    async def translate_component(self, component: PatternComponent) -> List[ProviderResource]:
        """Translate component to GCP resources."""
        # Simplified GCP implementation
        resources = []
        
        if component.type == 'backend':
            resources.append(ProviderResource(
                name=f"{component.name}-service",
                type='compute',
                provider_specific_name='cloud-run',
                configuration={
                    'service_name': component.name,
                    'image': f"gcr.io/PROJECT_ID/{component.name}",
                    'port': 8080,
                    'cpu': '1',
                    'memory': '512Mi'
                },
                cost_estimate=self.COST_ESTIMATES.get('cloud-run', 25.0)
            ))
        
        return resources
    
    async def generate_deployment_scripts(self, resources: List[ProviderResource]) -> List[str]:
        """Generate GCP deployment scripts."""
        return ["# GCP deployment script placeholder"]
    
    async def estimate_costs(self, resources: List[ProviderResource]) -> float:
        """Estimate GCP costs."""
        return sum(
            resource.cost_estimate or self.COST_ESTIMATES.get(resource.provider_specific_name, 0.0)
            for resource in resources
        )
    
    async def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """Validate GCP configuration."""
        errors = []
        required_fields = ['project_id', 'region']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required GCP configuration: {field}")
        return errors


class AzureAdapter(BaseProviderAdapter):
    """Microsoft Azure adapter."""
    
    def __init__(self):
        super().__init__(PatternProvider.AZURE)
    
    async def translate_component(self, component: PatternComponent) -> List[ProviderResource]:
        """Translate component to Azure resources."""
        # Placeholder implementation
        return []
    
    async def generate_deployment_scripts(self, resources: List[ProviderResource]) -> List[str]:
        """Generate Azure deployment scripts."""
        return ["# Azure deployment script placeholder"]
    
    async def estimate_costs(self, resources: List[ProviderResource]) -> float:
        """Estimate Azure costs."""
        return 50.0  # Placeholder
    
    async def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """Validate Azure configuration."""
        return []


class MultiProviderEngine:
    """Main engine for multi-provider pattern deployment."""
    
    def __init__(self):
        self.adapters = {
            PatternProvider.AWS: AWSAdapter(),
            PatternProvider.GCP: GCPAdapter(),
            PatternProvider.AZURE: AzureAdapter()
        }
    
    async def generate_deployment_plans(
        self, 
        pattern: Pattern,
        target_providers: List[PatternProvider],
        provider_configs: Optional[Dict[PatternProvider, Dict[str, Any]]] = None
    ) -> Dict[PatternProvider, DeploymentPlan]:
        """
        Generate deployment plans for multiple providers.
        
        Args:
            pattern: Pattern to deploy
            target_providers: List of providers to target
            provider_configs: Provider-specific configurations
        
        Returns:
            Dict mapping providers to deployment plans
        """
        if provider_configs is None:
            provider_configs = {}
        
        deployment_plans = {}
        
        for provider in target_providers:
            if provider not in self.adapters:
                logger.warning(f"No adapter available for provider: {provider}")
                continue
            
            try:
                logger.info(f"Generating deployment plan | pattern={pattern.id} | provider={provider.value}")
                
                adapter = self.adapters[provider]
                
                # Validate provider configuration
                config = provider_configs.get(provider, {})
                config_errors = await adapter.validate_configuration(config)
                
                if config_errors:
                    logger.warning(f"Configuration errors for {provider.value}: {config_errors}")
                    continue
                
                # Translate components to provider resources
                all_resources = []
                for component in pattern.components:
                    resources = await adapter.translate_component(component)
                    all_resources.extend(resources)
                
                # Generate deployment scripts
                deployment_scripts = await adapter.generate_deployment_scripts(all_resources)
                
                # Estimate costs
                estimated_cost = await adapter.estimate_costs(all_resources)
                
                # Create deployment plan
                deployment_plan = DeploymentPlan(
                    provider=provider,
                    resources=all_resources,
                    deployment_scripts=deployment_scripts,
                    configuration_files=await self._generate_config_files(pattern, provider, config),
                    estimated_cost=estimated_cost,
                    deployment_time_estimate=self._estimate_deployment_time(all_resources),
                    prerequisites=self._get_provider_prerequisites(provider),
                    post_deployment_steps=self._get_post_deployment_steps(pattern, provider)
                )
                
                deployment_plans[provider] = deployment_plan
                
                logger.info(f"Deployment plan generated | provider={provider.value} | resources={len(all_resources)} | cost=${estimated_cost:.2f}")
                
            except Exception as e:
                logger.error(f"Failed to generate deployment plan | provider={provider.value} | error={str(e)}")
        
        return deployment_plans
    
    async def _generate_config_files(
        self, 
        pattern: Pattern, 
        provider: PatternProvider, 
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate provider-specific configuration files."""
        config_files = {}
        
        if provider == PatternProvider.AWS:
            # Generate AWS-specific configs
            config_files['aws-config.json'] = json.dumps({
                'region': config.get('region', 'us-east-1'),
                'profile': config.get('profile', 'default')
            }, indent=2)
            
        elif provider == PatternProvider.GCP:
            # Generate GCP-specific configs
            config_files['gcp-config.json'] = json.dumps({
                'project_id': config.get('project_id', 'my-project'),
                'region': config.get('region', 'us-central1')
            }, indent=2)
        
        # Common configuration
        config_files['pattern-config.json'] = json.dumps({
            'pattern_id': pattern.id,
            'pattern_name': pattern.metadata.name,
            'provider': provider.value,
            'generated_at': str(asyncio.get_event_loop().time())
        }, indent=2)
        
        return config_files
    
    def _estimate_deployment_time(self, resources: List[ProviderResource]) -> int:
        """Estimate deployment time in minutes."""
        # Base time per resource type
        time_estimates = {
            'compute': 5,
            'database': 10,
            'storage': 2,
            'network': 3
        }
        
        total_time = sum(
            time_estimates.get(resource.type, 5)
            for resource in resources
        )
        
        return max(total_time, 5)  # Minimum 5 minutes
    
    def _get_provider_prerequisites(self, provider: PatternProvider) -> List[str]:
        """Get prerequisites for a provider."""
        prerequisites_map = {
            PatternProvider.AWS: [
                "AWS CLI installed and configured",
                "Valid AWS credentials",
                "CloudFormation permissions"
            ],
            PatternProvider.GCP: [
                "gcloud CLI installed and configured",
                "Valid GCP project with billing enabled",
                "Required APIs enabled"
            ],
            PatternProvider.AZURE: [
                "Azure CLI installed and configured",
                "Valid Azure subscription",
                "Resource group created"
            ]
        }
        
        return prerequisites_map.get(provider, ["Provider-specific setup required"])
    
    def _get_post_deployment_steps(self, pattern: Pattern, provider: PatternProvider) -> List[str]:
        """Get post-deployment steps for a pattern and provider."""
        steps = [
            "Verify all resources are running",
            "Test application endpoints",
            "Configure monitoring and alerting",
            "Set up backup and recovery procedures"
        ]
        
        # Add provider-specific steps
        if provider == PatternProvider.AWS:
            steps.extend([
                "Review CloudWatch logs",
                "Configure AWS Config rules",
                "Set up CloudTrail logging"
            ])
        elif provider == PatternProvider.GCP:
            steps.extend([
                "Review Cloud Logging",
                "Configure Cloud Monitoring",
                "Set up Cloud Security Command Center"
            ])
        
        return steps
    
    async def compare_provider_costs(
        self, 
        pattern: Pattern,
        providers: List[PatternProvider]
    ) -> Dict[PatternProvider, float]:
        """Compare estimated costs across providers."""
        cost_comparison = {}
        
        for provider in providers:
            if provider in self.adapters:
                adapter = self.adapters[provider]
                
                # Get resources for pattern
                all_resources = []
                for component in pattern.components:
                    resources = await adapter.translate_component(component)
                    all_resources.extend(resources)
                
                # Estimate cost
                cost = await adapter.estimate_costs(all_resources)
                cost_comparison[provider] = cost
        
        return cost_comparison
    
    async def get_provider_recommendations(
        self, 
        pattern: Pattern,
        requirements: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get provider recommendations based on pattern and requirements."""
        if requirements is None:
            requirements = {}
        
        recommendations = []
        
        # Get cost comparison
        all_providers = list(self.adapters.keys())
        cost_comparison = await self.compare_provider_costs(pattern, all_providers)
        
        for provider, cost in cost_comparison.items():
            score = 0.0
            reasons = []
            
            # Cost factor
            if cost <= requirements.get('max_monthly_cost', float('inf')):
                score += 0.3
                reasons.append(f"Cost-effective: ${cost:.2f}/month")
            
            # Provider-specific factors
            if provider == PatternProvider.AWS:
                score += 0.4  # Most mature ecosystem
                reasons.append("Mature ecosystem with extensive services")
            elif provider == PatternProvider.GCP:
                score += 0.3  # Good for ML/AI workloads
                reasons.append("Excellent for ML/AI and analytics workloads")
            elif provider == PatternProvider.AZURE:
                score += 0.2  # Good for enterprise/Microsoft shops
                reasons.append("Great integration with Microsoft ecosystem")
            
            # Technology compatibility
            supported_technologies = self._get_supported_technologies(provider)
            pattern_technologies = {tech.name.lower() for tech in pattern.metadata.technologies}
            
            compatibility = len(pattern_technologies.intersection(supported_technologies)) / len(pattern_technologies)
            score += compatibility * 0.3
            
            if compatibility > 0.8:
                reasons.append("High technology compatibility")
            
            recommendations.append({
                'provider': provider,
                'score': score,
                'estimated_cost': cost,
                'reasons': reasons,
                'recommendation': 'highly_recommended' if score >= 0.8 else 'recommended' if score >= 0.6 else 'suitable'
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations
    
    def _get_supported_technologies(self, provider: PatternProvider) -> Set[str]:
        """Get technologies well-supported by a provider."""
        support_map = {
            PatternProvider.AWS: {
                'python', 'nodejs', 'java', 'dotnet', 'go', 'ruby',
                'docker', 'kubernetes', 'postgresql', 'mysql', 'redis',
                'react', 'angular', 'vue'
            },
            PatternProvider.GCP: {
                'python', 'nodejs', 'java', 'go', 'php',
                'docker', 'kubernetes', 'postgresql', 'mysql',
                'tensorflow', 'pytorch', 'react'
            },
            PatternProvider.AZURE: {
                'dotnet', 'python', 'nodejs', 'java',
                'docker', 'kubernetes', 'mssql', 'postgresql',
                'react', 'angular'
            }
        }
        
        return support_map.get(provider, set())