"""
Security API Routes
Comprehensive API for all security components
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import logging

from ...agents.security.security_framework import SecurityFramework
from ...agents.security.zero_trust_model import ZeroTrustModel, TrustLevel, EntityType
from ...agents.security.threat_detection import ThreatDetectionSystem, ThreatType, ThreatSeverity
from ...agents.security.encryption_service import EncryptionService, EncryptionAlgorithm
from ...agents.security.audit_logger import AuditLogger, AuditLevel, AuditCategory, ComplianceFramework
from ...agents.security.access_control import AccessControlManager, PermissionType, AccessDecision

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/security", tags=["security"])
security = HTTPBearer(auto_error=False)

# Initialize security components
security_framework = SecurityFramework()
zero_trust_model = ZeroTrustModel()
threat_detection = ThreatDetectionSystem()
encryption_service = EncryptionService()
audit_logger = AuditLogger()
access_control = AccessControlManager()


# Pydantic models for request/response
class AuthenticationRequest(BaseModel):
    username: str
    credentials: Dict[str, Any]
    ip_address: str
    user_agent: str


class AuthenticationResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    token: Optional[str] = None
    expires_at: Optional[datetime] = None
    message: str


class AuthorizationRequest(BaseModel):
    session_id: str
    resource: str
    action: str
    policy_id: Optional[str] = "user_policy"


class AuthorizationResponse(BaseModel):
    authorized: bool
    reason: str
    expires_at: Optional[datetime] = None


class ThreatAnalysisRequest(BaseModel):
    event_data: Dict[str, Any]
    source_type: str = "api"


class ThreatAnalysisResponse(BaseModel):
    threat_detected: bool
    threat_type: Optional[str] = None
    severity: Optional[str] = None
    confidence: float = 0.0
    threat_id: Optional[str] = None
    recommended_actions: List[str] = []


class EncryptionRequest(BaseModel):
    data: str
    key_id: Optional[str] = None
    algorithm: Optional[str] = None


class EncryptionResponse(BaseModel):
    success: bool
    encrypted_data: Optional[str] = None
    key_id: str
    algorithm: str
    error: Optional[str] = None


class DecryptionRequest(BaseModel):
    encrypted_data: Dict[str, Any]


class DecryptionResponse(BaseModel):
    success: bool
    decrypted_data: Optional[str] = None
    error: Optional[str] = None


class AuditLogRequest(BaseModel):
    level: str
    category: str
    event_type: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None
    details: Dict[str, Any] = {}
    compliance_frameworks: List[str] = []


class AuditLogResponse(BaseModel):
    success: bool
    event_id: Optional[str] = None
    error: Optional[str] = None


class AuditQueryRequest(BaseModel):
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    levels: List[str] = []
    categories: List[str] = []
    user_ids: List[str] = []
    source_ips: List[str] = []
    limit: int = 100


class AccessControlRequest(BaseModel):
    subject_id: str
    resource: str
    action: str
    context: Dict[str, Any] = {}


class AccessControlResponse(BaseModel):
    decision: str
    reason: str
    processing_time_ms: float


class CreateSubjectRequest(BaseModel):
    subject_id: str
    subject_type: str = "user"
    roles: Set[str] = set()
    attributes: Dict[str, Any] = {}


class CreateRoleRequest(BaseModel):
    role_id: str
    name: str
    description: str = ""
    permissions: Set[str] = set()
    parent_roles: Set[str] = set()


class SecurityMetricsResponse(BaseModel):
    security_framework: Dict[str, Any]
    zero_trust: Dict[str, Any]
    threat_detection: Dict[str, Any]
    encryption: Dict[str, Any]
    audit: Dict[str, Any]
    access_control: Dict[str, Any]


# Dependency for authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Verify token (simplified - would use proper JWT verification)
    token = credentials.credentials
    payload = security_framework.verify_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return payload.get("user_id")


# Authentication endpoints
@router.post("/authenticate", response_model=AuthenticationResponse)
async def authenticate_user(request: AuthenticationRequest):
    """Authenticate user and create session"""
    try:
        context = security_framework.authenticate_user(
            username=request.username,
            credentials=request.credentials,
            ip_address=request.ip_address,
            user_agent=request.user_agent
        )
        
        if context:
            # Generate JWT token
            token = security_framework.generate_secure_token({
                "user_id": context.user_id,
                "session_id": context.session_id
            })
            
            # Log authentication event
            await audit_logger.log_authentication(
                user_id=context.user_id,
                result="success",
                source_ip=request.ip_address
            )
            
            return AuthenticationResponse(
                success=True,
                session_id=context.session_id,
                token=token,
                expires_at=context.expires_at,
                message="Authentication successful"
            )
        else:
            # Log failed authentication
            await audit_logger.log_authentication(
                user_id=request.username,
                result="failure",
                source_ip=request.ip_address
            )
            
            return AuthenticationResponse(
                success=False,
                message="Authentication failed"
            )
            
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication service error")


@router.post("/authorize", response_model=AuthorizationResponse)
async def authorize_access(request: AuthorizationRequest, user_id: str = Depends(get_current_user)):
    """Check authorization for resource access"""
    try:
        # Get security context
        context = security_framework.active_sessions.get(request.session_id)
        if not context:
            raise HTTPException(status_code=401, detail="Invalid session")
        
        authorized = security_framework.authorize_access(
            context=context,
            resource=request.resource,
            action=request.action,
            policy_id=request.policy_id
        )
        
        # Also check with access control manager
        access_response = await access_control.check_access(
            subject_id=user_id,
            resource=request.resource,
            action=request.action
        )
        
        final_authorized = authorized and access_response.decision == AccessDecision.PERMIT
        reason = access_response.reason if not final_authorized else "Access granted"
        
        return AuthorizationResponse(
            authorized=final_authorized,
            reason=reason,
            expires_at=context.expires_at
        )
        
    except Exception as e:
        logger.error(f"Authorization error: {str(e)}")
        raise HTTPException(status_code=500, detail="Authorization service error")


@router.post("/logout")
async def logout_user(session_id: str = Body(...), user_id: str = Depends(get_current_user)):
    """Logout user and invalidate session"""
    try:
        success = security_framework.invalidate_session(session_id)
        
        if success:
            await audit_logger.log_authentication(
                user_id=user_id,
                result="logout",
                details={"session_id": session_id}
            )
            
        return {"success": success, "message": "Logout successful" if success else "Session not found"}
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(status_code=500, detail="Logout service error")


# Threat Detection endpoints
@router.post("/threat/analyze", response_model=ThreatAnalysisResponse)
async def analyze_threat(request: ThreatAnalysisRequest, user_id: str = Depends(get_current_user)):
    """Analyze event for potential threats"""
    try:
        threat_event = await threat_detection.analyze_event(request.event_data)
        
        if threat_event:
            return ThreatAnalysisResponse(
                threat_detected=True,
                threat_type=threat_event.threat_type.value,
                severity=threat_event.severity.value,
                confidence=threat_event.confidence,
                threat_id=threat_event.event_id,
                recommended_actions=threat_event.mitigation_actions
            )
        else:
            return ThreatAnalysisResponse(
                threat_detected=False,
                confidence=0.0
            )
            
    except Exception as e:
        logger.error(f"Threat analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Threat analysis service error")


@router.get("/threat/metrics")
async def get_threat_metrics(user_id: str = Depends(get_current_user)):
    """Get threat detection metrics"""
    try:
        metrics = threat_detection.get_threat_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Threat metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail="Threat metrics service error")


@router.get("/threat/recent")
async def get_recent_threats(hours: int = 24, severity: Optional[str] = None, user_id: str = Depends(get_current_user)):
    """Get recent threat events"""
    try:
        severity_filter = None
        if severity:
            severity_filter = ThreatSeverity(severity)
            
        threats = threat_detection.get_recent_threats(hours=hours, severity=severity_filter)
        return {"threats": threats}
    except Exception as e:
        logger.error(f"Recent threats error: {str(e)}")
        raise HTTPException(status_code=500, detail="Recent threats service error")


# Encryption endpoints
@router.post("/encrypt", response_model=EncryptionResponse)
async def encrypt_data(request: EncryptionRequest, user_id: str = Depends(get_current_user)):
    """Encrypt sensitive data"""
    try:
        algorithm = None
        if request.algorithm:
            algorithm = EncryptionAlgorithm(request.algorithm)
            
        result = encryption_service.encrypt(
            data=request.data,
            key_id=request.key_id,
            algorithm=algorithm
        )
        
        return EncryptionResponse(
            success=True,
            encrypted_data=result.to_dict(),
            key_id=result.key_id,
            algorithm=result.algorithm.value
        )
        
    except Exception as e:
        logger.error(f"Encryption error: {str(e)}")
        return EncryptionResponse(
            success=False,
            error=str(e),
            key_id="",
            algorithm=""
        )


@router.post("/decrypt", response_model=DecryptionResponse)
async def decrypt_data(request: DecryptionRequest, user_id: str = Depends(get_current_user)):
    """Decrypt sensitive data"""
    try:
        decrypted_data = encryption_service.decrypt(request.encrypted_data)
        
        return DecryptionResponse(
            success=True,
            decrypted_data=decrypted_data.decode('utf-8')
        )
        
    except Exception as e:
        logger.error(f"Decryption error: {str(e)}")
        return DecryptionResponse(
            success=False,
            error=str(e)
        )


@router.get("/encryption/keys")
async def list_encryption_keys(user_id: str = Depends(get_current_user)):
    """List encryption keys"""
    try:
        keys = encryption_service.key_manager.list_keys()
        return {"keys": keys}
    except Exception as e:
        logger.error(f"List keys error: {str(e)}")
        raise HTTPException(status_code=500, detail="Encryption key service error")


@router.post("/encryption/keys/generate")
async def generate_encryption_key(
    algorithm: str = Body(...),
    key_id: Optional[str] = Body(None),
    expires_in_hours: Optional[int] = Body(None),
    user_id: str = Depends(get_current_user)
):
    """Generate new encryption key"""
    try:
        expires_in = None
        if expires_in_hours:
            expires_in = timedelta(hours=expires_in_hours)
            
        key = encryption_service.key_manager.generate_key(
            algorithm=EncryptionAlgorithm(algorithm),
            key_id=key_id,
            expires_in=expires_in,
            owner=user_id
        )
        
        return {"success": True, "key_id": key.key_id, "algorithm": key.algorithm.value}
        
    except Exception as e:
        logger.error(f"Key generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Key generation service error")


# Audit Logging endpoints
@router.post("/audit/log", response_model=AuditLogResponse)
async def create_audit_log(request: AuditLogRequest, user_id: str = Depends(get_current_user)):
    """Create audit log entry"""
    try:
        compliance_tags = set()
        for framework in request.compliance_frameworks:
            compliance_tags.add(ComplianceFramework(framework))
            
        event_id = await audit_logger.log(
            level=AuditLevel(request.level),
            category=AuditCategory(request.category),
            event_type=request.event_type,
            user_id=request.user_id or user_id,
            session_id=request.session_id,
            source_ip=request.source_ip,
            resource=request.resource,
            action=request.action,
            result=request.result,
            details=request.details,
            compliance_tags=compliance_tags
        )
        
        return AuditLogResponse(
            success=True,
            event_id=event_id
        )
        
    except Exception as e:
        logger.error(f"Audit log error: {str(e)}")
        return AuditLogResponse(
            success=False,
            error=str(e)
        )


@router.post("/audit/query")
async def query_audit_logs(request: AuditQueryRequest, user_id: str = Depends(get_current_user)):
    """Query audit logs"""
    try:
        from ...agents.security.audit_logger import AuditFilter
        
        audit_filter = AuditFilter(
            start_time=request.start_time,
            end_time=request.end_time,
            levels=[AuditLevel(level) for level in request.levels],
            categories=[AuditCategory(cat) for cat in request.categories],
            user_ids=request.user_ids,
            source_ips=request.source_ips,
            limit=request.limit
        )
        
        events = await audit_logger.query(audit_filter)
        
        return {
            "events": [event.to_dict() for event in events],
            "total": len(events)
        }
        
    except Exception as e:
        logger.error(f"Audit query error: {str(e)}")
        raise HTTPException(status_code=500, detail="Audit query service error")


@router.get("/audit/verify")
async def verify_audit_integrity(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    user_id: str = Depends(get_current_user)
):
    """Verify audit log integrity"""
    try:
        result = await audit_logger.verify_integrity(start_time, end_time)
        return result
    except Exception as e:
        logger.error(f"Audit verification error: {str(e)}")
        raise HTTPException(status_code=500, detail="Audit verification service error")


# Access Control endpoints
@router.post("/access/check", response_model=AccessControlResponse)
async def check_access_control(request: AccessControlRequest, user_id: str = Depends(get_current_user)):
    """Check access control permissions"""
    try:
        response = await access_control.check_access(
            subject_id=request.subject_id,
            resource=request.resource,
            action=request.action,
            context=request.context
        )
        
        return AccessControlResponse(
            decision=response.decision.value,
            reason=response.reason,
            processing_time_ms=response.processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Access control error: {str(e)}")
        raise HTTPException(status_code=500, detail="Access control service error")


@router.post("/access/subjects")
async def create_subject(request: CreateSubjectRequest, user_id: str = Depends(get_current_user)):
    """Create new subject for access control"""
    try:
        subject = access_control.create_subject(
            subject_id=request.subject_id,
            subject_type=request.subject_type,
            roles=request.roles,
            attributes=request.attributes
        )
        
        return {"success": True, "subject": subject.to_dict()}
        
    except Exception as e:
        logger.error(f"Create subject error: {str(e)}")
        raise HTTPException(status_code=500, detail="Subject creation service error")


@router.post("/access/roles")
async def create_role(request: CreateRoleRequest, user_id: str = Depends(get_current_user)):
    """Create new role for access control"""
    try:
        role = access_control.create_role(
            role_id=request.role_id,
            name=request.name,
            description=request.description,
            permissions=request.permissions,
            parent_roles=request.parent_roles
        )
        
        return {"success": True, "role": role.to_dict()}
        
    except Exception as e:
        logger.error(f"Create role error: {str(e)}")
        raise HTTPException(status_code=500, detail="Role creation service error")


@router.get("/access/subjects")
async def list_subjects(subject_type: Optional[str] = None, user_id: str = Depends(get_current_user)):
    """List all subjects"""
    try:
        subjects = access_control.list_subjects(subject_type=subject_type)
        return {"subjects": subjects}
    except Exception as e:
        logger.error(f"List subjects error: {str(e)}")
        raise HTTPException(status_code=500, detail="List subjects service error")


@router.get("/access/roles")
async def list_roles(user_id: str = Depends(get_current_user)):
    """List all roles"""
    try:
        roles = access_control.list_roles()
        return {"roles": roles}
    except Exception as e:
        logger.error(f"List roles error: {str(e)}")
        raise HTTPException(status_code=500, detail="List roles service error")


@router.get("/access/permissions")
async def list_permissions(user_id: str = Depends(get_current_user)):
    """List all permissions"""
    try:
        permissions = access_control.list_permissions()
        return {"permissions": permissions}
    except Exception as e:
        logger.error(f"List permissions error: {str(e)}")
        raise HTTPException(status_code=500, detail="List permissions service error")


@router.post("/access/subjects/{subject_id}/roles/{role_id}")
async def assign_role(subject_id: str, role_id: str, user_id: str = Depends(get_current_user)):
    """Assign role to subject"""
    try:
        success = access_control.assign_role_to_subject(subject_id, role_id)
        
        if success:
            await audit_logger.log_configuration_change(
                user_id=user_id,
                resource=f"subjects/{subject_id}/roles",
                action="assign_role",
                details={"role_id": role_id}
            )
            
        return {"success": success}
        
    except Exception as e:
        logger.error(f"Assign role error: {str(e)}")
        raise HTTPException(status_code=500, detail="Role assignment service error")


@router.delete("/access/subjects/{subject_id}/roles/{role_id}")
async def revoke_role(subject_id: str, role_id: str, user_id: str = Depends(get_current_user)):
    """Revoke role from subject"""
    try:
        success = access_control.revoke_role_from_subject(subject_id, role_id)
        
        if success:
            await audit_logger.log_configuration_change(
                user_id=user_id,
                resource=f"subjects/{subject_id}/roles",
                action="revoke_role",
                details={"role_id": role_id}
            )
            
        return {"success": success}
        
    except Exception as e:
        logger.error(f"Revoke role error: {str(e)}")
        raise HTTPException(status_code=500, detail="Role revocation service error")


# Zero Trust endpoints
@router.get("/zero-trust/entities")
async def list_trust_entities(user_id: str = Depends(get_current_user)):
    """List zero trust entities"""
    try:
        entities = []
        for entity in zero_trust_model.entities.values():
            entities.append({
                "entity_id": entity.entity_id,
                "entity_type": entity.entity_type.value,
                "name": entity.name,
                "trust_level": entity.trust_level.name,
                "verification_status": entity.verification_status.value,
                "risk_score": entity.risk_score,
                "needs_verification": entity.needs_verification()
            })
        return {"entities": entities}
    except Exception as e:
        logger.error(f"Zero trust entities error: {str(e)}")
        raise HTTPException(status_code=500, detail="Zero trust service error")


@router.post("/zero-trust/evaluate")
async def evaluate_zero_trust(
    entity_id: str = Body(...),
    resource: str = Body(...),
    action: str = Body(...),
    context: Dict[str, Any] = Body({}),
    policy_id: str = Body("standard"),
    user_id: str = Depends(get_current_user)
):
    """Evaluate zero trust access"""
    try:
        result = await zero_trust_model.evaluate_trust(
            entity_id=entity_id,
            resource=resource,
            action=action,
            context=context,
            policy_id=policy_id
        )
        return result
    except Exception as e:
        logger.error(f"Zero trust evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Zero trust evaluation service error")


# Security Metrics endpoint
@router.get("/metrics", response_model=SecurityMetricsResponse)
async def get_security_metrics(user_id: str = Depends(get_current_user)):
    """Get comprehensive security metrics"""
    try:
        metrics = SecurityMetricsResponse(
            security_framework=security_framework.get_security_metrics(),
            zero_trust=zero_trust_model.get_zero_trust_metrics(),
            threat_detection=threat_detection.get_threat_metrics(),
            encryption=encryption_service.get_encryption_metrics(),
            audit=audit_logger.get_audit_metrics(),
            access_control=access_control.get_access_control_metrics()
        )
        return metrics
    except Exception as e:
        logger.error(f"Security metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail="Security metrics service error")


# Security Health Check
@router.get("/health")
async def security_health_check():
    """Security system health check"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "security_framework": "operational",
                "zero_trust_model": "operational", 
                "threat_detection": "operational",
                "encryption_service": "operational",
                "audit_logger": "operational" if audit_logger.processing else "stopped",
                "access_control": "operational"
            },
            "metrics": {
                "active_sessions": len(security_framework.active_sessions),
                "recent_threats": len([t for t in threat_detection.threat_events if (datetime.now() - t.timestamp).total_seconds() < 3600]),
                "audit_queue_size": audit_logger.event_queue.qsize() if audit_logger.processing else 0,
                "total_subjects": len(access_control.subjects),
                "total_policies": len(access_control.policies)
            }
        }
        
        # Check if any component is unhealthy
        if "stopped" in health_status["components"].values():
            health_status["status"] = "degraded"
            
        return health_status
        
    except Exception as e:
        logger.error(f"Security health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }