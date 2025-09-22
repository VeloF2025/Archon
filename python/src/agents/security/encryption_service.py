"""
Advanced Encryption Service
Comprehensive encryption and cryptographic operations with key management
"""

import asyncio
import logging
import os
import secrets
import base64
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization, padding
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, AESOCB3, ChaCha20Poly1305
from cryptography.x509 import load_pem_x509_certificate
import jwt

logger = logging.getLogger(__name__)


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms"""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    AES_256_CTR = "aes_256_ctr"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    FERNET = "fernet"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"


class KeyType(Enum):
    """Types of cryptographic keys"""
    SYMMETRIC = "symmetric"
    ASYMMETRIC_PUBLIC = "asymmetric_public"
    ASYMMETRIC_PRIVATE = "asymmetric_private"
    DERIVED = "derived"
    EPHEMERAL = "ephemeral"


class KeyUsage(Enum):
    """Key usage purposes"""
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    SIGN = "sign"
    VERIFY = "verify"
    KEY_EXCHANGE = "key_exchange"
    DERIVE = "derive"


@dataclass
class CryptographicKey:
    """Cryptographic key with metadata"""
    key_id: str
    key_type: KeyType
    algorithm: EncryptionAlgorithm
    key_data: bytes
    usage: List[KeyUsage]
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    owner: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    
    def is_expired(self) -> bool:
        """Check if key is expired"""
        return self.expires_at and datetime.now() > self.expires_at
    
    def can_be_used_for(self, usage: KeyUsage) -> bool:
        """Check if key can be used for specific purpose"""
        return usage in self.usage and not self.is_expired()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding key data)"""
        return {
            "key_id": self.key_id,
            "key_type": self.key_type.value,
            "algorithm": self.algorithm.value,
            "usage": [u.value for u in self.usage],
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "owner": self.owner,
            "metadata": self.metadata,
            "version": self.version,
            "expired": self.is_expired()
        }


@dataclass
class EncryptionResult:
    """Result of encryption operation"""
    encrypted_data: bytes
    algorithm: EncryptionAlgorithm
    key_id: str
    iv_or_nonce: Optional[bytes] = None
    tag: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission"""
        return {
            "encrypted_data": base64.b64encode(self.encrypted_data).decode('utf-8'),
            "algorithm": self.algorithm.value,
            "key_id": self.key_id,
            "iv_or_nonce": base64.b64encode(self.iv_or_nonce).decode('utf-8') if self.iv_or_nonce else None,
            "tag": base64.b64encode(self.tag).decode('utf-8') if self.tag else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncryptionResult':
        """Create from dictionary"""
        return cls(
            encrypted_data=base64.b64decode(data["encrypted_data"]),
            algorithm=EncryptionAlgorithm(data["algorithm"]),
            key_id=data["key_id"],
            iv_or_nonce=base64.b64decode(data["iv_or_nonce"]) if data.get("iv_or_nonce") else None,
            tag=base64.b64decode(data["tag"]) if data.get("tag") else None,
            metadata=data.get("metadata", {})
        )


class KeyManager:
    """Cryptographic key management system"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.keys: Dict[str, CryptographicKey] = {}
        self.key_hierarchy: Dict[str, List[str]] = {}  # parent -> children
        self.master_key = master_key or Fernet.generate_key()
        self.fernet = Fernet(self.master_key)
        
    def generate_key(self, algorithm: EncryptionAlgorithm, key_id: Optional[str] = None,
                    usage: List[KeyUsage] = None, expires_in: Optional[timedelta] = None,
                    owner: Optional[str] = None) -> CryptographicKey:
        """Generate new cryptographic key"""
        if not key_id:
            key_id = str(uuid.uuid4())
            
        if not usage:
            usage = [KeyUsage.ENCRYPT, KeyUsage.DECRYPT]
            
        expires_at = datetime.now() + expires_in if expires_in else None
        
        # Generate key based on algorithm
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            key_data = secrets.token_bytes(32)  # 256 bits
            key_type = KeyType.SYMMETRIC
        elif algorithm == EncryptionAlgorithm.AES_256_CBC:
            key_data = secrets.token_bytes(32)
            key_type = KeyType.SYMMETRIC
        elif algorithm == EncryptionAlgorithm.AES_256_CTR:
            key_data = secrets.token_bytes(32)
            key_type = KeyType.SYMMETRIC
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            key_data = secrets.token_bytes(32)
            key_type = KeyType.SYMMETRIC
        elif algorithm == EncryptionAlgorithm.FERNET:
            key_data = Fernet.generate_key()
            key_type = KeyType.SYMMETRIC
        elif algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
            key_size = 2048 if algorithm == EncryptionAlgorithm.RSA_2048 else 4096
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size
            )
            key_data = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            key_type = KeyType.ASYMMETRIC_PRIVATE
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        key = CryptographicKey(
            key_id=key_id,
            key_type=key_type,
            algorithm=algorithm,
            key_data=key_data,
            usage=usage,
            expires_at=expires_at,
            owner=owner
        )
        
        self.keys[key_id] = key
        logger.info(f"Generated key {key_id} with algorithm {algorithm.value}")
        
        return key
        
    def get_key(self, key_id: str) -> Optional[CryptographicKey]:
        """Get key by ID"""
        return self.keys.get(key_id)
        
    def derive_key(self, parent_key_id: str, salt: bytes, info: bytes,
                   algorithm: EncryptionAlgorithm, key_id: Optional[str] = None) -> CryptographicKey:
        """Derive new key from existing key using HKDF"""
        parent_key = self.get_key(parent_key_id)
        if not parent_key:
            raise ValueError(f"Parent key {parent_key_id} not found")
            
        if not parent_key.can_be_used_for(KeyUsage.DERIVE):
            raise ValueError("Parent key cannot be used for derivation")
            
        # Use PBKDF2 for key derivation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        derived_key_data = kdf.derive(parent_key.key_data)
        
        if not key_id:
            key_id = str(uuid.uuid4())
            
        derived_key = CryptographicKey(
            key_id=key_id,
            key_type=KeyType.DERIVED,
            algorithm=algorithm,
            key_data=derived_key_data,
            usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
            metadata={"parent_key_id": parent_key_id}
        )
        
        self.keys[key_id] = derived_key
        
        # Update hierarchy
        if parent_key_id not in self.key_hierarchy:
            self.key_hierarchy[parent_key_id] = []
        self.key_hierarchy[parent_key_id].append(key_id)
        
        logger.info(f"Derived key {key_id} from parent {parent_key_id}")
        
        return derived_key
        
    def rotate_key(self, key_id: str) -> CryptographicKey:
        """Rotate existing key (generate new version)"""
        old_key = self.get_key(key_id)
        if not old_key:
            raise ValueError(f"Key {key_id} not found")
            
        # Generate new key with same parameters
        new_key = self.generate_key(
            algorithm=old_key.algorithm,
            key_id=f"{key_id}_v{old_key.version + 1}",
            usage=old_key.usage,
            owner=old_key.owner
        )
        
        new_key.version = old_key.version + 1
        new_key.metadata = old_key.metadata.copy()
        new_key.metadata["previous_version"] = key_id
        
        # Mark old key as expired
        old_key.expires_at = datetime.now()
        
        logger.info(f"Rotated key {key_id} to version {new_key.version}")
        
        return new_key
        
    def delete_key(self, key_id: str) -> bool:
        """Securely delete key"""
        if key_id in self.keys:
            # Overwrite key data with random bytes (simplified)
            key = self.keys[key_id]
            key.key_data = secrets.token_bytes(len(key.key_data))
            
            # Remove from storage
            del self.keys[key_id]
            
            # Clean up hierarchy
            if key_id in self.key_hierarchy:
                del self.key_hierarchy[key_id]
                
            logger.info(f"Deleted key {key_id}")
            return True
        return False
        
    def export_public_key(self, key_id: str) -> Optional[bytes]:
        """Export public key for asymmetric key pair"""
        key = self.get_key(key_id)
        if not key or key.key_type != KeyType.ASYMMETRIC_PRIVATE:
            return None
            
        private_key = serialization.load_pem_private_key(key.key_data, password=None)
        public_key = private_key.public_key()
        
        return public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
    def list_keys(self, owner: Optional[str] = None, algorithm: Optional[EncryptionAlgorithm] = None) -> List[Dict[str, Any]]:
        """List keys with optional filtering"""
        result = []
        
        for key in self.keys.values():
            if owner and key.owner != owner:
                continue
            if algorithm and key.algorithm != algorithm:
                continue
                
            result.append(key.to_dict())
            
        return result


class EncryptionService:
    """Main encryption service providing high-level encryption operations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.key_manager = KeyManager(self.config.get("master_key"))
        self.default_algorithm = EncryptionAlgorithm.AES_256_GCM
        
        # Initialize with default encryption key
        self._initialize_default_keys()
        
    def _initialize_default_keys(self) -> None:
        """Initialize default encryption keys"""
        # Generate default symmetric key
        self.key_manager.generate_key(
            algorithm=self.default_algorithm,
            key_id="default_symmetric",
            usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
            owner="system"
        )
        
        # Generate default RSA key pair
        self.key_manager.generate_key(
            algorithm=EncryptionAlgorithm.RSA_2048,
            key_id="default_rsa",
            usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT, KeyUsage.SIGN, KeyUsage.VERIFY],
            owner="system"
        )
        
        logger.info("Initialized default encryption keys")
        
    def encrypt(self, data: Union[str, bytes], key_id: Optional[str] = None,
               algorithm: Optional[EncryptionAlgorithm] = None,
               additional_data: Optional[bytes] = None) -> EncryptionResult:
        """Encrypt data using specified key and algorithm"""
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        if not key_id:
            key_id = "default_symmetric"
            
        if not algorithm:
            key = self.key_manager.get_key(key_id)
            if key:
                algorithm = key.algorithm
            else:
                algorithm = self.default_algorithm
                
        key = self.key_manager.get_key(key_id)
        if not key:
            raise ValueError(f"Key {key_id} not found")
            
        if not key.can_be_used_for(KeyUsage.ENCRYPT):
            raise ValueError(f"Key {key_id} cannot be used for encryption")
            
        return self._encrypt_with_algorithm(data, key, algorithm, additional_data)
        
    def _encrypt_with_algorithm(self, data: bytes, key: CryptographicKey,
                               algorithm: EncryptionAlgorithm,
                               additional_data: Optional[bytes] = None) -> EncryptionResult:
        """Encrypt data with specific algorithm"""
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._encrypt_aes_gcm(data, key, additional_data)
        elif algorithm == EncryptionAlgorithm.AES_256_CBC:
            return self._encrypt_aes_cbc(data, key)
        elif algorithm == EncryptionAlgorithm.AES_256_CTR:
            return self._encrypt_aes_ctr(data, key)
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._encrypt_chacha20(data, key, additional_data)
        elif algorithm == EncryptionAlgorithm.FERNET:
            return self._encrypt_fernet(data, key)
        elif algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
            return self._encrypt_rsa(data, key)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
    def _encrypt_aes_gcm(self, data: bytes, key: CryptographicKey,
                        additional_data: Optional[bytes] = None) -> EncryptionResult:
        """Encrypt with AES-256-GCM"""
        aesgcm = AESGCM(key.key_data)
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        
        encrypted_data = aesgcm.encrypt(nonce, data, additional_data)
        
        # GCM mode includes authentication tag in the encrypted data
        # Split encrypted data and tag
        ciphertext = encrypted_data[:-16]
        tag = encrypted_data[-16:]
        
        return EncryptionResult(
            encrypted_data=ciphertext,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_id=key.key_id,
            iv_or_nonce=nonce,
            tag=tag
        )
        
    def _encrypt_aes_cbc(self, data: bytes, key: CryptographicKey) -> EncryptionResult:
        """Encrypt with AES-256-CBC"""
        iv = os.urandom(16)  # 128-bit IV for CBC
        
        # Apply PKCS7 padding
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        
        cipher = Cipher(algorithms.AES(key.key_data), modes.CBC(iv))
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        return EncryptionResult(
            encrypted_data=encrypted_data,
            algorithm=EncryptionAlgorithm.AES_256_CBC,
            key_id=key.key_id,
            iv_or_nonce=iv
        )
        
    def _encrypt_aes_ctr(self, data: bytes, key: CryptographicKey) -> EncryptionResult:
        """Encrypt with AES-256-CTR"""
        nonce = os.urandom(16)  # 128-bit nonce for CTR
        
        cipher = Cipher(algorithms.AES(key.key_data), modes.CTR(nonce))
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(data) + encryptor.finalize()
        
        return EncryptionResult(
            encrypted_data=encrypted_data,
            algorithm=EncryptionAlgorithm.AES_256_CTR,
            key_id=key.key_id,
            iv_or_nonce=nonce
        )
        
    def _encrypt_chacha20(self, data: bytes, key: CryptographicKey,
                         additional_data: Optional[bytes] = None) -> EncryptionResult:
        """Encrypt with ChaCha20-Poly1305"""
        chacha = ChaCha20Poly1305(key.key_data)
        nonce = os.urandom(12)  # 96-bit nonce
        
        encrypted_data = chacha.encrypt(nonce, data, additional_data)
        
        # Split ciphertext and tag
        ciphertext = encrypted_data[:-16]
        tag = encrypted_data[-16:]
        
        return EncryptionResult(
            encrypted_data=ciphertext,
            algorithm=EncryptionAlgorithm.CHACHA20_POLY1305,
            key_id=key.key_id,
            iv_or_nonce=nonce,
            tag=tag
        )
        
    def _encrypt_fernet(self, data: bytes, key: CryptographicKey) -> EncryptionResult:
        """Encrypt with Fernet"""
        fernet = Fernet(key.key_data)
        encrypted_data = fernet.encrypt(data)
        
        return EncryptionResult(
            encrypted_data=encrypted_data,
            algorithm=EncryptionAlgorithm.FERNET,
            key_id=key.key_id
        )
        
    def _encrypt_rsa(self, data: bytes, key: CryptographicKey) -> EncryptionResult:
        """Encrypt with RSA"""
        private_key = serialization.load_pem_private_key(key.key_data, password=None)
        public_key = private_key.public_key()
        
        # RSA encryption with OAEP padding
        encrypted_data = public_key.encrypt(
            data,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return EncryptionResult(
            encrypted_data=encrypted_data,
            algorithm=key.algorithm,
            key_id=key.key_id
        )
        
    def decrypt(self, encryption_result: Union[EncryptionResult, Dict[str, Any]],
               additional_data: Optional[bytes] = None) -> bytes:
        """Decrypt data using encryption result"""
        if isinstance(encryption_result, dict):
            encryption_result = EncryptionResult.from_dict(encryption_result)
            
        key = self.key_manager.get_key(encryption_result.key_id)
        if not key:
            raise ValueError(f"Key {encryption_result.key_id} not found")
            
        if not key.can_be_used_for(KeyUsage.DECRYPT):
            raise ValueError(f"Key {encryption_result.key_id} cannot be used for decryption")
            
        return self._decrypt_with_algorithm(encryption_result, key, additional_data)
        
    def _decrypt_with_algorithm(self, encryption_result: EncryptionResult,
                               key: CryptographicKey,
                               additional_data: Optional[bytes] = None) -> bytes:
        """Decrypt data with specific algorithm"""
        algorithm = encryption_result.algorithm
        
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._decrypt_aes_gcm(encryption_result, key, additional_data)
        elif algorithm == EncryptionAlgorithm.AES_256_CBC:
            return self._decrypt_aes_cbc(encryption_result, key)
        elif algorithm == EncryptionAlgorithm.AES_256_CTR:
            return self._decrypt_aes_ctr(encryption_result, key)
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._decrypt_chacha20(encryption_result, key, additional_data)
        elif algorithm == EncryptionAlgorithm.FERNET:
            return self._decrypt_fernet(encryption_result, key)
        elif algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
            return self._decrypt_rsa(encryption_result, key)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
    def _decrypt_aes_gcm(self, encryption_result: EncryptionResult,
                        key: CryptographicKey,
                        additional_data: Optional[bytes] = None) -> bytes:
        """Decrypt AES-256-GCM"""
        aesgcm = AESGCM(key.key_data)
        
        # Reconstruct encrypted data with tag
        encrypted_with_tag = encryption_result.encrypted_data + encryption_result.tag
        
        return aesgcm.decrypt(encryption_result.iv_or_nonce, encrypted_with_tag, additional_data)
        
    def _decrypt_aes_cbc(self, encryption_result: EncryptionResult,
                        key: CryptographicKey) -> bytes:
        """Decrypt AES-256-CBC"""
        cipher = Cipher(algorithms.AES(key.key_data), modes.CBC(encryption_result.iv_or_nonce))
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encryption_result.encrypted_data) + decryptor.finalize()
        
        # Remove PKCS7 padding
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()
        
    def _decrypt_aes_ctr(self, encryption_result: EncryptionResult,
                        key: CryptographicKey) -> bytes:
        """Decrypt AES-256-CTR"""
        cipher = Cipher(algorithms.AES(key.key_data), modes.CTR(encryption_result.iv_or_nonce))
        decryptor = cipher.decryptor()
        return decryptor.update(encryption_result.encrypted_data) + decryptor.finalize()
        
    def _decrypt_chacha20(self, encryption_result: EncryptionResult,
                         key: CryptographicKey,
                         additional_data: Optional[bytes] = None) -> bytes:
        """Decrypt ChaCha20-Poly1305"""
        chacha = ChaCha20Poly1305(key.key_data)
        
        # Reconstruct encrypted data with tag
        encrypted_with_tag = encryption_result.encrypted_data + encryption_result.tag
        
        return chacha.decrypt(encryption_result.iv_or_nonce, encrypted_with_tag, additional_data)
        
    def _decrypt_fernet(self, encryption_result: EncryptionResult,
                       key: CryptographicKey) -> bytes:
        """Decrypt Fernet"""
        fernet = Fernet(key.key_data)
        return fernet.decrypt(encryption_result.encrypted_data)
        
    def _decrypt_rsa(self, encryption_result: EncryptionResult,
                    key: CryptographicKey) -> bytes:
        """Decrypt RSA"""
        private_key = serialization.load_pem_private_key(key.key_data, password=None)
        
        return private_key.decrypt(
            encryption_result.encrypted_data,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
    def sign_data(self, data: Union[str, bytes], key_id: str) -> Dict[str, Any]:
        """Sign data with private key"""
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        key = self.key_manager.get_key(key_id)
        if not key:
            raise ValueError(f"Key {key_id} not found")
            
        if not key.can_be_used_for(KeyUsage.SIGN):
            raise ValueError(f"Key {key_id} cannot be used for signing")
            
        private_key = serialization.load_pem_private_key(key.key_data, password=None)
        
        signature = private_key.sign(
            data,
            asym_padding.PSS(
                mgf=asym_padding.MGF1(hashes.SHA256()),
                salt_length=asym_padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return {
            "signature": base64.b64encode(signature).decode('utf-8'),
            "key_id": key_id,
            "algorithm": key.algorithm.value,
            "data_hash": hashlib.sha256(data).hexdigest()
        }
        
    def verify_signature(self, data: Union[str, bytes], signature_info: Dict[str, Any]) -> bool:
        """Verify digital signature"""
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        key = self.key_manager.get_key(signature_info["key_id"])
        if not key:
            raise ValueError(f"Key {signature_info['key_id']} not found")
            
        # Get public key
        public_key_data = self.key_manager.export_public_key(key.key_id)
        if not public_key_data:
            raise ValueError("Cannot export public key")
            
        public_key = serialization.load_pem_public_key(public_key_data)
        signature = base64.b64decode(signature_info["signature"])
        
        try:
            public_key.verify(
                signature,
                data,
                asym_padding.PSS(
                    mgf=asym_padding.MGF1(hashes.SHA256()),
                    salt_length=asym_padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
            
    def hash_data(self, data: Union[str, bytes], algorithm: str = "sha256") -> str:
        """Hash data with specified algorithm"""
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        if algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif algorithm == "sha512":
            return hashlib.sha512(data).hexdigest()
        elif algorithm == "blake2b":
            return hashlib.blake2b(data).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
    def generate_hmac(self, data: Union[str, bytes], key_id: str) -> str:
        """Generate HMAC for data integrity"""
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        key = self.key_manager.get_key(key_id)
        if not key:
            raise ValueError(f"Key {key_id} not found")
            
        return hmac.new(key.key_data, data, hashlib.sha256).hexdigest()
        
    def verify_hmac(self, data: Union[str, bytes], hmac_value: str, key_id: str) -> bool:
        """Verify HMAC"""
        computed_hmac = self.generate_hmac(data, key_id)
        return hmac.compare_digest(computed_hmac, hmac_value)
        
    def generate_jwt_token(self, payload: Dict[str, Any], key_id: str,
                          expires_in: timedelta = timedelta(hours=1)) -> str:
        """Generate JWT token"""
        key = self.key_manager.get_key(key_id)
        if not key:
            raise ValueError(f"Key {key_id} not found")
            
        payload["exp"] = datetime.utcnow() + expires_in
        payload["iat"] = datetime.utcnow()
        payload["key_id"] = key_id
        
        if key.algorithm == EncryptionAlgorithm.FERNET:
            # Use HMAC for symmetric keys
            return jwt.encode(payload, key.key_data, algorithm="HS256")
        else:
            # Use RSA for asymmetric keys
            private_key = serialization.load_pem_private_key(key.key_data, password=None)
            return jwt.encode(payload, private_key, algorithm="RS256")
            
    def verify_jwt_token(self, token: str, key_id: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        key = self.key_manager.get_key(key_id)
        if not key:
            raise ValueError(f"Key {key_id} not found")
            
        try:
            if key.algorithm == EncryptionAlgorithm.FERNET:
                # Use HMAC for symmetric keys
                payload = jwt.decode(token, key.key_data, algorithms=["HS256"])
            else:
                # Use RSA for asymmetric keys
                public_key_data = self.key_manager.export_public_key(key_id)
                public_key = serialization.load_pem_public_key(public_key_data)
                payload = jwt.decode(token, public_key, algorithms=["RS256"])
                
            return payload
        except jwt.InvalidTokenError:
            return None
            
    def get_encryption_metrics(self) -> Dict[str, Any]:
        """Get encryption service metrics"""
        total_keys = len(self.key_manager.keys)
        expired_keys = sum(1 for key in self.key_manager.keys.values() if key.is_expired())
        
        algorithm_distribution = {}
        for key in self.key_manager.keys.values():
            algo = key.algorithm.value
            algorithm_distribution[algo] = algorithm_distribution.get(algo, 0) + 1
            
        return {
            "total_keys": total_keys,
            "expired_keys": expired_keys,
            "algorithm_distribution": algorithm_distribution,
            "key_hierarchy_depth": len(self.key_manager.key_hierarchy),
            "default_algorithm": self.default_algorithm.value
        }