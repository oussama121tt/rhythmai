"""
CardioScan AI — Auth Utilities
bcrypt password hashing + JWT access tokens
"""

import os
import hashlib
import hmac
import base64
import json
import time
from datetime import datetime, timedelta, timezone

# ── Simple bcrypt-compatible hashing (uses hashlib if passlib not installed)
# You can force the simple SHA-256 fallback by setting FORCE_SIMPLE_HASH=1
_force_simple = os.environ.get("FORCE_SIMPLE_HASH", "0") == "1"
if not _force_simple:
    try:
        from passlib.context import CryptContext
        _pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

        def hash_password(plain: str) -> str:
            """Hash a plain password using bcrypt.
            
            Args:
                plain: Plain text password
            Returns:
                Bcrypt hashed password string
            """
            return _pwd_ctx.hash(plain)

        def verify_password(plain: str, hashed: str) -> bool:
            """Verify a plain password against a bcrypt hash.
            
            Args:
                plain: Plain text password to verify
                hashed: Bcrypt hashed password
            Returns:
                True if password matches, False otherwise
            """
            return _pwd_ctx.verify(plain, hashed)

    except Exception:
        _force_simple = True

if _force_simple:
    # Fallback: SHA-256 with salt (still safe for demo)
    def hash_password(plain: str) -> str:
        """Hash password using SHA-256 with salt (fallback when bcrypt unavailable).
        
        Args:
            plain: Plain text password
        Returns:
            String in format "salt:hash" using SHA-256
        """
        salt = os.urandom(16).hex()
        h = hashlib.sha256(f"{salt}{plain}".encode()).hexdigest()
        return f"{salt}:{h}"

    def verify_password(plain: str, hashed: str) -> bool:
        """Verify password against SHA-256 hash (fallback).
        
        Args:
            plain: Plain text password to verify
            hashed: SHA-256 hashed password in format "salt:hash"
        Returns:
            True if password matches, False otherwise
        """
        try:
            salt, h = hashed.split(":", 1)
            return hmac.compare_digest(
                hashlib.sha256(f"{salt}{plain}".encode()).hexdigest(), h
            )
        except Exception:
            return False


# ── JWT (minimal, no external dependency) ────────────────────────────────────
SECRET_KEY  = os.environ.get("JWT_SECRET", "dev-secret-change-in-production")
ALGORITHM   = "HS256"
ACCESS_TTL  = int(os.environ.get("ACCESS_TTL_HOURS", 24))   # hours


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    padding = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * (padding % 4))


def create_token(user_id: int, role: str, is_admin: bool = False) -> str:
    """Create a JWT access token for a user.
    
    Args:
        user_id: User ID to embed in token
        role: User role (doctor, resident, student, etc.)
        is_admin: Whether user is admin
    Returns:
        JWT token string valid for ACCESS_TTL hours
    """
    header  = _b64(json.dumps({"alg": ALGORITHM, "typ": "JWT"}).encode())
    expires = int((datetime.now(timezone.utc) + timedelta(hours=ACCESS_TTL)).timestamp())
    payload = _b64(json.dumps({"sub": user_id, "role": role,
                                "admin": is_admin, "exp": expires}).encode())
    sig_input = f"{header}.{payload}".encode()
    sig = _b64(hmac.new(SECRET_KEY.encode(), sig_input, hashlib.sha256).digest())
    return f"{header}.{payload}.{sig}"


def decode_token(token: str) -> dict:
    """Decode and verify a JWT token.
    
    Args:
        token: JWT token string
    Returns:
        Decoded payload dict containing user_id, role, admin, exp
    Raises:
        ValueError: If token is malformed, signature invalid, or expired
    """
    try:
        header, payload, sig = token.split(".")
    except ValueError:
        raise ValueError("Malformed token")

    # Verify signature
    expected_sig = _b64(
        hmac.new(SECRET_KEY.encode(),
                 f"{header}.{payload}".encode(),
                 hashlib.sha256).digest()
    )
    if not hmac.compare_digest(sig, expected_sig):
        raise ValueError("Invalid token signature")

    data = json.loads(_unb64(payload))
    if data.get("exp", 0) < time.time():
        raise ValueError("Token expired")
    return data


def get_current_user_id(authorization: str) -> int:
    """Extract user_id from 'Bearer <token>' header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise ValueError("Missing authorization header")
    token = authorization.split(" ", 1)[1]
    return decode_token(token)["sub"]


def require_admin(authorization: str) -> bool:
    if not authorization or not authorization.startswith("Bearer "):
        raise ValueError("Missing authorization")
    token = authorization.split(" ", 1)[1]
    data = decode_token(token)
    if not data.get("admin"):
        raise ValueError("Admin access required")
    return True
