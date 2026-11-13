from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from passlib.context import CryptContext

from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(subject: str, expires_delta: timedelta) -> str:
    expire = datetime.now(timezone.utc) + expires_delta
    return jwt.encode(
        {"sub": subject, "exp": expire, "type": "access"},
        settings.secret_key,
        algorithm=settings.algorithm,
    )


def create_refresh_token(subject: str, expires_delta: timedelta) -> str:
    expire = datetime.now(timezone.utc) + expires_delta
    return jwt.encode(
        {"sub": subject, "exp": expire, "type": "refresh"},
        settings.secret_key,
        algorithm=settings.algorithm,
    )


def decode_token(token: str, expected_type: str = "access") -> str:
    """Returns the subject (user_id) or raises JWTError.

    Rejects tokens whose `type` claim doesn't match, so a long-lived refresh
    token can't be used as a bearer access token (and vice versa).
    """
    payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    if payload.get("type") != expected_type:
        raise JWTError(f"Expected {expected_type} token")
    sub: str | None = payload.get("sub")
    if sub is None:
        raise JWTError("Token missing subject")
    return sub



def create_upload_token(match_id: str, user_id: str, key: str,
                        expires_delta: timedelta | None = None) -> str:
    """Short-lived token authorising one local-storage video PUT for one match.
    Travels in the query string because the client sends no Authorization header."""
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.upload_token_expire_minutes))
    return jwt.encode(
        {"sub": user_id, "mid": match_id, "key": key, "exp": expire, "type": "upload"},
        settings.secret_key,
        algorithm=settings.algorithm,
    )


def decode_upload_token(token: str, match_id: str) -> dict:
    """Returns the claims or raises JWTError (bad signature, expired, wrong match)."""
    payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    if payload.get("type") != "upload":
        raise JWTError("Expected upload token")
    if payload.get("mid") != match_id or not payload.get("key"):
        raise JWTError("Token not valid for this match")
    return payload
