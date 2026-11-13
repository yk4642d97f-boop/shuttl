from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from jose import JWTError
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.config import settings
from app.models.user import User
from app.schemas.auth import RefreshRequest, TokenResponse, UserCreate
from app.core.security import (
    verify_password,
    hash_password,
    create_access_token,
    create_refresh_token,
    decode_token,
)

router = APIRouter()


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.scalar(select(User).where(User.email == payload.email))
    if existing:
        raise HTTPException(status_code=400, detail="该邮箱已注册")
    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        nickname=payload.nickname,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return _make_tokens(str(user.id))


@router.post("/login", response_model=TokenResponse)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    user = await db.scalar(select(User).where(User.email == form.username))
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="邮箱或密码错误",
        )
    return _make_tokens(str(user.id))


@router.post("/refresh", response_model=TokenResponse)
async def refresh(payload: RefreshRequest, db: AsyncSession = Depends(get_db)):
    invalid = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="刷新令牌无效或已过期，请重新登录",
    )
    try:
        user_id = decode_token(payload.refresh_token, expected_type="refresh")
        user_uuid = uuid.UUID(user_id)
    except (JWTError, ValueError):
        raise invalid
    # The account may have been deleted since the refresh token was issued.
    user = await db.get(User, user_uuid)
    if user is None:
        raise invalid
    return _make_tokens(user_id)


def _make_tokens(user_id: str) -> TokenResponse:
    access = create_access_token(
        subject=user_id,
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
    )
    refresh = create_refresh_token(
        subject=user_id,
        expires_delta=timedelta(days=settings.refresh_token_expire_days),
    )
    return TokenResponse(access_token=access, refresh_token=refresh)
