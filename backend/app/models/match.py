import uuid
import enum
from datetime import datetime, timezone
from sqlalchemy import JSON, String, Float, Integer, DateTime, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional, List

from app.core.database import Base


class MatchStatus(str, enum.Enum):
    uploading = "uploading"
    analyzing = "analyzing"
    done = "done"
    failed = "failed"


class ShotType(str, enum.Enum):
    smash = "smash"
    clear = "clear"
    drop = "drop"
    net = "net"
    drive = "drive"
    serve = "serve"
    unknown = "unknown"


class PlayerSide(str, enum.Enum):
    near = "near"
    far = "far"
    unknown = "unknown"


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    status: Mapped[MatchStatus] = mapped_column(SAEnum(MatchStatus), default=MatchStatus.uploading)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    video_oss_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    celery_job_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    fps: Mapped[float] = mapped_column(Float, default=60.0)
    # Full v2 analysis document (see services/tracking/serialize.py)
    analysis: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    rallies: Mapped[List["Rally"]] = relationship(back_populates="match", cascade="all, delete-orphan")


class Rally(Base):
    __tablename__ = "rallies"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    match_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"), index=True)
    start_frame: Mapped[int] = mapped_column(Integer)
    end_frame: Mapped[int] = mapped_column(Integer)
    winner: Mapped[PlayerSide] = mapped_column(SAEnum(PlayerSide), default=PlayerSide.unknown)

    match: Mapped["Match"] = relationship(back_populates="rallies")
    shots: Mapped[List["ShotPoint"]] = relationship(back_populates="rally", cascade="all, delete-orphan")


class ShotPoint(Base):
    __tablename__ = "shot_points"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    rally_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("rallies.id", ondelete="CASCADE"), index=True)
    frame: Mapped[int] = mapped_column(Integer)
    x_px: Mapped[float] = mapped_column(Float)
    y_px: Mapped[float] = mapped_column(Float)
    x_meters: Mapped[float] = mapped_column(Float, default=0.0)
    y_meters: Mapped[float] = mapped_column(Float, default=0.0)
    speed_kmh: Mapped[float] = mapped_column(Float, default=0.0)
    shot_type: Mapped[ShotType] = mapped_column(SAEnum(ShotType), default=ShotType.unknown)

    rally: Mapped["Rally"] = relationship(back_populates="shots")
