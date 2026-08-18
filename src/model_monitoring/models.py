"""Pydantic schemas for model inventory and monitoring records."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RiskTier(str, Enum):
    """Permitted model materiality tiers."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ModelInventoryRecord(BaseModel):
    """One governed model in the enterprise inventory."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model_id: str = Field(pattern=r"^M\d{3}$")
    model_name: str = Field(min_length=3)
    risk_tier: RiskTier
    owner: str = Field(min_length=2)


class MonitoringRecord(BaseModel):
    """Monthly performance measurements for one model."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model_id: str = Field(pattern=r"^M\d{3}$")
    period: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    auc: float = Field(ge=0.0, le=1.0)
    psi: float = Field(ge=0.0)
    bad_rate: float = Field(ge=0.0, le=1.0)
    approval_rate: float = Field(ge=0.0, le=1.0)
