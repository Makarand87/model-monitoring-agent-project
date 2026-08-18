"""Pydantic schemas used by the model-monitoring application."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RiskTier(str, Enum):
    """Permitted model materiality tiers."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class MonitoringStatus(str, Enum):
    """Traffic-light classification used by monitoring policy."""

    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"


class MetricName(str, Enum):
    """Metrics currently supported by threshold classification."""

    PSI = "psi"
    AUC_CHANGE = "auc_change"


class ModelMetadata(BaseModel):
    """One governed model in the enterprise inventory."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model_id: str = Field(pattern=r"^M\d{3}$")
    model_name: str = Field(min_length=3)
    risk_tier: RiskTier
    owner: str = Field(min_length=2)


class MonitoringMetrics(BaseModel):
    """Monthly performance measurements for one model."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model_id: str = Field(pattern=r"^M\d{3}$")
    period: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    auc: float = Field(ge=0.0, le=1.0)
    psi: float = Field(ge=0.0)
    bad_rate: float = Field(ge=0.0, le=1.0)
    approval_rate: float = Field(ge=0.0, le=1.0)


class Threshold(BaseModel):
    """Two policy boundaries separating green, amber, and red."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: MetricName
    green_max: float = Field(ge=0.0)
    amber_max: float = Field(ge=0.0)

    @model_validator(mode="after")
    def check_order(self) -> "Threshold":
        if self.amber_max <= self.green_max:
            raise ValueError("amber_max must be greater than green_max")
        return self


class Breach(BaseModel):
    """Classification result for one observed monitoring value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: MetricName
    value: float = Field(ge=0.0)
    status: MonitoringStatus
    threshold: float = Field(ge=0.0)
    breach: bool


class MonitoringResult(BaseModel):
    """Complete result produced for one model and monitoring period."""

    model_config = ConfigDict(extra="forbid")

    model_metadata: ModelMetadata
    current_metrics: MonitoringMetrics
    previous_metrics: MonitoringMetrics | None = None
    metric_changes: dict[str, float] = Field(default_factory=dict)
    classifications: list[Breach] = Field(default_factory=list)
    breaches: list[Breach] = Field(default_factory=list)
    overall_status: MonitoringStatus


# Backward-compatible names retained for existing loaders, tests, and callers.
ModelInventoryRecord = ModelMetadata
MonitoringRecord = MonitoringMetrics
