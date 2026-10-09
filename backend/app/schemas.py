"""Pydantic response/request schemas for the AI4Seva API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    environment: str


class BaselineModelInfo(BaseModel):
    present: bool
    metadata_readable: bool
    architecture: str | None = None
    num_classes: int | None = None
    label_names: list[str] | None = None
    input_size: list[int] | None = None
    source: str


class FineTunedModelInfo(BaseModel):
    present: bool
    verified: bool = False
    num_classes: int | None = None
    class_names: list[str] | None = None
    weights_files: list[str] = Field(default_factory=list)
    note: str | None = None


class CropInfo(BaseModel):
    crop: str
    display_name: str
    baseline: BaselineModelInfo
    fine_tuned: FineTunedModelInfo


class CropsResponse(BaseModel):
    count: int
    crops: list[CropInfo]


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ProbabilityItem(BaseModel):
    index: int
    label: str
    probability: float


class PredictionModelMeta(BaseModel):
    crop: str
    family: str
    implementation: str
    num_classes: int
    input_size: list[int]
    weights: str
    label_source: str


class PredictResponse(BaseModel):
    crop: str
    display_name: str
    model: PredictionModelMeta
    predicted_index: int
    predicted_label: str
    confidence: float
    probabilities: list[ProbabilityItem]
    disclaimer: str


# --- Part 6C: speech synthesis ----------------------------------------------


class SpeechRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize (max 200 characters).")


class TranscribeResponse(BaseModel):
    text: str
    language: str | None = None


# --- Part 6H: farmer-friendly speech response builder ------------------------


class PredictionSpeechRequest(BaseModel):
    """Prediction fields accepted by ``POST /speech/response``.

    Mirrors the relevant fields of ``PredictResponse`` (``crop``/``display_name``,
    ``predicted_label``, ``confidence``) and ignores any additional prediction
    fields, so a caller may forward the full ``/predict`` response verbatim.
    Field validation (including confidence bounds) is applied by the shared
    builder, not duplicated here.
    """

    model_config = ConfigDict(extra="ignore")

    crop: str | None = Field(
        default=None, description="Crop identifier, e.g. 'maize'."
    )
    display_name: str | None = Field(
        default=None, description="Human-readable crop name, e.g. 'Maize'."
    )
    predicted_label: str | None = Field(
        default=None, description="Predicted class label from the prediction."
    )
    confidence: Any = Field(
        default=None,
        description="Optional model confidence in [0, 1]; validated by the builder.",
    )


class PredictionSpeechResponse(BaseModel):
    text: str = Field(..., description="Concise farmer-friendly sentence.")


# --- Part 5: model inventory + comparison -----------------------------------


class MetricItem(BaseModel):
    name: str
    value: float
    split: str | None = None
    provenance: str
    source: str


class ModelMetrics(BaseModel):
    available: bool
    provenance: str
    disclaimer: str
    items: list[MetricItem] = Field(default_factory=list)
    test_samples: int | None = None
    notes: list[str] = Field(default_factory=list)


class ModelEntry(BaseModel):
    crop: str
    display_name: str
    family: str
    display_role: str
    architecture: str | None = None
    implementation: str | None = None
    num_classes: int | None = None
    labels: list[str] | None = None
    input_size: list[int] | None = None
    checkpoint_present: bool
    checkpoint_loads: bool | None = None
    verification_status: str
    preprocessing_verified: bool
    live_inference_enabled: bool
    live_inference_blocked_reason: str | None = None
    metrics: ModelMetrics
    training_info: dict[str, Any] = Field(default_factory=dict)


class ModelsResponse(BaseModel):
    count: int
    models: list[ModelEntry]


class SharedLabel(BaseModel):
    key: str
    baseline_label: str
    fine_tuned_label: str


class LabelOnly(BaseModel):
    key: str
    label: str


class ClassSetCounts(BaseModel):
    baseline: int
    fine_tuned: int
    shared: int
    baseline_only: int
    fine_tuned_only: int


class ClassSets(BaseModel):
    baseline_labels: list[str]
    fine_tuned_labels: list[str] | None = None
    shared: list[SharedLabel] = Field(default_factory=list)
    baseline_only: list[LabelOnly] = Field(default_factory=list)
    fine_tuned_only: list[LabelOnly] = Field(default_factory=list)
    counts: ClassSetCounts
    direct_class_alignment_possible: bool


class ModelComparison(BaseModel):
    live_prediction_comparison_possible: bool
    live_prediction_comparison_blocked_reason: str | None = None
    live_prediction_endpoint: str
    note: str


class ComparisonResponse(BaseModel):
    crop: str
    display_name: str
    baseline: ModelEntry
    fine_tuned: ModelEntry | None = None
    class_sets: ClassSets
    model_comparison: ModelComparison
    disclaimer: str
