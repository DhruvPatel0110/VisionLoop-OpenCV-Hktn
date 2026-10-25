"""OpenCV 5 Edge Detection Workload Module.

Provides parameterized CPU-based edge detection pipelines with strict parameter bounds,
validation, and timing metadata.
"""

from typing import Tuple, Optional
import time
import cv2
import numpy as np
from pydantic import BaseModel, Field, field_validator


class EdgeDetectorConfig(BaseModel):
    """Configuration parameters for OpenCV 5 Edge Detection."""
    low_threshold: int = Field(default=50, ge=10, le=250, description="Canny lower hysteresis threshold")
    high_threshold: int = Field(default=150, ge=20, le=350, description="Canny upper hysteresis threshold")
    aperture_size: int = Field(default=3, description="Sobel aperture size for Canny (3, 5, or 7)")
    l2_gradient: bool = Field(default=True, description="Use L2 norm for gradient magnitude calculation")
    blur_kernel: int = Field(default=5, ge=1, le=15, description="Gaussian pre-filtering kernel size (odd integer, 1=None)")
    blur_sigma: float = Field(default=1.2, ge=0.0, le=5.0, description="Gaussian pre-filtering sigma")

    @field_validator("high_threshold")
    @classmethod
    def validate_thresholds(cls, v: int, info) -> int:
        low = info.data.get("low_threshold", 50)
        if v <= low:
            raise ValueError(f"high_threshold ({v}) must be greater than low_threshold ({low})")
        return v

    @field_validator("aperture_size")
    @classmethod
    def validate_aperture(cls, v: int) -> int:
        if v not in (3, 5, 7):
            raise ValueError(f"aperture_size must be 3, 5, or 7; got {v}")
        return v

    @field_validator("blur_kernel")
    @classmethod
    def validate_blur_kernel(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError(f"blur_kernel must be an odd integer, got {v}")
        return v


class EdgeDetectionResult(BaseModel):
    """Execution results from a single edge detection run."""
    config: EdgeDetectorConfig
    image_shape: Tuple[int, int]
    edge_pixel_count: int
    edge_density: float
    execution_latency_ms: float
    status: str = "SUCCESS"
    error_message: Optional[str] = None


class EdgeDetector:
    """CPU-friendly OpenCV 5 Edge Detection Workload."""

    MAX_DIMENSION: int = 1024

    @staticmethod
    def get_baseline_config() -> EdgeDetectorConfig:
        """Returns the standardized baseline configuration."""
        return EdgeDetectorConfig(
            low_threshold=50,
            high_threshold=150,
            aperture_size=3,
            l2_gradient=True,
            blur_kernel=5,
            blur_sigma=1.2
        )

    @classmethod
    def execute(
        cls,
        image: np.ndarray,
        config: Optional[EdgeDetectorConfig] = None
    ) -> Tuple[np.ndarray, EdgeDetectionResult]:
        """
        Executes the edge detection pipeline on an input image.

        Args:
            image: Input image (grayscale or BGR)
            config: Parameter configuration (defaults to baseline if None)

        Returns:
            Tuple of (binary edge map as uint8 ndarray, EdgeDetectionResult metadata)
        """
        if config is None:
            config = cls.get_baseline_config()

        # Validate input image
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a valid non-empty numpy array")

        if image.size == 0:
            raise ValueError("Input image has 0 pixels")

        h, w = image.shape[:2]
        if h > cls.MAX_DIMENSION or w > cls.MAX_DIMENSION:
            raise ValueError(f"Image dimension ({w}x{h}) exceeds safe CPU limit ({cls.MAX_DIMENSION}x{cls.MAX_DIMENSION})")

        # Start high-precision timer
        t_start = time.perf_counter_ns()

        # 1. Grayscale conversion if 3-channel
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. Pre-processing: Gaussian smoothing
        if config.blur_kernel > 1:
            processed = cv2.GaussianBlur(
                gray,
                (config.blur_kernel, config.blur_kernel),
                sigmaX=config.blur_sigma
            )
        else:
            processed = gray

        # 3. OpenCV 5 Canny Edge Detection
        edges = cv2.Canny(
            processed,
            threshold1=config.low_threshold,
            threshold2=config.high_threshold,
            apertureSize=config.aperture_size,
            L2gradient=config.l2_gradient
        )

        # Stop timer
        t_elapsed_ns = time.perf_counter_ns() - t_start
        latency_ms = t_elapsed_ns / 1_000_000.0

        edge_pixels = int(np.count_nonzero(edges))
        total_pixels = h * w
        density = edge_pixels / total_pixels if total_pixels > 0 else 0.0

        result = EdgeDetectionResult(
            config=config,
            image_shape=(h, w),
            edge_pixel_count=edge_pixels,
            edge_density=round(density, 5),
            execution_latency_ms=round(latency_ms, 3),
            status="SUCCESS"
        )

        return edges, result
