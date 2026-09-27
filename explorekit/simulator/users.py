"""Пользователи синтетической среды.

Политики видят только вектор ``features`` и сегмент. Истинная склонность
кликать по категориям спрятана в reward-функции и пользователю/политике
напрямую не отдаётся.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_SEGMENTS = ("student", "parent", "professional", "senior")

__all__ = ["UserContext", "UserGenerator", "DEFAULT_SEGMENTS"]


@dataclass
class UserContext:
    """Один пользователь в момент показа.

    ``features`` — это context для бандита: длина совпадает с
    ``N_CONTEXT_FEATURES`` / ``policy.context_dim``.
    """

    segment: str
    segment_id: int
    features: np.ndarray

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=float)
        if self.features.ndim != 1 or self.features.size == 0:
            raise ValueError("features must be a non-empty 1D vector")
        if not np.all(np.isfinite(self.features)):
            raise ValueError("features must be finite")


class UserGenerator:
    """Сэмплирует пользователей из смеси сегментов.

    У каждого сегмента свой центр в пространстве признаков, поэтому
    разные группы по-разному «любят» разные товары.
    """

    def __init__(
        self,
        feature_dim: int,
        segments: tuple[str, ...] = DEFAULT_SEGMENTS,
        *,
        rng: np.random.Generator,
        center_scale: float = 0.6,
        noise_scale: float = 0.35,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if not segments:
            raise ValueError("segments must be non-empty")

        self.feature_dim = int(feature_dim)
        self.segments = tuple(segments)
        self.n_segments = len(self.segments)
        self.noise_scale = float(noise_scale)
        self.centers = rng.normal(
            scale=center_scale, size=(self.n_segments, self.feature_dim)
        )

    def sample(self, rng: np.random.Generator) -> UserContext:
        """Один случайный пользователь."""

        segment_id = int(rng.integers(self.n_segments))
        noise = rng.normal(scale=self.noise_scale, size=self.feature_dim)
        features = self.centers[segment_id] + noise
        return UserContext(
            segment=self.segments[segment_id],
            segment_id=segment_id,
            features=features,
        )

    def sample_batch(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Векторный сэмпл: features ``(n, d)``, segment_id ``(n,)``, имена."""

        segment_ids = rng.integers(self.n_segments, size=n)
        noise = rng.normal(scale=self.noise_scale, size=(n, self.feature_dim))
        features = self.centers[segment_ids] + noise
        names = [self.segments[int(i)] for i in segment_ids]
        return features, segment_ids.astype(int), names
