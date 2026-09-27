"""Каталог айтемов: тёплые с дня 0 и волны новинок по расписанию.

``item_id`` всегда int-индекс в ``[0, n_actions)``. Строковые sku в Decision /
DecisionLog / OPE не попадают — это требование Полины и Сергея.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_CATEGORIES = ("electronics", "fashion", "home", "media", "grocery")

__all__ = [
    "Item",
    "ItemCatalog",
    "DEFAULT_CATEGORIES",
    "catalog_layout",
]


@dataclass
class Item:
    """Один айтем каталога.

    ``created_at_day`` для тёплых товаров отрицательный, чтобы к дню 0 они
    уже не считались cold по возрасту.
    """

    item_id: int
    category: str
    category_id: int
    features: np.ndarray
    created_at_day: int

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, (int, np.integer)):
            raise TypeError("item_id must be an int action index")
        self.item_id = int(self.item_id)
        self.features = np.asarray(self.features, dtype=float)
        if self.features.ndim != 1 or self.features.size == 0:
            raise ValueError("item features must be a non-empty 1D vector")


def catalog_layout(n_actions: int) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Сколько тёплых айтемов и какие волны новинок влезают в ``n_actions``.

    По ТЗ: день 0 — старые, затем две волны новых. Масштабируется и на
    маленький каталог (тесты), и на боевые 50–100 айтемов.
    """

    if n_actions < 4:
        raise ValueError("n_actions must be at least 4 to fit warm + new items")
    n_new_total = max(2, n_actions // 4)
    n_new_total = min(n_new_total, n_actions - 2)
    n_warm = n_actions - n_new_total
    n_wave1 = max(1, n_new_total // 2)
    n_wave2 = n_new_total - n_wave1
    waves: list[tuple[int, int]] = [(5, n_wave1)]
    if n_wave2 > 0:
        waves.append((10, n_wave2))
    return n_warm, tuple(waves)


class ItemCatalog:
    """Фиксированный каталог на весь прогон симулятора."""

    def __init__(
        self,
        n_actions: int,
        feature_dim: int,
        *,
        rng: np.random.Generator,
        categories: tuple[str, ...] = DEFAULT_CATEGORIES,
        warm_age_days: int = 30,
        feature_scale: float = 0.6,
    ) -> None:
        if n_actions <= 0 or feature_dim <= 0:
            raise ValueError("n_actions and feature_dim must be positive")
        if not categories:
            raise ValueError("categories must be non-empty")

        n_warm, waves = catalog_layout(n_actions)
        self.n_actions = int(n_actions)
        self.feature_dim = int(feature_dim)
        self.categories = tuple(categories)
        self.n_categories = len(self.categories)
        self.n_warm = n_warm
        self.waves = waves

        created = np.empty(n_actions, dtype=int)
        created[:n_warm] = -int(warm_age_days)
        cursor = n_warm
        for day, count in waves:
            created[cursor : cursor + count] = int(day)
            cursor += count

        category_ids = rng.integers(self.n_categories, size=n_actions)
        features = rng.normal(scale=feature_scale, size=(n_actions, feature_dim))
        centers = rng.normal(scale=0.4, size=(self.n_categories, feature_dim))
        features = features + centers[category_ids]

        self.created_at_day = created
        self.category_ids = category_ids.astype(int)
        self.features = features.astype(float)
        self.items = [
            Item(
                item_id=i,
                category=self.categories[int(category_ids[i])],
                category_id=int(category_ids[i]),
                features=features[i],
                created_at_day=int(created[i]),
            )
            for i in range(n_actions)
        ]

    def available_at(self, day: int) -> np.ndarray:
        """Int-индексы айтемов, уже появившихся к этому дню."""

        return np.flatnonzero(self.created_at_day <= int(day))

    def new_item_ids(self) -> np.ndarray:
        """Айтемы из волн новинок (не тёплое ядро дня 0)."""

        return np.flatnonzero(np.arange(self.n_actions) >= self.n_warm)

    def age_days(self, day: int) -> np.ndarray:
        """Возраст каждого айтема в днях на дату ``day``."""

        return int(day) - self.created_at_day
