"""Stable mapping between external item ids and policy action indices."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable

__all__ = ["ActionIndexMap"]


@dataclass(frozen=True)
class ActionIndexMap:
    """Bidirectional mapping used at the simulator/policy boundary.

    Policies and OPE operate only on integer indices. A simulator that owns
    string ``item_id`` values should create one stable map for its action space
    and encode ids before calling ``select_action`` or creating ``DecisionLog``.
    """

    item_ids: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        if not self.item_ids:
            raise ValueError("item_ids must be non-empty")
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("item_ids must be unique")

    @classmethod
    def from_items(cls, item_ids: Iterable[Hashable]) -> "ActionIndexMap":
        return cls(tuple(item_ids))

    @property
    def n_actions(self) -> int:
        return len(self.item_ids)

    @property
    def item_to_index(self) -> dict[Hashable, int]:
        return {item_id: idx for idx, item_id in enumerate(self.item_ids)}

    def encode(self, item_id: Hashable) -> int:
        try:
            return self.item_to_index[item_id]
        except KeyError as exc:
            raise KeyError(f"unknown item_id: {item_id!r}") from exc

    def encode_many(self, item_ids: Iterable[Hashable]) -> list[int]:
        mapping = self.item_to_index
        result: list[int] = []
        for item_id in item_ids:
            if item_id not in mapping:
                raise KeyError(f"unknown item_id: {item_id!r}")
            result.append(mapping[item_id])
        return result

    def decode(self, action_index: int) -> Hashable:
        if not isinstance(action_index, int):
            raise TypeError("action_index must be int")
        if not 0 <= action_index < self.n_actions:
            raise IndexError("action_index out of range")
        return self.item_ids[action_index]
