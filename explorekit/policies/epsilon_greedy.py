from __future__ import annotations

import numpy as np

from .base import BasePolicy, Decision


class EpsilonGreedyPolicy(BasePolicy):
    """P(best) = 1 − ε + ε/|E|, P(other in E) = ε/|E|,
    где E — exploration-подмножество (по умолчанию = все allowed кандидаты).

    Параметры:
      epsilon            — целевая доля exploration; клипается в [min, max];
      min_epsilon        — нижняя граница для epsilon;
      max_epsilon        — верхняя граница для epsilon;
      whitelist          — int-индексы разрешённых айтемов (None = все);
      blacklist          — int-индексы запрещённых айтемов;
      exploration_subset — int-индексы, среди которых размазывается ε
                           (None = все allowed кандидаты). Если подмножество
                           не пересекается с allowed — откат на все allowed.
    """

    name = "epsilon_greedy"

    def __init__(
        self,
        n_actions: int,
        epsilon: float = 0.1,
        seed: int | None = None,
        whitelist: np.ndarray | None = None,
        blacklist: np.ndarray | None = None,
        exploration_subset: np.ndarray | None = None,
        min_epsilon: float = 0.0,
        max_epsilon: float = 1.0,
    ) -> None:
        super().__init__(n_actions=n_actions, seed=seed)

        if not 0.0 <= min_epsilon <= max_epsilon <= 1.0:
            raise ValueError(
                f"need 0 <= min_epsilon <= max_epsilon <= 1, "
                f"got min={min_epsilon}, max={max_epsilon}"
            )
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")

        self.min_epsilon = float(min_epsilon)
        self.max_epsilon = float(max_epsilon)
        self.epsilon = float(np.clip(epsilon, self.min_epsilon, self.max_epsilon))

        self.whitelist = whitelist
        self.blacklist = blacklist
        self.exploration_subset = exploration_subset

                                               

    def _allowed_mask(self, candidate_items: np.ndarray | None) -> np.ndarray:
        """(n_actions,) boolean mask разрешённых айтемов."""
        if candidate_items is None:
            mask = np.ones(self.n_actions, dtype=bool)
        else:
            mask = np.zeros(self.n_actions, dtype=bool)
            mask[candidate_items] = True

        if self.whitelist is not None:
            wl = np.zeros(self.n_actions, dtype=bool)
            wl[self.whitelist] = True
            mask &= wl

        if self.blacklist is not None:
            bl = np.zeros(self.n_actions, dtype=bool)
            bl[self.blacklist] = True
            mask &= ~bl

        return mask

    def _exploration_mask(self, allowed: np.ndarray) -> np.ndarray:
        """(n_actions,) маска для exploration-ветки.
        Всегда подмножество allowed и непустое (fallback на allowed)."""
        if self.exploration_subset is None:
            return allowed

        expl = np.zeros(self.n_actions, dtype=bool)
        expl[self.exploration_subset] = True
        expl &= allowed

                                                   
        return expl if expl.any() else allowed

                                                      

    def action_distribution(
        self,
        context: np.ndarray,
        base_scores: np.ndarray,
        candidate_items: np.ndarray | None = None,
    ) -> np.ndarray:
        single_round = base_scores.ndim == 1
        scores = base_scores[None, :] if single_round else base_scores          
        n, n_actions = scores.shape

        allowed = self._allowed_mask(candidate_items)
        if not allowed.any():
            raise ValueError("No allowed candidates for this policy")

        expl = self._exploration_mask(allowed)

        masked_scores = np.where(allowed[None, :], scores, -np.inf)
        best = np.argmax(masked_scores, axis=1)                              

        dist = np.zeros((n, n_actions), dtype=float)
                                         
        dist[:, expl] = self.epsilon / expl.sum()
                                                            
        dist[np.arange(n), best] += 1.0 - self.epsilon

        return dist[0] if single_round else dist

    def select_action(
        self,
        context: np.ndarray,
        base_scores: np.ndarray,
        candidate_items: np.ndarray | None = None,
    ) -> Decision:
        if base_scores.ndim != 1:
            raise ValueError(
                f"select_action expects 1D base_scores (single round), "
                f"got shape {base_scores.shape}. "
                f"Use action_distribution for batches."
            )

        dist = self.action_distribution(context, base_scores, candidate_items)
        chosen = int(self.rng.choice(self.n_actions, p=dist))

        allowed = self._allowed_mask(candidate_items)
        greedy = int(np.argmax(np.where(allowed, base_scores, -np.inf)))

        return Decision(
            chosen_item=chosen,
            propensity=float(dist[chosen]),
            probabilities=dist,
            is_exploration=(chosen != greedy),
            policy_name=self.name,
        )

    def update(
        self,
        decision: Decision,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        return None
