from __future__ import annotations

import numpy as np

from .base import BasePolicy, Decision


class ThompsonSamplingPolicy(BasePolicy):
    """Linear Thompson Sampling.

    Апостериор для каждого действия a:
        θ[a] ~ N(θ̂[a], A[a]^{-1}),  θ̂[a] = A[a]^{-1} b[a]
    Действие: θ̃[a] ~ N(θ̂[a], A[a]^{-1}) для всех a, argmax_a θ̃[a]ᵀ x.

    Для OPE (common support) используется Monte Carlo: сэмплируем K наборов
    θ̃, считаем argmax на каждом, propensity[a] = (#раз a победил) / K.
    Это оценка P(π(a|x)) без аналитической формулы.

    Reference: Agrawal & Goyal, "Thompson Sampling for Contextual Bandits
    with Linear Payoffs", ICML 2013.
    """

    name = "thompson"

    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        mc_samples: int = 200,
        reg: float = 1.0,
        seed: int | None = None,
        whitelist: np.ndarray | None = None,
        blacklist: np.ndarray | None = None,
    ) -> None:
        super().__init__(n_actions=n_actions, seed=seed)
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if mc_samples <= 0:
            raise ValueError("mc_samples must be positive")
        if reg <= 0:
            raise ValueError("reg must be > 0")

        self.context_dim = int(context_dim)
        self.mc_samples = int(mc_samples)
        self.reg = float(reg)
        self.whitelist = whitelist
        self.blacklist = blacklist

        self.A = np.stack([reg * np.eye(self.context_dim) for _ in range(self.n_actions)])
        self.b = np.zeros((self.n_actions, self.context_dim))

        self._last_context: np.ndarray | None = None

                                               

    def _allowed_mask(self, candidate_items: np.ndarray | None) -> np.ndarray:
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

    def _sample_thetas(self, n_samples: int) -> np.ndarray:
        """Сэмплирует n_samples наборов параметров.

        Возвращает (n_samples, n_actions, context_dim).
        Для каждого действия:
            θ̂ = A^{-1} b
            L = cholesky(A)      -> A = L Lᵀ
            z ~ N(0, I)
            θ̃ = θ̂ + solve(Lᵀ, z)  (ковариация = A^{-1}, что и нужно)
        """
        thetas = np.zeros((n_samples, self.n_actions, self.context_dim))
        for a in range(self.n_actions):
            A_a = self.A[a]
            theta_hat = np.linalg.solve(A_a, self.b[a])

            L = np.linalg.cholesky(A_a)
            z = self.rng.standard_normal((n_samples, self.context_dim))
            noise = np.linalg.solve(L.T, z.T).T                       
            thetas[:, a, :] = theta_hat + noise
        return thetas

    def _mean_scores(self, x: np.ndarray) -> np.ndarray:
        """θ̂[a]ᵀ x для всех a — для определения is_exploration."""
        scores = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            theta_hat = np.linalg.solve(self.A[a], self.b[a])
            scores[a] = theta_hat @ x
        return scores

                                                      

    def action_distribution(
        self,
        context: np.ndarray,
        base_scores: np.ndarray,
        candidate_items: np.ndarray | None = None,
    ) -> np.ndarray:
        single_round = context.ndim == 1
        X = context[None, :] if single_round else context               
        n = X.shape[0]

        allowed = self._allowed_mask(candidate_items)
        if not allowed.any():
            raise ValueError("No allowed candidates for this policy")

                                                          
        thetas = self._sample_thetas(self.mc_samples)                      

                                                                    
        chunk = 512
        dist = np.zeros((n, self.n_actions), dtype=float)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            Xc = X[start:end]                                            

                               
            scores = np.einsum("kad,cd->cka", thetas, Xc)
            scores = np.where(allowed[None, None, :], scores, -np.inf)

            winners = np.argmax(scores, axis=-1)                         

            for a in range(self.n_actions):
                dist[start:end, a] = (winners == a).mean(axis=1)

        return dist[0] if single_round else dist

    def select_action(
        self,
        context: np.ndarray,
        base_scores: np.ndarray,
        candidate_items: np.ndarray | None = None,
    ) -> Decision:
        self._last_context = np.array(context, copy=True)

        x = np.asarray(context, dtype=float).reshape(-1)
        if x.shape[0] != self.context_dim:
            raise ValueError(
                f"context has dim {x.shape[0]}, policy expects {self.context_dim}"
            )

        allowed = self._allowed_mask(candidate_items)
        if not allowed.any():
            raise ValueError("No allowed candidates for this policy")

                                                               
        thetas = self._sample_thetas(self.mc_samples)                    
        scores = np.einsum("kad,d->ka", thetas, x)                    
        scores = np.where(allowed[None, :], scores, -np.inf)
        winners = np.argmax(scores, axis=1)                         

                                                                   
        chosen = int(winners[0])

                                                                        
        probs = np.bincount(winners, minlength=self.n_actions).astype(float)
        probs /= self.mc_samples
        propensity = float(probs[chosen])

                                                     
        mean_scores = self._mean_scores(x)
        mean_scores = np.where(allowed, mean_scores, -np.inf)
        greedy = int(np.argmax(mean_scores))

        return Decision(
            chosen_item=chosen,
            propensity=propensity,
            probabilities=probs,
            is_exploration=(chosen != greedy),
            policy_name=self.name,
        )

    def update(
        self,
        decision: Decision,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        if context is None:
            if self._last_context is None:
                raise ValueError(
                    "context must be provided to update() if select_action "
                    "was not called on this policy"
                )
            context = self._last_context

        x = np.asarray(context, dtype=float).reshape(-1)
        if x.shape[0] != self.context_dim:
            raise ValueError(
                f"context has dim {x.shape[0]}, policy expects {self.context_dim}"
            )

        a = decision.chosen_item
        self.A[a] += np.outer(x, x)
        self.b[a] += float(reward) * x
