from __future__ import annotations

import numpy as np

from .base import BasePolicy, Decision


class LinUCBPolicy(BasePolicy):
    """Linear UCB (Li et al., WWW 2010) со softmax-обёрткой для OPE.

    Состояние для каждого действия a:
        A[a] : (d, d) матрица, инициализируется reg * I
        b[a] : (d,) вектор, инициализируется нулями
        theta[a] = A[a]^{-1} b[a]

    Скор:
        score[a] = theta[a]^T x + alpha * sqrt(x^T A[a]^{-1} x)

    Для OPE используется softmax(score / temperature): это даёт строго
    положительные вероятности для всех действий (common support) и
    корректный propensity выбранного действия.
    """

    name = "linucb"

    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        alpha: float = 1.0,
        temperature: float = 1.0,
        ope_temperature: float | None = None,
        reg: float = 1.0,
        seed: int | None = None,
        whitelist: np.ndarray | None = None,
        blacklist: np.ndarray | None = None,
    ) -> None:
        super().__init__(n_actions=n_actions, seed=seed)
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if ope_temperature is not None and ope_temperature <= 0:
            raise ValueError("ope_temperature must be > 0")
        if reg <= 0:
            raise ValueError("reg must be > 0")

        self.context_dim = int(context_dim)
        self.alpha = float(alpha)
        self.temperature = float(temperature)                                            
        self.ope_temperature = (                                                      
            float(ope_temperature) if ope_temperature is not None
            else float(temperature)
        )
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

    def _ucb_scores(self, context_batch: np.ndarray) -> np.ndarray:
        """context_batch: (n, d) -> (n, n_actions) UCB-скоров.

        Численно устойчиво: используем np.linalg.solve вместо явного inv,
        где это возможно. Для бонуса нужна квадратичная форма x^T A^{-1} x,
        её считаем через solve(A, x) = A^{-1} x.
        """
        n, d = context_batch.shape
        if d != self.context_dim:
            raise ValueError(
                f"context has dim {d}, policy expects {self.context_dim}"
            )

        scores = np.zeros((n, self.n_actions), dtype=float)
        for a in range(self.n_actions):
            A_a = self.A[a]                                        
            b_a = self.b[a]                                      

                                    
            theta_a = np.linalg.solve(A_a, b_a)                  

                                         
            mean = context_batch @ theta_a

                                                                  
            A_inv_x = np.linalg.solve(A_a, context_batch.T).T

                                                            
            quad = np.einsum("nd,nd->n", context_batch, A_inv_x)
            bonus = self.alpha * np.sqrt(np.maximum(quad, 0.0))                  

            scores[:, a] = mean + bonus
        return scores

    @staticmethod
    def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
        z = scores / temperature
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

                                                      

    def action_distribution(
        self,
        context: np.ndarray,
        base_scores: np.ndarray,
        candidate_items: np.ndarray | None = None,
    ) -> np.ndarray:
        """OPE-friendly распределение (использует ope_temperature).

        Используется как evaluation_action_dist в build_ope_input.
        При ope_temperature > temperature даёт более гладкие вероятности,
        что улучшает ESS и стабилизирует веса.
        """
        single_round = context.ndim == 1
        x = context[None, :] if single_round else context

        ucb = self._ucb_scores(x)
        allowed = self._allowed_mask(candidate_items)
        if not allowed.any():
            raise ValueError("No allowed candidates for this policy")
        ucb = np.where(allowed[None, :], ucb, -np.inf)

        dist = self._softmax(ucb, self.ope_temperature)
        return dist[0] if single_round else dist

    def select_action(
        self,
        context: np.ndarray,
        base_scores: np.ndarray,
        candidate_items: np.ndarray | None = None,
    ) -> Decision:
        """Выбирает действие с self.temperature (как политика действует).

        Decision.propensity и Decision.probabilities отражают РЕАЛЬНОЕ
        распределение выбора (self.temperature), а не OPE-распределение
        (self.ope_temperature). Это делает логи честными.
        """
        self._last_context = np.array(context, copy=True)

        x = np.asarray(context, dtype=float).reshape(-1)
        if x.shape[0] != self.context_dim:
            raise ValueError(
                f"context has dim {x.shape[0]}, policy expects {self.context_dim}"
            )

        allowed = self._allowed_mask(candidate_items)
        if not allowed.any():
            raise ValueError("No allowed candidates for this policy")

        ucb = self._ucb_scores(x[None, :])[0]
        ucb_masked = np.where(allowed, ucb, -np.inf)

        select_dist = self._softmax(ucb_masked, self.temperature)
        chosen = int(self.rng.choice(self.n_actions, p=select_dist))

        greedy = int(np.argmax(ucb_masked))

        return Decision(
            chosen_item=chosen,
            propensity=float(select_dist[chosen]),
            probabilities=select_dist,
            is_exploration=(chosen != greedy),
            policy_name=self.name,
        )

    def update(
        self,
        decision: Decision,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        """Обновление состояния для выбранного действия.

        Если context не передан, используется последний контекст из
        select_action (работает для пошагового режима).
        """
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
