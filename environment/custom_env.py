from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.rendering import MaternalChildHealthRenderer


@dataclass
class MissionConfig:
    max_days: int = 180
    initial_budget: float = 130.0
    random_event_rate: float = 0.08
    severe_weather_rate: float = 0.03
    focused_mnh_mode: bool = False


class MaternalChildHealthMissionEnv(gym.Env):
    """
    Mission-based RL environment for maternal and child health operations.

    Mission objective:
    Use limited budget and workforce actions to minimize maternal/neonatal risk,
    avoid preventable adverse outcomes, and improve district service coverage.

    Action space (Discrete(8)):
    0 -> deploy_mobile_antenatal_clinic
    1 -> emergency_referral_transport
    2 -> nutrition_supplement_distribution
    3 -> routine_immunization_campaign
    4 -> community_health_worker_training
    5 -> medicine_and_blood_restock
    6 -> remote_telemedicine_consultation
    7 -> hold_and_monitor (do nothing)

    Observation space (default Box(12,)):
    [0] maternal_risk_index        in [0,1]
    [1] neonatal_risk_index        in [0,1]
    [2] immunization_gap           in [0,1]
    [3] nutrition_deficit          in [0,1]
    [4] referral_backlog           in [0,1]
    [5] medical_supply_stress      in [0,1]
    [6] community_trust            in [0,1]
    [7] weather_severity           in [0,1]
    [8] staff_fatigue              in [0,1]
    [9] budget_remaining_norm      in [0,1]
    [10] day_progress              in [0,1]
    [11] recent_shock_flag         in [0,1]

    Focused maternal-neonatal mode (config.focused_mnh_mode=True):
    Observation space is Box(6,) with only core mission indicators:
    [0] maternal_risk_index
    [1] neonatal_risk_index
    [2] referral_backlog
    [3] medical_supply_stress
    [4] community_trust
    [5] budget_remaining_norm
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    ACTIONS = {
        0: "deploy_mobile_antenatal_clinic",
        1: "emergency_referral_transport",
        2: "nutrition_supplement_distribution",
        3: "routine_immunization_campaign",
        4: "community_health_worker_training",
        5: "medicine_and_blood_restock",
        6: "remote_telemedicine_consultation",
        7: "hold_and_monitor",
    }

    ACTION_COST = np.array([6.0, 8.0, 5.0, 5.5, 4.0, 7.0, 3.0, 1.2], dtype=np.float32)

    def __init__(
        self,
        config: Optional[MissionConfig] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config or MissionConfig()
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.focused_mnh_mode = bool(self.config.focused_mnh_mode)

        obs_dim = 6 if self.focused_mnh_mode else 12
        self.observation_space = spaces.Box(
            low=np.zeros(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.renderer: Optional[MaternalChildHealthRenderer] = None
        self.render_fps = int(self.metadata["render_fps"])
        self.state = np.zeros(12, dtype=np.float32)
        self.day = 0
        self.total_reward = 0.0
        self.preventable_adverse_events = 0
        self.last_action_name = "-"
        self.last_reward = 0.0
        self.risk_burden_history: list[float] = []
        self.reward_history: list[float] = []

    def _observe(self) -> np.ndarray:
        if not self.focused_mnh_mode:
            return self.state.copy()

        # Focus only on core maternal/neonatal operations signals.
        # Full state indices: maternal(0), neonatal(1), referral(4), supply(5), trust(6), budget(9)
        return self.state[[0, 1, 4, 5, 6, 9]].astype(np.float32)

    def set_render_fps(self, fps: int) -> None:
        self.render_fps = max(1, int(fps))
        if self.renderer is not None:
            self.renderer.set_fps(self.render_fps)

    def _sample_initial_state(self) -> np.ndarray:
        base = np.array(
            [
                0.52,  # maternal risk
                0.48,  # neonatal risk
                0.45,  # immunization gap
                0.44,  # nutrition deficit
                0.32,  # referral backlog
                0.40,  # supply stress
                0.55,  # community trust
                0.20,  # weather severity
                0.24,  # staff fatigue
                1.00,  # budget norm
                0.00,  # day progress
                0.00,  # recent shock flag
            ],
            dtype=np.float32,
        )
        noise = self.rng.normal(0.0, 0.05, size=12).astype(np.float32)
        noise[9] = 0.0
        noise[10] = 0.0
        noise[11] = 0.0
        return np.clip(base + noise, 0.0, 1.0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.day = 0
        self.total_reward = 0.0
        self.preventable_adverse_events = 0
        self.last_action_name = "-"
        self.last_reward = 0.0
        self.risk_burden_history = []
        self.reward_history = []
        self.state = self._sample_initial_state()

        if self.render_mode == "human" and self.renderer is None:
            self.renderer = MaternalChildHealthRenderer(fps=self.render_fps)

        info = {
            "action_names": self.ACTIONS,
            "mission": "maternal_child_health",
            "budget": float(self.config.initial_budget),
            "focused_mnh_mode": self.focused_mnh_mode,
        }
        return self._observe(), info

    def _apply_action_effects(self, action: int) -> None:
        # local aliases
        m, n, i, nu, r, s, t, w, f, b, dp, shock = self.state

        # baseline drift (without intervention) - slightly worsening risks with fatigue/backlog
        m += 0.006 + 0.01 * f + 0.005 * r
        n += 0.005 + 0.008 * nu + 0.005 * s
        i += 0.004
        nu += 0.004
        r += 0.003 + 0.004 * w
        s += 0.003
        t -= 0.002
        f += 0.004

        # action-specific effects
        if action == 0:  # mobile antenatal clinic
            m -= 0.06
            n -= 0.02
            r -= 0.03
            t += 0.02
            f += 0.02
        elif action == 1:  # emergency referral transport
            m -= 0.05
            n -= 0.05
            r -= 0.08
            s += 0.01
            f += 0.03
        elif action == 2:  # nutrition supplementation
            nu -= 0.08
            n -= 0.03
            t += 0.01
        elif action == 3:  # immunization campaign
            i -= 0.10
            n -= 0.02
            r += 0.01
            f += 0.01
            t += 0.015
        elif action == 4:  # CHW training
            m -= 0.03
            n -= 0.02
            r -= 0.02
            t += 0.03
            f -= 0.02
        elif action == 5:  # restock
            s -= 0.10
            m -= 0.02
            n -= 0.02
            t += 0.01
        elif action == 6:  # telemedicine
            m -= 0.02
            n -= 0.02
            r -= 0.02
            t += 0.015
            f -= 0.01
        elif action == 7:  # hold and monitor
            f -= 0.03
            t -= 0.005

        # stochastic shocks
        shock = 0.0
        if self.rng.random() < self.config.random_event_rate:
            event_strength = float(self.rng.uniform(0.03, 0.08))
            m += event_strength
            n += event_strength * 0.9
            r += event_strength * 0.7
            nu += event_strength * 0.5
            shock = 1.0

        if self.rng.random() < self.config.severe_weather_rate:
            weather_pulse = float(self.rng.uniform(0.15, 0.35))
            w = min(1.0, w + weather_pulse)
            r += 0.08
            s += 0.04
            shock = 1.0
        else:
            w = max(0.0, w - 0.02)

        # budget dynamics
        remaining_budget = max(0.0, b * self.config.initial_budget - float(self.ACTION_COST[action]))
        b = remaining_budget / self.config.initial_budget

        # increasing risk under severe budget depletion
        if b < 0.15:
            m += 0.02
            n += 0.02
            s += 0.03
            t -= 0.02

        # temporal progress
        self.day += 1
        dp = self.day / self.config.max_days

        self.state = np.clip(np.array([m, n, i, nu, r, s, t, w, f, b, dp, shock], dtype=np.float32), 0.0, 1.0)

    def _compute_reward(self, prev_state: np.ndarray, action: int) -> float:
        cur = self.state

        prev_risk_burden = float(prev_state[0] + prev_state[1] + prev_state[2] + prev_state[3] + prev_state[4] + prev_state[5])
        cur_risk_burden = float(cur[0] + cur[1] + cur[2] + cur[3] + cur[4] + cur[5])

        risk_improvement = prev_risk_burden - cur_risk_burden
        trust_bonus = float(cur[6] - 0.5) * 0.8
        fatigue_penalty = float(cur[8]) * 0.6
        cost_penalty = float(self.ACTION_COST[action]) * 0.05
        shock_penalty = float(cur[11]) * 0.8

        preventable_event = 1 if (cur[0] > 0.8 or cur[1] > 0.8) and cur[4] > 0.65 else 0
        if preventable_event:
            self.preventable_adverse_events += 1

        event_penalty = 2.5 * preventable_event

        reward = 3.2 * risk_improvement + trust_bonus - fatigue_penalty - cost_penalty - shock_penalty - event_penalty

        # mission success shaping
        if cur[0] < 0.25 and cur[1] < 0.25 and cur[2] < 0.2:
            reward += 2.0

        return float(reward)

    def _is_terminated(self) -> bool:
        m, n, i, nu, r, s, t, w, f, b, dp, shock = self.state

        # catastrophic outcome
        if (m > 0.95 and n > 0.9) or self.preventable_adverse_events >= 4:
            return True

        # successful mission state
        if m < 0.2 and n < 0.2 and i < 0.18 and nu < 0.2 and r < 0.18:
            return True

        # budget collapse
        if b <= 0.01:
            return True

        # max duration
        return self.day >= self.config.max_days

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        prev_state = self.state.copy()
        self._apply_action_effects(action)
        reward = self._compute_reward(prev_state, action)
        self.total_reward += reward
        self.last_action_name = self.ACTIONS[int(action)]
        self.last_reward = float(reward)

        risk_burden = float(self.state[0] + self.state[1] + self.state[2] + self.state[3] + self.state[4] + self.state[5])
        self.risk_burden_history.append(risk_burden)
        self.reward_history.append(float(reward))

        terminated = self._is_terminated()
        truncated = self.day >= self.config.max_days

        info = {
            "day": self.day,
            "action_name": self.ACTIONS[int(action)],
            "total_reward": float(self.total_reward),
            "preventable_adverse_events": int(self.preventable_adverse_events),
            "supply_explainer": "medical_supply_stress tracks stockout pressure for medicines/blood/PPE; lower is better",
        }
        return self._observe(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode is None:
            return None

        if self.renderer is None:
            self.renderer = MaternalChildHealthRenderer(fps=self.render_fps)

        frame = self.renderer.draw(
            state=self.state,
            day=self.day,
            max_days=self.config.max_days,
            total_reward=self.total_reward,
            last_action=self.last_action_name,
            last_reward=self.last_reward,
            risk_trend=self.risk_burden_history,
            reward_trend=self.reward_history,
        )

        if self.render_mode == "human":
            self.renderer.show(frame)
            return None

        if self.render_mode == "rgb_array":
            return frame

        raise NotImplementedError(f"Unsupported render_mode {self.render_mode}")

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
