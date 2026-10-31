from datetime import datetime

import numpy as np

from stochastic_fx_pricing.models import (
    RangeAccrualContract,
    SimulationConfig,
    Trajectories,
)


class SDESimulation:
    def __init__(
        self,
        range_accrual_contract: RangeAccrualContract,
        simulation_config: SimulationConfig,
    ):
        self.range_accrual_contract = range_accrual_contract
        self.simulation_config = simulation_config

    # TODO: Пока фейк симуляция для начала
    def run(self):
        """
        Основной метод для прогона симуляций
        """
        n_paths = self.simulation_config.n_trajectories
        timestamps = self._setup_time_grid()
        n_steps = len(timestamps)

        rng = np.random.default_rng(self.simulation_config.seed)

        rub_initial = 0.08
        usd_initial = 0.02
        fx_initial = 80.0

        kappa_rub, theta_rub, sigma_rub = 2.0, 0.08, 0.018
        kappa_usd, theta_usd, sigma_usd = 1.5, 0.025, 0.009

        sigma_fx = 0.25

        corr = np.array(
            [
                [1.0, 0.2, -0.3],
                [0.2, 1.0, 0.1],
                [-0.3, 0.1, 1.0],
            ]
        )

        chol = np.linalg.cholesky(corr)

        shape = (n_paths, n_steps)

        rub_rates = np.empty(shape)
        usd_rates = np.empty(shape)
        fx_rates = np.empty(shape)

        rub_rates[:, 0] = rub_initial
        usd_rates[:, 0] = usd_initial
        fx_rates[:, 0] = fx_initial

        seconds_in_year = 365 * 24 * 60 * 60

        for k in range(n_steps - 1):
            dt = (timestamps[k + 1] - timestamps[k]).total_seconds() / seconds_in_year

            sqrt_dt = np.sqrt(dt)

            z = rng.standard_normal((n_paths, 3)) @ chol.T

            rub = rub_rates[:, k]
            usd = usd_rates[:, k]
            fx = fx_rates[:, k]

            rub_rates[:, k + 1] = np.maximum(
                rub
                + kappa_rub * (theta_rub - rub) * dt
                + sigma_rub * sqrt_dt * z[:, 0],
                0.0,
            )

            usd_rates[:, k + 1] = np.maximum(
                usd
                + kappa_usd * (theta_usd - usd) * dt
                + sigma_usd * sqrt_dt * z[:, 1],
                0.0,
            )

            fx_rates[:, k + 1] = fx * np.exp(
                (rub - usd - 0.5 * sigma_fx**2) * dt + sigma_fx * sqrt_dt * z[:, 2]
            )

        return Trajectories(
            timestamps=timestamps,
            rub_rates=rub_rates,
            usd_rates=usd_rates,
            fx_rates=fx_rates,
        )

    def _setup_fixing_points(self) -> tuple[datetime, ...]:
        """
        Метод для формирования временных меток, когда мы будем
        фиксировать исходы по деривативу
        """
        contract = self.range_accrual_contract

        observations = []
        current = contract.start_date + contract.observation_step
        while current <= contract.end_date:
            observations.append(current)
            current += contract.observation_step

        return tuple(observations)

    def _setup_time_grid(self) -> tuple[datetime, ...]:
        """
        Метод для формирования вообще всех временных меток,
        которым мы хотим сопоставить значения процессов
        """
        contract = self.range_accrual_contract
        simulation_cfg = self.simulation_config

        start = contract.start_date
        end = contract.end_date

        timestamps = {start, end}
        current = start + simulation_cfg.step

        while current < end:
            timestamps.add(current)
            current += simulation_cfg.step

        timestamps.update(self._setup_fixing_points())

        return tuple(sorted(timestamps))
