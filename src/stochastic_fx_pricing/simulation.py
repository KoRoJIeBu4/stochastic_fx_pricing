from datetime import datetime

import numpy as np

from stochastic_fx_pricing.models import (
    MarketModelConfig,
    RangeAccrualContract,
    SimulationConfig,
    Trajectories,
)


class SDESimulation:
    def __init__(
        self,
        range_accrual_contract: RangeAccrualContract,
        simulation_config: SimulationConfig,
        market_model_config: MarketModelConfig,
    ):
        self.range_accrual_contract = range_accrual_contract
        self.simulation_config = simulation_config
        self.market_model_config = market_model_config

    def run(self, n_paths=None):
        """
        Основной метод для прогона симуляций
        """
        cfg = self.simulation_config
        model = self.market_model_config

        rub = model.rub_cir
        usd = model.usd_cir
        fx = model.fx

        timestamps = self._setup_time_grid()

        n_paths = cfg.n_trajectories if not n_paths else n_paths
        n_times = len(timestamps)
        rng = np.random.default_rng(cfg.seed)

        chol = np.linalg.cholesky(model.corr)

        rub_rates = np.empty((n_paths, n_times))
        usd_rates = np.empty((n_paths, n_times))
        fx_rates = np.empty((n_paths, n_times))

        rub_rates[:, 0] = rub.start_value
        usd_rates[:, 0] = usd.start_value
        fx_rates[:, 0] = fx.start_value

        seconds_in_year = 365 * 24 * 60 * 60

        for t in range(n_times - 1):
            dt = (timestamps[t + 1] - timestamps[t]).total_seconds() / seconds_in_year
            z = rng.normal(loc=0, scale=1, size=(n_paths, 3))
            dW = np.sqrt(dt) * (z @ chol.T)

            rub_current = rub_rates[:, t]
            usd_current = usd_rates[:, t]
            fx_current = fx_rates[:, t]

            rub_next = np.maximum(
                rub_current
                + dt * rub.a * (rub.b - rub_current)
                + rub.sigma * np.sqrt(rub_current) * dW[:, 0],
                0,
            )
            usd_next = np.maximum(
                usd_current
                + dt * usd.a * (usd.b - usd_current)
                + usd.sigma * np.sqrt(usd_current) * dW[:, 1],
                0,
            )
            fx_next = np.exp(
                np.log(fx_current)
                + dt * (rub_current - usd_current + fx.lmbda - 0.5 * fx.sigma**2)
                + fx.sigma * dW[:, 2]
            )

            rub_rates[:, t + 1] = rub_next
            usd_rates[:, t + 1] = usd_next
            fx_rates[:, t + 1] = fx_next

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
