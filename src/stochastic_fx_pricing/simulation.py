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
        n_trajectories = self.simulation_config.n_trajectories
        seed = self.simulation_config.seed

        timestamps = self._setup_time_grid()
        n_steps = len(timestamps)

        rng = np.random.default_rng(seed)

        rub_rate0 = 0.08
        rub_rates = np.column_stack(
            [
                np.full(n_trajectories, rub_rate0),
                rub_rate0
                + np.cumsum(
                    rng.normal(
                        loc=0.0, scale=0.0005, size=(n_trajectories, n_steps - 1)
                    ),
                    axis=1,
                ),
            ]
        )

        usd_rate0 = 0.02
        usd_rates = np.column_stack(
            [
                np.full(n_trajectories, usd_rate0),
                usd_rate0
                + np.cumsum(
                    rng.normal(
                        loc=0.0, scale=0.0005, size=(n_trajectories, n_steps - 1)
                    ),
                    axis=1,
                ),
            ]
        )

        increments = rng.normal(
            loc=0.0,
            scale=0.5,
            size=(n_trajectories, n_steps - 1),
        )

        fx_rates = np.column_stack(
            [np.full(n_trajectories, 80.0), 80 + np.cumsum(increments, axis=1)]
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
