from datetime import datetime

import numpy as np

from stochastic_fx_pricing.config import load_models
from stochastic_fx_pricing.models import (
    MarketModelConfig,
    RangeAccrualContract,
    SimulationConfig,
    Trajectories,
)
from stochastic_fx_pricing.simulation import SDESimulation


class Pricer:
    def __init__(self):
        models = load_models()

        self.range_accrual_contract: RangeAccrualContract = models[
            "range_accrual_contract"
        ]
        self.simulation_config: SimulationConfig = models["simulation_config"]
        self.market_model_config: MarketModelConfig = models["market_model_config"]

        self.simulation = SDESimulation(
            range_accrual_contract=self.range_accrual_contract,
            simulation_config=self.simulation_config,
            market_model_config=self.market_model_config,
        )

    def get_fair_price(self) -> float:
        fixing_points: tuple[datetime, ...] = self.simulation._setup_fixing_points()

        trajectories: Trajectories = self.simulation.run()

        indecies = np.searchsorted(
            a=trajectories.timestamps,
            v=fixing_points,
        )

        payout = self._calc_payout(
            indecies=indecies,
            fx_rates=trajectories.fx_rates,
        )

        discount_factor = self._calc_discont_factor(
            rub_rates=trajectories.rub_rates,
            timestamps=trajectories.timestamps,
        )

        return float(np.mean(discount_factor * payout))

    def _calc_payout(
        self,
        indecies: np.ndarray,
        fx_rates: np.ndarray,
    ) -> np.ndarray:
        lower_bound = self.range_accrual_contract.lower_bound
        upper_bound = self.range_accrual_contract.upper_bound

        share = (
            (fx_rates[:, indecies] >= lower_bound)
            & (fx_rates[:, indecies] <= upper_bound)
        ).mean(axis=1)

        return share * self.range_accrual_contract.notional

    def _calc_discont_factor(
        self,
        rub_rates: np.ndarray,
        timestamps: tuple[datetime, ...],
    ) -> np.ndarray:
        seconds_in_year = 365 * 24 * 60 * 60

        dt = np.array(
            [
                (t_next - t_current).total_seconds() / seconds_in_year
                for t_next, t_current in zip(
                    timestamps[1:], timestamps[:-1], strict=False
                )
            ]
        )

        discount_factor = np.exp(-1 * np.sum(dt * rub_rates[:, :-1], axis=1))

        return discount_factor
