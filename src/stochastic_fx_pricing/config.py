from pathlib import Path

from yaml import safe_load

from stochastic_fx_pricing.models import (
    MarketModelConfig,
    RangeAccrualContract,
    SimulationConfig,
)

current_file = Path(__file__).resolve()
config_path = current_file.parent.parent.parent / "configs" / "config.yaml"


def load_models() -> dict[
    str, RangeAccrualContract | SimulationConfig | MarketModelConfig
]:
    with open(config_path, "r") as config_file:
        config = safe_load(config_file)

        range_accrual_contract = RangeAccrualContract.from_config(
            config=config,
        )

        simulation_config = SimulationConfig.from_config(
            config=config,
        )

        market_model_config = MarketModelConfig.from_config(
            config=config,
        )

        return {
            "range_accrual_contract": range_accrual_contract,
            "simulation_config": simulation_config,
            "market_model_config": market_model_config,
        }
