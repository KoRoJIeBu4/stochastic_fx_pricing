from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class RangeAccrualContract:
    lower_bound: float | None
    upper_bound: float | None
    notional: float
    start_date: date
    end_date: date
    observation_step: timedelta

    def __post_init__(self):
        if self.notional <= 0:
            raise ValueError("Номинал должен быть положительным")
        if self.start_date >= self.end_date:
            raise ValueError(
                "Дата начала действия контракта должна "
                "быть строго меньше даты его окончания"
            )
        if self.observation_step <= timedelta(0):
            raise ValueError(
                "Длина временного шага наблюдения за состоянием "
                "контракта должны быть положительной"
            )
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("Нижняя граница должна быть не больше верхней")
        if self.lower_bound is None and self.upper_bound is None:
            raise ValueError("Нужно задать хотя бы одну границу")

    @classmethod
    def from_config(cls, config: dict):
        info = config["range_accrual_contract"]

        return cls(
            lower_bound=info["lower_bound"],
            upper_bound=info["upper_bound"],
            notional=info["notional"],
            start_date=date.fromisoformat(info["start_date"]),
            end_date=date.fromisoformat(info["end_date"]),
            observation_step=timedelta(**info["observation_step"]),
        )


@dataclass(frozen=True)
class SimulationConfig:
    step: timedelta
    n_paths: int
    seed: int | None

    def __post_init__(self):
        if self.step <= timedelta(0):
            raise ValueError(
                "Длина временного шага симуляции должна быть положительной"
            )
        if self.n_paths <= 0:
            raise ValueError("Число траекторий должно быть положительным")

    @classmethod
    def from_config(cls, config: dict):
        info = config["simulation"]

        return cls(
            step=timedelta(**info["step"]),
            n_paths=info["n_paths"],
            seed=info["seed"],
        )
