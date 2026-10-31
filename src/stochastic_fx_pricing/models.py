from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np


@dataclass(frozen=True)
class RangeAccrualContract:
    lower_bound: float | None
    upper_bound: float | None
    notional: float
    start_date: datetime
    end_date: datetime
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
            start_date=datetime.fromisoformat(info["start_date"]),
            end_date=datetime.fromisoformat(info["end_date"]),
            observation_step=timedelta(**info["observation_step"]),
        )


@dataclass(frozen=True)
class SimulationConfig:
    step: timedelta
    n_trajectories: int
    seed: int
    max_plot_trajectories: int

    def __post_init__(self):
        if self.step <= timedelta(0):
            raise ValueError(
                "Длина временного шага симуляции должна быть положительной"
            )
        if self.n_trajectories <= 0:
            raise ValueError("Число траекторий должно быть положительным")

    @classmethod
    def from_config(cls, config: dict):
        info = config["simulation"]

        return cls(
            step=timedelta(**info["step"]),
            n_trajectories=info["n_trajectories"],
            seed=info["seed"],
            max_plot_trajectories=info["max_plot_trajectories"],
        )


@dataclass(frozen=True)
class Trajectories:
    timestamps: tuple[datetime, ...]
    rub_rates: np.ndarray
    usd_rates: np.ndarray
    fx_rates: np.ndarray

    def __post_init__(self):
        shape = self.fx_rates.shape

        if len(shape) != 2:
            raise ValueError("Массив траекторий должен быть двумерным")
        if shape[0] == 0 or shape[1] != len(self.timestamps):
            raise ValueError("Размер массива не соотвествует временной сетке")
        if self.rub_rates.shape != shape or self.usd_rates.shape != shape:
            raise ValueError("Размерности всех трех массивов должны быть согласованны")
        if any(
            t1 >= t2
            for t1, t2 in zip(self.timestamps, self.timestamps[1:], strict=False)
        ):
            raise ValueError("Временные метки должны строго возрастать")
