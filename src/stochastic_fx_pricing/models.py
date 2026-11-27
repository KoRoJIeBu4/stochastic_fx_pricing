from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RangeAccrualContract:
    lower_bound: float
    upper_bound: float
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
            self.lower_bound != -float("inf")
            and self.upper_bound != float("inf")
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("Нижняя граница должна быть не больше верхней")
        if self.lower_bound == -float("inf") and self.upper_bound == float("inf"):
            raise ValueError("Нужно задать хотя бы одну границу")
        if self.start_date + self.observation_step > self.end_date:
            raise ValueError(
                "За срок действия контракта нужно зафиксировать хотя бы одно состояние"
            )

    @classmethod
    def from_config(cls, config: dict):
        info = config["range_accrual_contract"]

        lower_bound = info.get("lower_bound", None)
        if lower_bound is None:
            lower_bound = -float("inf")

        upper_bound = info.get("upper_bound", None)
        if upper_bound is None:
            upper_bound = float("inf")

        return cls(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            notional=info["notional"],
            start_date=datetime.fromisoformat(info["start_date"]),
            end_date=datetime.fromisoformat(info["end_date"]),
            observation_step=timedelta(**info["observation_step"]),
        )


@dataclass(frozen=True)
class SimulationConfig:
    step: timedelta
    n_trajectories: int
    seed: Optional[int]
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


@dataclass(frozen=True)
class CIRParams:
    a: float
    b: float
    sigma: float
    start_value: float

    def __post_init__(self):
        values = (
            self.a,
            self.b,
            self.sigma,
            self.start_value,
        )

        if any(value is None for value in values):
            raise ValueError("Все значения в конфиге должны быть заполнены")

        if self.sigma < 0 or self.start_value <= 0 or self.a <= 0 or self.b <= 0:
            raise ValueError("Некорректные параметры")


@dataclass(frozen=True)
class FXParams:
    lmbda: float
    sigma: float
    start_value: float

    def __post_init__(self):
        values = (
            self.lmbda,
            self.sigma,
            self.start_value,
        )

        if any(value is None for value in values):
            raise ValueError("Все значения в конфиге должны быть заполнены")

        if self.sigma < 0 or self.start_value <= 0:
            raise ValueError("Некорректные параметры")


@dataclass(frozen=True)
class MarketModelConfig:
    rub_cir: CIRParams
    usd_cir: CIRParams
    fx: FXParams
    corr: np.ndarray

    @classmethod
    def from_config(cls, config: dict):
        market_models = config["market_models"]

        rub_cir = CIRParams(**market_models["rub_cir"])
        usd_cir = CIRParams(**market_models["usd_cir"])
        fx = FXParams(**market_models["fx"])

        # Будем придерживаться порядка [rub, usd, fx]
        rub_fx = market_models["corr"]["rub_fx"]
        rub_usd = market_models["corr"]["rub_usd"]
        fx_usd = market_models["corr"]["fx_usd"]

        corr = np.array(
            [
                [1, rub_usd, rub_fx],
                [rub_usd, 1, fx_usd],
                [rub_fx, fx_usd, 1],
            ]
        )

        return cls(
            rub_cir=rub_cir,
            usd_cir=usd_cir,
            fx=fx,
            corr=corr,
        )

    def __post_init__(self):
        matrix = np.array(
            self.corr,
            dtype=float,
            copy=True,
        )

        if not np.all(np.isfinite(matrix)):
            raise ValueError("Заполните корреляционную матрицу 3x3")

        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError("Матрица должна быть положительно определённой") from error
