import logging
import typing
import numpy as np

from functools import reduce
from ConfigSpace import ConfigurationSpace
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from autotune.optimizer.surrogate.base.base_model import AbstractModel

logger = logging.getLogger(__name__)


class Hybrid(AbstractModel):
    def __init__(
        self,
        configspace: ConfigurationSpace,
        types: typing.List[int],
        bounds: typing.List[typing.Tuple[float, float]],
        ensemble_size: int = 10,
        normalize_y: bool = True,
        decay: float = 0.05,
        seed: int = 42,
    ):
        super().__init__(types=types, bounds=bounds)
        self.configspace = configspace
        
        pass
