import numpy as np
from autotune.transfer.tlbo.base import BaseTLSurrogate
from openbox.utils.config_space.util import convert_configurations_to_array
import networkx as nx


class CAUSAL(BaseTLSurrogate):
    def __init__(
        self,
        config_space,
        source_hpo_data,
        seed,
        surrogate_type="perf",
        num_src_hpo_trial=50,
        only_source=False,
        causal_graph=None,
    ):
        super().__init__(
            config_space,
            source_hpo_data,
            seed,
            surrogate_type=surrogate_type,
            num_src_hpo_trial=num_src_hpo_trial,
        )
        np.random.seed(seed)
        self.method_id = "causal"
        self.only_source = only_source
        self.build_source_surrogates(normalize="standardize")
        self.iteration_id = 0
        self._initialize()

    def _initialize(self):
        pass
    

    