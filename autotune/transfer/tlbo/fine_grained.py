import os
import pickle
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
from autotune.transfer.tlbo.base import BaseTLSurrogate
from autotune.optimizer.surrogate.core import build_surrogate
from openbox.utils.config_space.util import convert_configurations_to_array

from autotune.utils.history_container import HistoryContainer
from autotune.utils.normalization import (
    normalize_y,
    zero_mean_unit_var_normalization,
    zero_one_normalization,
)

NUMERIC_METRICS = [  # counter
    # global
    "buffers_alloc",
    "buffers_backend",
    "buffers_backend_fsync",
    "buffers_checkpoint",
    "buffers_clean",
    "checkpoints_req",
    "checkpoints_timed",
    "checkpoint_sync_time",
    "checkpoint_write_time",
    "maxwritten_clean",
    "archived_count",
    "failed_count",
    # db
    "blk_read_time",
    "blks_hit",
    "blks_read",
    "blk_write_time",
    "conflicts",
    "deadlocks",
    "temp_bytes",
    "temp_files",
    "tup_deleted",
    "tup_fetched",
    "tup_inserted",
    "tup_returned",
    "tup_updated",
    "xact_commit",
    "xact_rollback",
    "confl_tablespace",
    "confl_lock",
    "confl_snapshot",
    "confl_bufferpin",
    "confl_deadlock",
    # table
    "analyze_count",
    "autoanalyze_count",
    "autovacuum_count",
    "heap_blks_hit",
    "heap_blks_read",
    "idx_blks_hit",
    "idx_blks_read",
    "idx_scan",
    "idx_tup_fetch",
    "n_dead_tup",
    "n_live_tup",
    "n_tup_del",
    "n_tup_hot_upd",
    "n_tup_ins",
    "n_tup_upd",
    "n_mod_since_analyze",
    "seq_scan",
    "seq_tup_read",
    "tidx_blks_hit",
    "tidx_blks_read",
    "toast_blks_hit",
    "toast_blks_read",
    "vacuum_count",
    # index
    "idx_blks_hit",
    "idx_blks_read",
    "idx_scan",
    "idx_tup_fetch",
    "idx_tup_read",
]


class FineGrained(BaseTLSurrogate):
    def __init__(
        self,
        config_space,
        source_hpo_data,
        seed,
        surrogate_type="prf",
        num_src_hpo_trial=50,
        only_source=False,
        good_perf_threshold=50,
    ):
        super().__init__(
            config_space,
            source_hpo_data,
            seed,
            surrogate_type=surrogate_type,
            num_src_hpo_trial=num_src_hpo_trial,
        )
        np.random.seed(seed)
        self.surrogate_type = "prf"
        self.mothod_id = "fine_grained"
        self.normalize = "standardize"
        self.history_model = []
        self.only_source = only_source
        # To decide whether a performance is good
        self.good_perf_threshold = good_perf_threshold
        self.current_context = None
        self.iteration_id = 0
        self.config_evaluated = 0
        self.estimator_list = None
        self.cold_start = 10
        self.use_hybrid = False
        self.hist_weight = None
        self.decay = 0.05
        self._initialize()

    def _initialize(self):
        self.build_source_surrogates(normalize=self.normalize)
        self._extract_common_good_space()

    def _extract_common_good_space(self):
        estimator_list = []

        for history_container in self.source_hpo_data:
            X = convert_configurations_to_array(history_container.configurations)
            y = history_container.get_transformed_perfs()

            y_threshold = np.percentile(y, self.good_perf_threshold)
            y_label = np.zeros(y.shape)
            y_label[y < y_threshold] = 1
            # dtc = ensemble.RandomForestClassifier()
            dtc = SVC()
            dtc.fit(X, y_label)
            estimator_list.append(dtc)
        self.estimator_list = estimator_list

    def _predict_history(self, X):
        vote_res = np.zeros(X.shape[0]).reshape(-1, 1)
        var_res = np.zeros(X.shape[0]).reshape(-1, 1)
        perf_res = np.zeros(X.shape[0]).reshape(-1, 1)
        for i in range(self.K):
            if self.hist_weight[i] <= 0:
                continue
            votes = self.estimator_list[i].predict(X).reshape(-1, 1)
            vote_res += votes * self.hist_weight[i]

            perfs, vars = self.source_surrogates[i].predict(X)
            var_res  += vars * self.hist_weight[i]
            perf_res += perfs * self.hist_weight[i]

        return perf_res, var_res, vote_res

    def train(self, target_hpo_data: HistoryContainer, p=30, exp=5):
        # assign weights
        weights = []
        for history_container in self.source_hpo_data:
            weights.append(self._hpo_similarity(target_hpo_data, history_container))
        weights = np.array(weights)
        weights[weights < np.percentile(weights, 100 - p)] = 0
        weights, _, _ = zero_one_normalization(np.array(weights) ** exp)
        self.hist_weight = weights / sum(weights)
        self.config_evaluated = target_hpo_data.config_counter
        if self.config_evaluated >= self.cold_start:
            X, y, _, _ = self._hc_to_dataframe(target_hpo_data)
            self.target_surrogate = self.build_single_surrogate(
                X.to_numpy(), y.to_numpy(), normalize="scale"
            )
            self.use_hybrid = True
        self.iteration_id += 1
    
    def _hpo_similarity(
        self, history_container1: HistoryContainer, history_container2: HistoryContainer
    ):
        _, _, internal_metrics1, resource1 = self._hc_to_dataframe(history_container1)
        _, _, internal_metrics2, resource2 = self._hc_to_dataframe(history_container2)
        metrics1 = pd.concat([internal_metrics1, resource1], axis=1).mean()
        metrics2 = pd.concat([internal_metrics2, resource2], axis=1).mean()

        distance = np.sum(metrics1 * metrics2) / np.sqrt(
            np.sum(metrics1**2) * np.sum(metrics2**2)
        )
        return distance

    def _hc_to_dataframe(self, history_container: HistoryContainer, normalize="scale"):
        X = convert_configurations_to_array(history_container.configurations)
        y = history_container.get_transformed_perfs()
        resource = history_container.resource
        internal_metrics = history_container.internal_metrics

        # drop invalid data
        drop_indexes = set()
        for i in range(len(X)):
            if len(internal_metrics[i]) == 0:
                drop_indexes.add(i)
        X = pd.DataFrame(
            [item for idx, item in enumerate(X) if idx not in drop_indexes]
        )
        y = pd.DataFrame(
            [item for idx, item in enumerate(y) if idx not in drop_indexes]
        )
        y = normalize_y(y=y.to_numpy(), normalize=self.normalize)
        # untransform to df
        y = pd.DataFrame(y)

        resource_metric = resource[0].keys()
        resource = pd.DataFrame(
            [
                [item[k] for k in resource_metric]
                for idx, item in enumerate(resource)
                if idx not in drop_indexes
            ]
        )
        internal_metrics = pd.DataFrame(
            [
                item
                for idx, item in enumerate(internal_metrics)
                if idx not in drop_indexes
            ]
        )

        X_col = [e.name for e in history_container.config_space.get_hyperparameters()]
        y_col = history_container.info["objs"]
        internal_metrics_col = NUMERIC_METRICS
        resource_col = [k for k in resource_metric]

        X.columns = X_col
        y.columns = y_col
        resource.columns = resource_col
        internal_metrics.columns = internal_metrics_col

        return X, y, internal_metrics, resource

    def predict(self, X: np.ndarray):
        res_transfer = self._predict_history(X)
        shape = res_transfer[0].shape
        if not self.use_hybrid:
            return np.zeros(shape), np.zeros(shape), res_transfer[0], res_transfer[1], res_transfer[2], 1. 
        beta = max(1 - (self.config_evaluated - self.cold_start) * self.decay, 0)
        res = self.target_surrogate.predict(X)
        return res[0], res[1], res_transfer[0], res_transfer[1], res_transfer[2], beta

    @staticmethod
    def _workload_shift(data1: np.array, data2: np.array, threshold=0.5) -> bool:
        assert len(data1) == len(data2)
        data = pd.concat([data1, data2], axis=0)
        # decompose the data
        data = PCA(n_components=10).fit_transform(data)
        _, labels, _ = k_means(X=data, n_clusters=2)
        res1 = {}
        res2 = {}
        for i in range(len(data1)):
            res1[labels[i]] = res1.get(labels[i], 0) + 1
        for i in range(len(data1), len(data)):
            res2[labels[i]] = res2.get(labels[i], 0) + 1

        label1, label2 = res1.keys()[0], res1.keys()[1]
        q1 = res1.get(label1, 0) + res2.get(label2, 0)
        q2 = res1.get(label2, 0) + res2.get(label1, 0)

        return max(q1, q2) > (len(data1) + len(data2)) * threshold
