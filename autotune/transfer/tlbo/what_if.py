import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis, cosine
from autotune.utils.extract_data import extract_data
from autotune.utils.normalization import normalize_y
from autotune.transfer.tlbo.base import BaseTLSurrogate
from autotune.utils.history_container import HistoryContainer
from autotune.utils.config_space.util import convert_configurations_to_array

def task_id(history_container:HistoryContainer):
    return history_container.task_id.replace("history_","")

class WhatIf(BaseTLSurrogate):
    def __init__(
        self,
        config_space,
        source_hpo_data,
        seed,
        cache=True,
        surrogate_type="rule_fit",
        num_src_hpo_trial=50,
        only_source=False,
    ) -> None:
        super().__init__(
            config_space,
            source_hpo_data,
            seed,
            surrogate_type=surrogate_type,
            num_src_hpo_trial=num_src_hpo_trial,
        )

        np.random.seed(seed)
        self.method_id = "what_if"
        self.only_source = only_source
        self.normalize = "standardize"
        self.dist_model = []
        self.model_weights = []
        self.opt_keys = []
        self.suid_keys = []
        self.iteration_id = 0
        self.workload_feature = None
        self.operator_info = "workload_feature.json"
        self.real_dist = None
        self._initialize(cache)

    def _initialize(self, cache=True):
        self._require_cache_model(cache=cache)
        # extract all feature data needed
        data = self._load_all_data()
        # train distance model to predict how two config close
        self.dist_model = self._get_distance_model(data)

    def _require_cache_model(self, cache_path="what_if_surrogates.pkl", cache=True):
        # build surrogate models
        self.surrogates_cache_path = cache_path
        if os.path.exists(self.surrogates_cache_path):
            with open(self.surrogates_cache_path, "rb") as f:
                self.source_surrogates = pickle.load(f)
        elif cache:
            self.build_source_surrogates(normalize=self.normalize)
            with open(self.surrogates_cache_path, "wb") as f:
                self.source_surrogates = pickle.dump(self.source_surrogates, f)

    def _load_real_dist(self):
        def concordant_pairs(y, y_pred):
            pairs = 0
            for i in range(len(y)):
                for j in range(len(y_pred)):
                    if (y[i] >= y[j] and y_pred[i] >= y_pred[j]) or (
                        y[i] <= y[j] and y_pred[i] <= y_pred[j]
                    ):
                        pairs += 1
            return pairs / (len(y) * len(y_pred))

        self.estimator_map = dict()

        for history_container in self.source_hpo_data:
            X, y, _, _ = extract_data(history_container)
            y = y.values.flatten()
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X, y)
            self.estimator_map[task_id(history_container)] = model
        dist = {}
        for history_container1 in self.source_hpo_data:
            tmp = {}
            task_1 = task_id(history_container1)
            for history_container2 in self.source_hpo_data:
                task_2 = task_id(history_container2)
                if task_1 == task_2:
                    continue
                X, y, _, _ = extract_data(history_container1)
                model = self.estimator_map[task_2]
                y_pred = model.predict(X)
                tmp[task_2] = concordant_pairs(y=y.to_numpy(), y_pred=y_pred)
            dist[task_1] = tmp
        return dist

    def _extract_pair_feature(
        self, history_container1, history_container2, true_dist=True
    ):
        def get_feature_v(workload_feature):
            return np.array(
                [workload_feature["SUID"][x] for x in self.suid_keys]
                + [workload_feature["OPTS"].get(x, 0) for x in self.opt_keys]
            )

        def combine_feature_v(feature1, feature2):
            feature_comb = np.hstack((feature1, feature2))
            eu_dist = np.linalg.norm(feature1 - feature2)
            cov_matrix = np.cov(np.vstack((feature1, feature2)), rowvar=False)
            inverse = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-5)
            ma_dist = mahalanobis(feature1, feature2, inverse)
            # Cosine
            co_dist = cosine(feature1, feature2)
            return feature_comb, eu_dist, ma_dist, co_dist
        
        task_1 = task_id(history_container1)
        task_2 = task_id(history_container2)
        assert self.workload_feature is not None
        feature_1_v = get_feature_v(self.workload_feature[task_1])
        feature_2_v = get_feature_v(self.workload_feature[task_2])
        feature_comb, eu, ma, co = combine_feature_v(
            feature1=feature_1_v, feature2=feature_2_v
        )
        res = pd.DataFrame([feature_comb])
        if true_dist:
            real_dist = 1 if task_1 == task_2 else self.real_dist[task_1][task_2]
            res["Real_Distance"] = np.array([real_dist])
        res["Euclidean"] = np.array([eu])
        res["Mahalanobis"] = np.array([ma])
        res["Cosine"] = np.array([co])
        return res

    def _load_all_data(self):
        def keys(data):
            suid_keys = ["INSERT", "SELECT", "DELETE", "UPDATE"]
            opts_keys = []
            for data_keys in data.values():
                opts = data_keys["OPTS"]
                opts_keys = set(opts.keys()) | set(opts_keys)
            return suid_keys, opts_keys

        with open(self.operator_info, "r") as f:
            self.workload_feature = json.load(f)
        self.real_dist = self._load_real_dist()

        self.suid_keys, self.opt_keys = keys(self.workload_feature)
        feature_df = pd.DataFrame()
        for history_container1 in self.source_hpo_data:
            for history_container2 in self.source_hpo_data:
                df_tmp = self._extract_pair_feature(
                    history_container1, history_container2
                )
                feature_df = pd.concat([feature_df, df_tmp], axis=0)
        return feature_df

    def _get_distance_model(self, df):
        X = df.drop(columns=["Real_Distance"], axis=1)
        y = df["Real_Distance"]
        model_rf = RandomForestRegressor(n_estimators=100, max_depth=10)
        model_rf.fit(X, y)
        return model_rf

    def _assign_weights(self, target_history: HistoryContainer):
        feature_df = pd.DataFrame()
        for source_history in self.source_hpo_data:
            df_tmp = self._extract_pair_feature(target_history, source_history, true_dist=False)
            feature_df = pd.concat([feature_df, df_tmp], axis=0)
        # feature_df = feature_df.drop(columns=["Real_Distance"])
        res = self.dist_model.predict(feature_df)
        res = res.flatten()
        threshold = np.percentile(res, 100 - 30)
        res[res < threshold] = 0
        return res / res.sum()

    def train(self, target_hpo_data: HistoryContainer, reweight=False):
        if len(self.model_weights) == 0 or reweight:
            self.model_weights = self._assign_weights(target_history=target_hpo_data)

        # if len(target_hpo_data.configurations) > 10:
        #     X = convert_configurations_to_array(target_hpo_data.configurations)
        #     y = target_hpo_data.get_transformed_perfs()

        #     # Currently Useless
        #     self.target_surrogate = self.build_single_surrogate(
        #         X, y, normalize=self.normalize
        #     )

    def simple_predict(self, X:np.ndarray):
        res = np.zeros(X.shape[0])
        for i in range(len(self.source_surrogates)):
            tmp_res = self.source_surrogates[i].predict(X).flatten()
            res += self.model_weights[i] * tmp_res
        return res.reshape(-1,1)

    def predict(self, X: np.ndarray):
        res = np.zeros(X.shape[0])
        for i in range(len(self.source_surrogates)):
            tmp_res = self.source_surrogates[i].predict(X).flatten()
            tmp_vot = np.zeros_like(tmp_res)
            tmp_vot[tmp_res == tmp_res.max()] = 1
            res += self.model_weights[i] * tmp_vot
        return res.reshape(-1,1)