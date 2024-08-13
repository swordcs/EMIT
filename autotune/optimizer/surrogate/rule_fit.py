import typing
import numpy as np

from functools import reduce
from ConfigSpace import ConfigurationSpace
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from autotune.optimizer.surrogate.base.base_model import AbstractModel
from autotune.utils.normalization import normalize_y


class RuleFit(AbstractModel):
    def __init__(
        self,
        configspace: ConfigurationSpace,
        types: typing.List[int],
        bounds: typing.List[typing.Tuple[float, float]],
        feature_names=None,
        tree_size=4,
        sample_fract="default",
        max_rules=2000,
        memory_par=0.01,
        exp_rand_tree_size=True,
        Cs=None,
        cv=3,
        tol=0.0001,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(
            types=types,
            bounds=bounds,
        )
        self.configspace = configspace
        self.lscv = LassoCV()
        self.feature_names = feature_names
        self.tree_generator = RandomForestRegressor(max_depth=2)
        self.tree_size = tree_size
        self.sample_fract = sample_fract
        self.max_rules = max_rules
        self.memory_par = memory_par
        self.exp_rand_tree_size = exp_rand_tree_size
        self.Cs = Cs
        self.cv = cv
        self.tol = tol
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_iter = 1000

    def _train(self, X: np.ndarray, y: np.ndarray):

        if not self.exp_rand_tree_size:
            self.tree_generator.fit(X, y)
        else:
            np.random.seed(self.random_state)
            tree_sizes = np.random.exponential(
                scale=self.tree_size - 2,
                size=int(np.ceil(self.max_rules * 2 / self.tree_size)),
            )
            tree_sizes = np.asarray(
                [2 + np.floor(tree_sizes[i_]) for i_ in np.arange(len(tree_sizes))],
                dtype=int,
            )
            i = int(len(tree_sizes) / 4)
            while np.sum(tree_sizes[0:i]) < self.max_rules:
                i = i + 1
            tree_sizes = tree_sizes[0:i]
            self.tree_generator.set_params(warm_start=True)
            for i_size in range(len(tree_sizes)):
                size = tree_sizes[i_size]
                self.tree_generator.set_params(n_estimators=i_size + 1)
                self.tree_generator.set_params(max_leaf_nodes=size)
                random_state_add = self.random_state if self.random_state else 0
                self.tree_generator.set_params(random_state=i_size + random_state_add)
                self.tree_generator.fit(np.copy(X, order="C"), np.copy(y, order="C"))
            self.tree_generator.set_params(warm_start=False)

        tree_list = [[x] for x in self.tree_generator.estimators_]
        self.rule_ensemble = RuleEnsemble(
            tree_list=tree_list, feature_names=self.feature_names
        )
        X_rules = self.rule_ensemble.transform(X, None)
        X_concat = np.zeros([X.shape[0], 0])

        if X_rules.shape[0] > 0:
            X_concat = np.concatenate((X_concat, X_rules), axis=1)
        if self.Cs is None:
            n_alphas = 100
            alphas = None
        elif hasattr(self.Cs, "__len__"):
            n_alphas = None
            alphas = 1.0 / self.Cs
        else:
            n_alphas = self.Cs
            alphas = None
        self.lscv = LassoCV(
            n_alphas=n_alphas,
            alphas=alphas,
            cv=self.cv,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=False,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.lscv.fit(X_concat, y)
        self.coef_ = self.lscv.coef_
        self.intercept_ = self.lscv.intercept_

        return self

    def _rule_stats(self):
        res = {}
        res["total_rules"] = len(self.coef_)
        res["useful_rules"] = np.count_nonzero(self.coef_)
        return res

    def predict(self, X: np.ndarray):
        X_concat = np.zeros([X.shape[0], 0])
        rule_coefs = self.coef_[-len(self.rule_ensemble.rules) :]
        if len(rule_coefs) > 0:
            X_rules = self.rule_ensemble.transform(X, coefs=rule_coefs)
            if X_rules.shape[0] > 0:
                X_concat = np.concatenate((X_concat, X_rules), axis=1)
        return self.lscv.predict(X_concat)

    def _rules_hit(self, X):
        X_rules = self.rule_ensemble.transform(X, coefs=None)
        rules = []
        for i in range(len(X)):
            tmp = []
            for j in range(len(self.rule_ensemble.rules)):
                if X_rules[i][j] <= 0:
                    continue
                tmp.append((self.rule_ensemble.rules[j], self.coef_[j]))
            rules.append(sorted(tmp, key=lambda x: -abs(x[1])))
        return rules


class RuleCondition:
    def __init__(self, feature_index, threshold, operator, support, feature_name=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.operator = operator
        self.support = support
        self.feature_name = feature_name

    def __repr__(self):
        if self.feature_name:
            feature = self.feature_name
        else:
            feature = self.feature_index
        return "%s %s %.2f" % (feature, self.operator, self.threshold)

    def __str__(self):
        return self.__repr__()

    def transform(self, X):
        if self.operator == "<=":
            res = 1 * (X[:, self.feature_index] <= self.threshold)
        elif self.operator == ">":
            res = 1 * (X[:, self.feature_index] > self.threshold)
        return res

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(
            (self.feature_index, self.threshold, self.operator, self.feature_name)
        )


class Rule:
    def __init__(self, rule_conditions, prediction_value):
        self.support = min([x.support for x in rule_conditions])
        self.conditions = set(rule_conditions)
        self.prediction_value = prediction_value
        self.rule_direction = None

    def transform(self, X):
        rule_applies = [condition.transform(X) for condition in self.conditions]
        return reduce(lambda x, y: x * y, rule_applies)

    def __str__(self):
        return " & ".join([x.__str__() for x in self.conditions])

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return sum([condition.__hash__() for condition in self.conditions])

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


def extract_rules_from_tree(tree, feature_names=None):
    rules = set()

    def traverse_nodes(
        node_id=0, operator=None, threshold=None, feature=None, conditions=[]
    ):
        if node_id != 0:
            if feature_names is not None:
                feature_name = feature_names[feature]
            else:
                feature_name = feature
            rule_condition = RuleCondition(
                feature_index=feature,
                threshold=threshold,
                operator=operator,
                support=tree.n_node_samples[node_id] / float(tree.n_node_samples[0]),
                feature_name=feature_name,
            )
            new_conditions = conditions + [rule_condition]
        else:
            new_conditions = []
        if tree.children_left[node_id] != tree.children_right[node_id]:
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]

            left_node_id = tree.children_left[node_id]
            traverse_nodes(left_node_id, "<=", threshold, feature, new_conditions)

            right_node_id = tree.children_right[node_id]
            traverse_nodes(right_node_id, ">", threshold, feature, new_conditions)
        elif len(new_conditions) > 0:
            new_rule = Rule(new_conditions, tree.value[node_id][0][0])
            rules.update([new_rule])
            return None

    traverse_nodes()

    return rules


class RuleEnsemble:
    def __init__(self, tree_list, feature_names=None):
        self.tree_list = tree_list
        self.feature_names = feature_names
        self.rules = set()
        self._extract_rules()
        self.rules = list(self.rules)

    def _extract_rules(self):
        for tree in self.tree_list:
            rules = extract_rules_from_tree(
                tree[0].tree_, feature_names=self.feature_names
            )
            self.rules.update(rules)

    def filter_rules(self, func):
        self.rules = filter(lambda x: func(x), self.rules)

    def filter_short_rules(self, k):
        self.filter_rules(lambda x: len(x.conditions) > k)

    def transform(self, X, coefs=None):
        rule_list = list(self.rules)
        if coefs is None:
            return np.array([rule.transform(X) for rule in rule_list]).T
        else:
            res = np.array(
                [
                    rule_list[i_rule].transform(X)
                    for i_rule in np.arange(len(rule_list))
                    if coefs[i_rule] != 0
                ]
            ).T
            res_ = np.zeros([X.shape[0], len(rule_list)])
            res_[:, coefs != 0] = res
            return res_

    def __str__(self):
        return (map(lambda x: x.__str__(), self.rules)).__str__()
