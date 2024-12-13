import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class LinearRegressionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1, n_quantiles=10):
        if np.any([ft not in ["real", "categorical"] for ft in feature_types]):
            raise ValueError("There is an unknown feature type")
        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._n_quantiles = n_quantiles

    def _fit_node(self, sub_X, sub_y, node, cur_depth):
        if len(sub_y) < self._min_samples_split or (self._max_depth and cur_depth >= self._max_depth):
            node["type"] = "terminal"
            model = LinearRegression()
            model.fit(sub_X, sub_y)
            node["model"] = model
            return

        best_feature, best_threshold, best_error, best_split = None, None, float('inf'), None
        n_samples, n_features = sub_X.shape

        for feature in range(n_features):
            feature_type = self._feature_types[feature]
            feature_vector = sub_X[:, feature]

            if feature_type == "real":
                thresholds = np.quantile(np.unique(feature_vector), np.linspace(0, 1, self._n_quantiles + 2)[1:-1])
            else:
                continue

            for threshold in thresholds:
                left_indices = feature_vector < threshold
                right_indices = ~left_indices

                if np.sum(left_indices) < self._min_samples_leaf or np.sum(right_indices) < self._min_samples_leaf:
                    continue

                left_model = LinearRegression().fit(sub_X[left_indices], sub_y[left_indices])
                right_model = LinearRegression().fit(sub_X[right_indices], sub_y[right_indices])

                left_error = mean_squared_error(sub_y[left_indices], left_model.predict(sub_X[left_indices]))
                right_error = mean_squared_error(sub_y[right_indices], right_model.predict(sub_X[right_indices]))

                total_error = (len(sub_y[left_indices]) / n_samples) * left_error + \
                              (len(sub_y[right_indices]) / n_samples) * right_error

                if total_error < best_error:
                    best_feature = feature
                    best_threshold = threshold
                    best_error = total_error
                    best_split = left_indices

        if best_feature is None:
            node["type"] = "terminal"
            model = LinearRegression()
            model.fit(sub_X, sub_y)
            node["model"] = model
            return

        node["type"] = "nonterminal"
        node["feature_split"] = best_feature
        node["threshold"] = best_threshold

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[best_split], sub_y[best_split], node["left_child"], cur_depth + 1)
        self._fit_node(sub_X[~best_split], sub_y[~best_split], node["right_child"], cur_depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["model"].predict(x.reshape(1, -1))[0]
        feature_to_split = node["feature_split"]
        threshold = node["threshold"]
        if x[feature_to_split] < threshold:
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 1)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])

    def get_params(self, deep=False):
        return {'feature_types': self._feature_types,
                'max_depth': self._max_depth,
                'min_samples_split': self._min_samples_split,
                'min_samples_leaf': self._min_samples_leaf,
                'n_quantiles': self._n_quantiles}

