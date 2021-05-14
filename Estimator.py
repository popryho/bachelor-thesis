import inspect
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm


# from sklearn.utils import check_X_y
# from sklearn.utils.multiclass import check_classification_targets


class Estimator(object):

    def __init__(self, eps_0=1, p_=0.5, N_=45):

        self.eps_0 = eps_0
        self.tol_ = 0
        self.n_, self.m_ = 0, 0

        self.eigen_values_ = np.empty(0)
        self.eigen_vectors_ = np.empty(0)

        self.train_samples_ = 0
        self.coeff_ = np.empty(0)

        self.p_ = p_
        self.N_ = N_

        self.X_, self.y_ = np.empty(0), np.empty(0)
        self.X_train_, self.y_train_ = np.empty(0), np.empty(0)

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
               f'Ɛ₀={self.eps_0}, ' \
               f'tol={self.tol_}, ' \
               f'p={self.p_}, ' \
               f'N={self.N_})'

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    # --------------------------------------------------------------

    def weight(self, xi, xj):
        norm = np.linalg.norm(xi - xj)
        return np.exp(-norm ** 2 / (4 * self.tol_))

    def graph_Laplacian(self):
        pairwise_dists = squareform(pdist(self.X_, 'sqeuclidean'))
        L = - np.exp(- pairwise_dists / (4 * self.tol_))
        np.fill_diagonal(L, - L.sum(axis=0) - 1)
        return L / self.n_

    def epsilon_finder(self, q=0.9, lambda_thr=1e-6):

        j = 1
        eig_values_vectors = []

        while True:

            self.tol_ = self.eps_0 * q ** j
            L = self.graph_Laplacian()
            self.eigen_values_, self.eigen_vectors_ = np.linalg.eig(L)

            eig_values_vectors = np.append(eig_values_vectors, self.eigen_values_)
            if self.eigen_values_[1] < lambda_thr:
                break
            j += 1

        eig_values_vectors = np.reshape(eig_values_vectors, newshape=(-1, self.n_))

        if eig_values_vectors.shape[0] <= 1:
            return q ** (j - 1)
        else:
            d = {}
            for i in range(eig_values_vectors.shape[0] - 1):
                d[i] = np.linalg.norm(eig_values_vectors[i + 1] - eig_values_vectors[i])
            return q ** (min(d, key=d.get) + 1)

    # --------------------------------------------------------------

    def eigen_func(self, x, k):
        weights = [self.weight(xi=x, xj=xj) for xj in self.X_]
        return np.dot(weights, self.eigen_vectors_[:, k]) / (np.sum(weights) - self.n_ * self.eigen_values_[k])

    def kernel_function(self, x, t):
        K = 0
        for k in range(self.n_):
            K += self.eigen_func(x, k) * self.eigen_func(t, k) / (self.n_ * self.eigen_values_[k])
        return K

    def gram_matrix(self):

        K = np.zeros((self.train_samples_, self.train_samples_))
        for i in range(self.train_samples_):
            for j in range(self.train_samples_):
                K[i][j] = self.kernel_function(self.X_train_[i, :],
                                               self.X_train_[j, :])
        return K

    # --------------------------------------------------------------

    def coeff_c_finder(self, K):

        alpha = [self.p_ ** k for k in range(self.N_)]

        Identity = np.identity(self.train_samples_)
        coeffs = np.zeros((self.N_, self.train_samples_))

        for k in range(self.N_):
            coeffs[k] = np.dot(np.linalg.inv(alpha[k] * Identity + K), self.y_train_)
        d = {}
        for k in range(self.N_):
            d[alpha[k]] = np.linalg.norm(np.dot(K, coeffs[k]) - self.y_train_)
        a = min(d, key=d.get)

        return np.dot(np.linalg.inv(a * Identity + K), self.y_train_)

    def decision_function(self, x):

        res = 0
        for i in range(self.train_samples_):
            res += self.coeff_[i] * self.kernel_function(x, self.X_train_[i, :])
        return res

    # --------------------------------------------------------------

    def fit(self, X, y):

        self.X_ = X
        self.y_ = y
        self.n_, self.m_ = self.X_.shape

        self.tol_ = self.epsilon_finder()

        L = self.graph_Laplacian()
        self.eigen_values_, self.eigen_vectors_ = np.linalg.eig(L)

        self.X_train_, self.y_train_ = X[y != -1], y[y != -1]
        self.train_samples_ = self.X_train_.shape[0]

        K = self.gram_matrix()

        self.coeff_ = self.coeff_c_finder(K)
        return self

    def predict(self, X):
        return np.where(self.predict_proba(X) > .5, 1, 0)

    def predict_proba(self, X):
        y_pred = [self.decision_function(X[i, :]) for i in tqdm(range(X.shape[0]))]
        return np.array(y_pred)
