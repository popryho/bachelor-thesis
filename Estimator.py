import numpy as np


# from sklearn.utils import check_X_y
# from sklearn.utils.multiclass import check_classification_targets


class Estimator(object):

    def __init__(self, tol=1, p=0.5, N=45):

        self.tol_ = tol
        self.n_, self.m_ = 0, 0

        self.eigen_values_ = np.empty(0)
        self.eigen_vectors_ = np.empty(0)

        self.train_samples_ = 0
        self.coeff_ = np.empty(0)

        self.p_ = p
        self.N_ = N

        self.tols_ = np.empty(0)
        self.X_, self.y_ = np.empty(0), np.empty(0)
        self.X_train_, self.y_train_ = np.empty(0), np.empty(0)

    # --------------------------------------------------------------

    def weight(self, xi, xj):
        norm = np.linalg.norm(xi - xj)
        return np.exp(-norm ** 2 / (4 * self.tol_))

    def graph_Laplacian(self):
        L = np.zeros((self.n_, self.n_))
        for i in range(self.n_):
            for j in range(self.n_):
                if i != j:
                    L[i][j] = - self.weight(self.X_[i, :], self.X_[j, :])
                elif i == j:
                    for k in range(self.n_):
                        if i != k:
                            L[i][j] += self.weight(self.X_[i, :], self.X_[k, :])
        return L / self.n_

    def epsilon_finder(self, q=0.9, lmbd_thr=1e-3):

        j = 1
        eig_vals_vectors = []

        while True:

            self.tol_ = q ** j
            L = self.graph_Laplacian()
            eigen_values, eigen_vectors = np.linalg.eig(L)

            eig_vals_vectors = np.append(eig_vals_vectors, eigen_values)

            if eigen_values[1] < lmbd_thr:
                break
            j += 1

        eig_vals_vectors = np.resize(eig_vals_vectors, new_shape=(-1, self.n_))
        if eig_vals_vectors.shape[0] <= 1:
            return q ** (j - 1)
        else:
            d = {}
            for i in range(eig_vals_vectors.shape[0] - 1):
                d[i] = np.linalg.norm(eig_vals_vectors[i + 1] - eig_vals_vectors[i])
            return q ** (min(d, key=d.get) + 1)

    def finder_epsilon(self, q=0.9, lmbd_thr=1e-5):

        j = 1
        eig_vals_vectors = []

        while True:

            self.tol_ = q ** j
            L = self.graph_Laplacian()
            eigen_values, eigen_vectors = np.linalg.eig(L)

            eig_vals_vectors = np.append(eig_vals_vectors, eigen_values)

            if eigen_values[1] < lmbd_thr:
                break
            j += 1

        eig_vals_vectors = np.resize(eig_vals_vectors, new_shape=(-1, self.n_))

        if eig_vals_vectors.shape[0] <= 1:
            return np.array([q ** (j - 1)] * self.n_)
        else:
            diffs = np.array(
                [eig_vals_vectors[i + 1] - eig_vals_vectors[i] for i in range(eig_vals_vectors.shape[0] - 1)])
            idx, order = np.asarray(np.where(abs(diffs) == abs(diffs).min(axis=0)))

            return np.array([q ** idx[i] for i in order])

    # --------------------------------------------------------------

    def eigen_func(self, x, k):
        numerator, denominator = 0, 0

        # self.tol_ = self.tols_[k]
        # L = self.graph_Laplacian()
        # self.eigen_values_, self.eigen_vectors_ = np.linalg.eig(L)

        for j in range(self.n_):
            numerator += self.weight(x, self.X_[j, :]) * self.eigen_vectors_[j, k]
        for j in range(self.n_):
            denominator += self.weight(x, self.X_[j, :]) - self.n_ * self.eigen_values_[k]
        return numerator / denominator

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

        # L = self.graph_Laplacian()
        # self.eigen_values_, self.eigen_vectors_ = np.linalg.eig(L)
        # print(self.eigen_values_)

        self.tol_ = self.epsilon_finder()

        # self.tols_ = self.finder_epsilon()

        """self.tols_ = np.array(
            [0.0108, 0.0097, 0.0097, 0.0097, 0.0097, 0.0097, 0.0097, 0.0097, 0.0097, 0.0097, 0.0097, 0.0097, 0.0097,
             0.0097, 0.0097, 0.0097, 0.0097, 0.0182, 0.0164, 0.0133, 0.0097, 0.0097, 0.0120, 0.0164, 0.0182, 0.0097, 
             0.0097, 0.0097, 0.0133, 0.0278, 0.0133, 0.0108, 0.0133, 0.0097, 0.0148, 0.0133, 0.0097, 0.0108, 0.0108, 
             0.0120, 0.0133, 0.0097, 0.0097, 0.0097, 0.0097, 0.0097, 0.0097, 0.0097, 0.0108, 0.0120, 0.0108, 0.0097, 
             0.0108, 0.0108, 0.0108, 0.0097, 0.0097, 0.0097, 0.0097, 0.0097, 0.0097, 0.0133, 0.0097, 0.0120, 0.0097, 
             0.0108, 0.0097, 0.0108, 0.0097, 0.0097, 0.0097, 0.0097, 0.0097, 0.0097, 0.0120, 0.0120, 0.0097, 0.0097, 
             0.0108, 0.0097, 0.0120, 0.0097, 0.0120, 0.0108, 0.0097, 0.0108, 0.0108, 0.0108, 0.0097, 0.0097, 0.0097, 
             0.0097, 0.0108, 0.0097, 0.0097, 0.0097, 0.0097, 0.0097, 0.0097, 0.0097, ])"""

        L = self.graph_Laplacian()
        self.eigen_values_, self.eigen_vectors_ = np.linalg.eig(L)

        self.X_train_, self.y_train_ = X[y != -1], y[y != -1]
        self.train_samples_ = self.X_train_.shape[0]

        K = self.gram_matrix()

        self.coeff_ = self.coeff_c_finder(K)
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = []
        for i in range(n_samples):
            temp = self.decision_function(X[i, :])
            pred = 0 if temp <= 0.5 else 1
            y_pred.append(pred)
        return np.array(y_pred)
# predict proba
# get params
