import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm


# TODO:
#  add predict probabilities function
#  add get params function

# from sklearn.utils import check_X_y
# from sklearn.utils.multiclass import check_classification_targets


class Estimator(object):

    def __init__(self, eps_0=1, p=0.5, N=45):

        self.eps_0 = eps_0
        self.tol_ = 0
        self.n_, self.m_ = 0, 0

        self.eigen_values_ = np.empty(0)
        self.eigen_vectors_ = np.empty(0)

        self.train_samples_ = 0
        self.coeff_ = np.empty(0)

        self.p_ = p
        self.N_ = N

        self.X_, self.y_ = np.empty(0), np.empty(0)
        self.X_train_, self.y_train_ = np.empty(0), np.empty(0)

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
               f'Ɛ₀={self.eps_0}, ' \
               f'tol={self.tol_}, ' \
               f'p={self.p_}, ' \
               f'N={self.N_})'

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
