from functools import partial
import numpy as np


class GMM:
    def __init__(self, init_centers):
        """Gaussian Mixture Model.
        Args:
            init_centers :K, D
             K initial centers.
        """
        self.K, self.D = init_centers.shape
        assert self.K > 1, f"There must be at least 2 clusters. Got: {self.K}"

        # K x D
        self.centers = init_centers.copy()

        # K x D x D
        self.covariances = np.repeat(np.identity(self.D)[np.newaxis, ...],
                                     self.K, axis=0)

        # K x 1
        self.alpha = np.ones(shape=(self.K, 1)) / self.K

    def e_step(self, X):
        ''' E step.

        Args:
            X:

        Returns:
            r_matrix: N x K
                represents responsibilities of p(z_k|x_i).

        '''
        precisions = np.linalg.inv(self.covariances)  # K x D x D
        dets = np.linalg.det(self.covariances)  # K
        coeff = 1.0 / (np.power(2 * np.pi, self.D / 2) * np.sqrt(dets))
        delta = (X[:, None, :] - self.centers[None, ...])  # N x K x D
        dist = delta[:, :, None, :] @ precisions[None, ...] @ delta[..., None]
        dist = np.squeeze(dist)
        gamma = coeff[None, :] * np.exp(
            -1.0 / 2.0 * dist) * self.alpha.T
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        return gamma

    def m_step(self, X, probability_matrix):
        ''' M step.

        Args:
            X: N x D,
            probability_matrix:

        Returns:
            centers : (K, D)
                update Gaussian means.
            covariances :(K, D, D)
                update Gaussian covariance matrix.
            r_matrix :(K, 1)
                mixture proportion for each Gaussian component.

        '''

        (N, D) = X.shape

        Nk = probability_matrix.sum(0)  # K
        centers = np.sum(probability_matrix[..., None] * X[:, None, :],
                         axis=0) / Nk[:, None]  # K x D
        delta = X[:, None, :] - centers[None, ...]
        covariances = np.sum(probability_matrix[..., None, None] * (
                delta[..., None] @ delta[:, :, None, :]), axis=0) / Nk[:, None,
                                                                    None]
        alpha = Nk[:, None] / N

        return centers, covariances, alpha

    def fit(self, X, max_iterations=1000):
        '''

        Args:
            X: N x D
            max_iterations: maximum iterations

        Returns:

        '''

        labels = np.empty(X.shape[0])
        for _ in range(max_iterations):
            old_labels = labels
            # E-Step
            gamma = self.e_step(X)

            labels = np.argmax(gamma, axis=1)

            if np.allclose(old_labels, labels):
                break

            # M-Step
            self.centers, self.covariances, self.alpha = \
                self.m_step(X, probability_matrix=gamma)

    def predict_prob(self,X):
        '''

        Args:
            X: N, D

        Returns:
        '''
        return self.e_step(X)

    def predict(self, X):
        '''

        Args:
            X: N x D

        Returns:
            pred labels
        '''
        prob = self.e_step(X)
        return np.argmax(prob, axis=1)





if __name__ == '__main__':
    import pickle
    with open('../data/clustering/test_1.pkl', "rb") as f:
        test_data = pickle.load(f)

    model = GMM(test_data["init_centers"])
    model.fit(test_data["data"])
    labels = model.predict(test_data['data'])

    assert np.allclose(model.centers, test_data["gmm_centers"])
    assert np.allclose(model.covariances, test_data["gmm_covariances"])
    assert np.allclose(model.alpha, test_data["gmm_mixture_proportions"])
    assert np.allclose(labels, test_data["gmm_labels"].squeeze())



