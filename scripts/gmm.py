import numpy as np
import scipy.special as sp_special
from sklearn.mixture import GaussianMixture
import numpy.typing as npt
import scripts.tools as tools


class GMM:
    def __init__(self, means: list|np.ndarray, cov: list|np.ndarray|None=None, weights=None):
        """
        Args
        ----------
        means : array-like, shape (K, D)
            Initial means for each component.
        cov : array-like, shape (D, D)
            Known shared covariance matrix (must be positive definite). Defaults to identity.
        weights : array-like, shape (K,), optional
            Initial mixture weights. Defaults to uniform.
        """
        # TODO: initialize means with k-means++. need to add the n_components argument.
        self.means = np.array(means)               # shape (K, D)
        self.K, self.D = self.means.shape
        if cov is None:
            cov = np.eye(self.D)
        self.cov = np.array(cov)
        self.inv_cov = np.linalg.inv(self.cov)
        self.det_cov = np.linalg.det(self.cov)
        if weights is None:
            weights = np.ones(self.K) / self.K
        self.weights = np.array(weights)
        self.N = None
        self.responsibilities = None

    def _exponential_term(self, X: npt.NDArray, mean: npt.NDArray) -> npt.NDArray:
        """Compute the exponent term of the multivariate Gaussian PDF.
        Args:
            X : array-like, shape (N, D)
                Snippets.
            mean : array-like, shape (D,)
                Mean of the Gaussian.
        Returns:
            array-like, shape (N,)
                Exponent values for each snippet."""
        diff = X - mean
        exponent = -0.5 * np.sum(diff @ self.inv_cov * diff, axis=1)
        return exponent

    def _log_gaussian_pdf(self, X: npt.NDArray, mean: npt.NDArray) -> npt.NDArray:
        """Log of the multivariate Gaussian PDF (up to normalization constant).
        Args:
            X : array-like, shape (N, D)
                Snippets.
            mean : array-like, shape (D,)
                Mean of the Gaussian.
        Returns:
            array-like, shape (N,)
                Probability density values for each data point."""
        exponent = self._exponential_term(X, mean)
        log_norm_const = 0.5 * (self.D * np.log(2 * np.pi) + np.log(self.det_cov))
        return exponent - log_norm_const

    def _e_step(self, X: npt.NDArray) -> npt.NDArray:
        """Compute responsibilities (E-step)."""
        self.N = X.shape[0]
        log_prob = np.zeros((self.N, self.K))

        for k in range(self.K):
            log_prob[:, k] = np.log(self.weights[k]) + self._log_gaussian_pdf(X, self.means[k])

        # log-sum-exp normalization
        log_sum = sp_special.logsumexp(log_prob, axis=1, keepdims=True)
        log_resp = log_prob - log_sum

        self.responsibilities = np.exp(log_resp)
        return self.responsibilities

    def _m_step(self, X: npt.NDArray) -> None:
        """Update mixture weights only (M-step)."""
        Nk = self.responsibilities.sum(axis=0)
        self.weights = Nk / self.N
        self.means = (self.responsibilities.T @ X) / Nk[:, None]

    # TODO: add a method to compute the responsibilities given means and weights

    def _log_likelihood(self, X: npt.NDArray, means: npt.NDArray, weights: npt.NDArray) -> float:
        """Compute log-likelihood of data under the given parameters.
        Args:
            X : array-like, shape (N, D)
                Snippets.
            means : array-like, shape (K, D)
                Means for each component.
            weights : array-like, shape (K,)
                Mixture weights.
        Returns:
            float
                Log-likelihood value."""
        means = np.array(means)
        weights = np.array(weights)
        total = 0
        for x in X:
            log_pxk = []
            for k in range(self.K):
                log_px = self._log_gaussian_pdf(x[None, :], means[k])[0]
                log_pxk.append(np.log(weights[k]) + log_px)

            max_log = np.max(log_pxk)
            total += max_log + np.log(np.sum(np.exp(np.array(log_pxk) - max_log)))
        return total
    
    def log_likelihood(self, X: npt.NDArray, means: list|np.ndarray=None, weights: list|np.ndarray=None)->float:
        """Compute log-likelihood of data under the given parameters. If means or weights are None, use current parameters.

        Args:
            X : array-like, shape (N, D)
                snippets.
            means : array-like, shape (K, D), optional
                Means for each component. Defaults to current means.
            weights : array-like, shape (K,), optional
                Mixture weights. Defaults to current weights.
        Returns:
            float
                Log-likelihood value."""
        if means is None:
            means = self.means
        if weights is None:
            # TODO: use Hungarian algorithm to match components
            weights = self.weights
        return self._log_likelihood(X, means, weights)

    def _current_log_likelihood(self, X: npt.NDArray)->float:
        """Compute log-likelihood of data under current parameters.
        Args:
            X : array-like, shape (N, D)
                snippets.
        Returns:
            float
                Log-likelihood value."""

        return self.log_likelihood(X)

    def fit(self, X: npt.NDArray, n_iter: int=100) -> 'GMM':
        """Run EM with fixed means and shared covariance.
        Args:
            X : array-like, shape (N, D)
                snippets. Number of snippets = N, snippet dimension (snippet_length) = D.
            n_iter : int, optional
                Number of EM iterations. Defaults to 100.
        Returns:
            GMM
                Fitted GMM instance."""
        print("--- EM Optimization ---")
        for i in range(n_iter):
            self._e_step(X)
            self._m_step(X)
            ll = self._current_log_likelihood(X)
            if i in set([0, n_iter - 1]) or (i + 1) % 10 == 0:
                print(f"Iter {i+1}: log-likelihood = {ll:.3f}")
            if i > 0 and abs(ll - prev_ll) < 1e-6:
                print(f"Converged at iteration {i+1}.")
                break
            prev_ll = ll
        return self
    
    def fit_reorder_labels(self, X: npt.NDArray, label_true: npt.NDArray, n_iter: int=100) -> 'GMM':
        """Run EM with fixed means and shared covariance using HuangarianMatcher to align components such that the labels are consistent with the initial labels.

        This will change the order of the GMM components internally.

        Args:
            X : array-like, shape (N, D)
                snippets. Number of snippets = N, snippet dimension (snippet_length) = D.
            label_true : array-like, shape (N,)
                Ground truth labels for each snippet.
            n_iter : int, optional
                Number of EM iterations. Defaults to 100.
        Returns:
            GMM
                Fitted GMM instance."""
        self.fit(X, n_iter=n_iter)
        label_pred = self.predict(X)
        hungarian_matcher = tools.HungarianMatcher(label_true, label_pred)
        self.means, self.weights, self.responsibilities = hungarian_matcher.reorder_gmm_parameters(self.means, self.weights, self.responsibilities)
        return self

    def predict_proba(self, X: npt.NDArray) -> npt.NDArray:
        """Compute responsibilities (posterior probabilities) for each snippets.
        Args:
            X : array-like, shape (N, D)
                Snippets. Number of snippets = N, snippet dimension (snippet_length) = D.
        Returns:
            array-like, shape (N, K)
                Responsibilities for each snippet. Number of snippets = N, number of components = K.
        """
        if self.responsibilities is None:
            self._e_step(X)
        return self.responsibilities

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """Assign each snippet to the component with highest responsibility.
        Args:
            X : array-like, shape (N, D)
                Snippets. Number of snippets = N, snippet dimension (snippet_length) = D.
        Returns:
            array-like, shape (N,)
                Predicted component labels for each snippet.
        """
        return np.argmax(self.predict_proba(X), axis=1)

def test_gmm(true_means:npt.NDArray, 
             true_cov:npt.NDArray, 
             means_init:npt.NDArray, 
             true_weights:npt.NDArray,
             X: npt.NDArray,
             labels_true: npt.NDArray=None)->tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Test GMM implementation against sklearn's GaussianMixture. Prints log-likelihoods and weights.
    Args:
        true_means : array-like, shape (K, D)
            Ground truth means for each component.
        true_cov : array-like, shape (D, D)
            Ground truth shared covariance matrix.
        means_init : array-like, shape (K, D)
            Initial means for my GMM.
        true_weights : array-like, shape (K,)
            Ground truth mixture weights.
        X : array-like, shape (N, D)
            Snippets.
        labels_true : array-like, shape (N,), optional
            Ground truth labels for each snippet. If provided, used for component alignment.
    Returns:
        tuple of array-like
            Fitted means, weights, and responsibilities from my GMM.
    """
    print(f"Generated data: n_component = {true_means.shape[0]}")
    print("\n")
    # print(f"true mean = {true_means}, cov = {true_cov}, weights = {true_weights}")

    # Fit my GMM
    gmm = GMM(means=means_init, cov=true_cov)
    gmm.fit_reorder_labels(X, labels_true, n_iter=300)
    # Fit sklearn's GMM
    n_component = true_means.shape[0]
    gmm_sklearn = GaussianMixture(n_components=n_component, covariance_type='spherical', max_iter=1000, init_params='k-means++', random_state=1)
    gmm_sklearn.fit(X)
    
    ll = gmm.log_likelihood(X, means=true_means, weights=true_weights) # Compute true log-likelihood    
    ll_estimated = gmm.log_likelihood(X) # Estimated results using my GMM
    ll_sklearn = gmm_sklearn.score(X) * X.shape[0] # Sklearn log-likelihood
    
    print("\n")
    print("--- Mixture Weights ---")
    print("True:", true_weights)
    print("MyGMM:", gmm.weights)
    print("Sklearn:", gmm_sklearn.weights_)
    print("\n")
    # print("--- Means Comparison ---")
    # print("True means:\n", true_means)
    # print("MyGMM Final means:\n", gmm.means)
    # print("Sklearn Final means:\n", gmm_sklearn.means_)
    # print("\n")
    # print("--- Covariances Comparison ---")
    # print("True covariance:\n", true_cov)
    # print("MyGMM Final covariances:\n", gmm.cov)
    # print("Sklearn Final covariances:\n", gmm_sklearn.covariances_)
    # print("\n")
    print("--- Log-Likelihood Comparison ---")
    print("True log-likelihood of data:", ll)
    print("MyGMM estimated log-likelihood of data:", ll_estimated)
    print("Sklearn estimated log-likelihood of data:", ll_sklearn)
    print("\n")
    tools.Plotter.plot_template(gmm.means, title="Fitted GMM Means after EM")

    return gmm.means, gmm.weights, gmm.responsibilities

if __name__ == "__main__":
    # Simple test case
    np.random.seed(0)
    true_means = np.array([[0, 0], [5, 5], [10, 10]])
    true_cov = np.array([[1, 0], [0, 1]])
    true_weights = np.array([0.3, 0.4, 0.3])

    n_samples = 1000
    X = np.vstack([
        np.random.multivariate_normal(true_means[0], true_cov, size=int(n_samples * true_weights[0])),
        np.random.multivariate_normal(true_means[1], true_cov, size=int(n_samples * true_weights[1])),
        np.random.multivariate_normal(true_means[2], true_cov, size=int(n_samples * true_weights[2])),
    ])

    labels_true = np.hstack([
        np.full(int(n_samples * true_weights[0]), 0, dtype=int),
        np.full(int(n_samples * true_weights[1]), 1, dtype=int),
        np.full(int(n_samples * true_weights[2]), 2, dtype=int),
    ])
    
    means_init = np.array([[1, 1], [6, 6], [9, 9]])

    test_gmm(true_means, true_cov, means_init, true_weights, X, labels_true)