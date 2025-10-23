from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt

class HungarianMatcher:
    """Class to perform Hungarian matching between true and predicted labels."""
    def __init__(self, label_true: npt.NDArray, label_pred: npt.NDArray):
        self.label_true = label_true
        self.label_pred = label_pred
        self.mapping = self._compute_mapping()
    
    def _compute_mapping(self)->dict:
        """Compute the optimal mapping using the Hungarian algorithm."""
        cm = confusion_matrix(self.label_true, self.label_pred)
        row_ind, col_ind = linear_sum_assignment(-cm)
        mapping = {true_label: pred_label for true_label, pred_label in zip(row_ind, col_ind)}
        return mapping
    
    def reorder_labels(self)->npt.NDArray:
        """Reorder predicted labels according to the computed mapping.
        
        Returns:
            array-like
                Reordered predicted labels. 
        Example:
            y_true = np.array([0, 0, 1, 1, 2, 2])

            y_pred = np.array([2, 2, 0, 0, 1, 1])

            Then, the mapping is {0:2, 1:0, 2:1} and label_pred is [2,2,0,0,1,1],
            the output will be [0,0,1,1,2,2].
"""
        label_pred_aligned = np.vectorize(self.mapping.get)(self.label_pred)
        return label_pred_aligned

    def reorder_gmm_parameters(self, means: npt.NDArray, weights: npt.NDArray, responsibilities: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Reorder GMM parameters (means, weights, and responsibilities) according to the computed mapping.
        
        Args:
            means : array-like, shape (K, D)
                GMM component means. The number of components is K, and the snippet dimension is D.
            weights : array-like, shape (K,)
                GMM component weights.
            responsibilities : array-like, shape (N, K)
                GMM responsibilities (posterior probabilities).

        Returns:
            tuple of array-like
                Reordered means, weights, and responsibilities.
        
        Example:
            true_mean = np.array([[0, 0], [5, 5], [10, 10]])

            pred_mean = np.array([[5, 5], [10, 10], [0, 0]])

            true_weights = np.array([0.3, 0.4, 0.3])

            pred_weights = np.array([0.4, 0.3, 0.3])

            Then, the mapping is {0:2, 1:0, 2:1}, the output will be:

            means_aligned = np.array([[0, 0], [5, 5], [10, 10]])

            weights_aligned = np.array([0.3, 0.4, 0.3])
        """
        means_aligned = np.zeros_like(means)
        weights_aligned = np.zeros_like(weights)
        responsibilities_aligned = np.zeros_like(responsibilities)

        for true_idx, pred_idx in self.mapping.items():
            means_aligned[true_idx] = means[pred_idx]
            weights_aligned[true_idx] = weights[pred_idx]
            responsibilities_aligned[:, true_idx] = responsibilities[:, pred_idx]

        return means_aligned, weights_aligned, responsibilities_aligned

class Plotter:
    """Class for plotting synthetic intracellular potential data."""
    @staticmethod
    def plot_template(template: npt.NDArray, title = "Synthetic Intracellular Potential Templates (Sampled at 30 kHz)")->None:
        """Plot synthetic intracellular potential templates.
        Args:
            template : array-like, shape (num_templates, snippet_length)
                Synthetic intracellular potential templates.
            title : str, optional
                Title of the plot.
        Returns:
            None
        """
        plt.figure(figsize=(7, 4))
        time = np.arange(template.shape[1])
        for i in range(template.shape[0]):
            plt.plot(time, template[i], lw=2, label=f'Template {i+1}')

        plt.title(title)
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Potential (mV)")
        plt.axhline(0, color='gray', ls='--', lw=0.8, label='Resting potential')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    pass
    