import numpy as np
import numpy.typing as npt
from scipy.stats import truncnorm


def form_correlation_matrix(
    predictions: npt.NDArray[np.intp],
    targets: npt.NDArray[np.intp],
) -> npt.NDArray[np.intp]:
    """Forms a correlation matrix between true target classes and predicted source classes.

    This method creates a matrix where:
    - Each row corresponds to a true class in the target domain
    - Each column corresponds to a predicted class from the source domain
    - The value at position (i,j) counts how many times target domain class i
      was predicted as source domain class j

    Parameters
    ----------
    predictions : npt.NDArray[np.intp]
        Array of class predictions made by a source domain model
        on target domain data
    targets : npt.NDArray[np.intp]
        Array of ground truth class labels for the target domain data

    Returns
    -------
    npt.NDArray[np.intp]
        The correlation matrix of shape (n_classes_target, n_classes_source)
    """

    # Find the number of unique classes in both predictions and targets
    n_target_domain_classes = np.max(targets) + 1
    n_source_domain_classes = np.max(predictions) + 1

    # Initialize correlation matrix with zeros
    correlations = np.zeros(
        (n_target_domain_classes, n_source_domain_classes),
        dtype=np.intp,
    )

    # Count occurrences of each (target, prediction) pair
    for i, predicted_class in enumerate(predictions):
        true_class = targets[i]
        correlations[true_class, predicted_class] += 1

    return correlations


def foreign_prediction_distributions(
    correlations: npt.NDArray[np.intp],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]:
    """Calculates the most common prediction for each class and its confidence.

    This method analyzes a correlation matrix to determine:
    1. For each class in the target domain, which class from the source domain
       is most frequently predicted
    2. The confidence (probability) of each relationship based on prediction frequencies

    Parameters
    ----------
    correlations : npt.NDArray[np.intp]
        A correlation matrix where each element (i,j) represents how many times
        target domain class i was predicted as source domain class j

    Returns
    -------
    tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]
        A tuple containing:
        - Array of most commonly predicted source classes for each target class
        - Array of confidence values (probabilities) for those predictions
    """
    n_target_domain_classes = correlations.shape[0]
    most_common_classes = np.zeros(n_target_domain_classes, dtype=np.intp)
    confidence_values = np.zeros(n_target_domain_classes, dtype=np.float32)

    # For each class in the target domain
    for target_class_idx in range(n_target_domain_classes):
        # Find the most commonly predicted class from the source domain
        prediction_counts = correlations[target_class_idx, :]
        most_common_classes[target_class_idx] = np.argmax(prediction_counts)

        # Calculate confidence as the proportion of predictions for this class
        total_predictions = np.sum(prediction_counts)
        if total_predictions > 0:
            max_count = prediction_counts[most_common_classes[target_class_idx]]
            confidence_values[target_class_idx] = max_count / total_predictions
        else:
            confidence_values[target_class_idx] = 0.0

    return most_common_classes, confidence_values


def sample_truncated_normal(
    mean: float,
    variance: float,
    rng: np.random.Generator,
    lower_bound: float = 0,
    upper_bound: float = float("inf"),
) -> float:
    """Sample from a truncated normal distribution with specified parameters.

    Parameters
    ----------
    mean : float
        Mean of the normal distribution
    variance : float
        Variance of the normal distribution
    lower_bound : float, optional
        Minimum value to return, by default 0
    upper_bound : float, optional
        Maximum value to return, by default infinity

    Returns
    -------
    float
        A sample from the truncated normal distribution

    Notes
    -----
    Returns 0 on error (when parameters lead to invalid distribution)
    """
    # Calculate standardized bounds for truncated normal distribution
    a = (lower_bound - mean) / np.sqrt(variance)
    b = (upper_bound - mean) / np.sqrt(variance)

    try:
        # Sample from the truncated normal distribution
        return truncnorm.rvs(
            a=a, b=b, loc=mean, scale=np.sqrt(variance), random_state=rng
        )
    except ValueError:
        # Return 0 if parameters lead to invalid distribution
        return 0
