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


def hypothesize_relationships(
    probabilities: npt.NDArray[np.float64], upper_bound: int = 5
) -> list[int]:
    """Hypothesize the number of relationships based on a probability array.

    The function hypothesizes the number of relationships by assuming
    that every relationship should have an equal share of the total probability.
    It calculates how much each hypothesis deviates from the given probabilities
    and returns the most likely hypothesis that minimizes this deviation.

    Parameters
    ----------
    probabilities : npt.NDArray[np.float64]
        An array of probabilities for each relationship.
    upper_bound : int, optional
        The maximum number of relationships to consider, by default 5

    Returns
    -------
    int
        The indices of the source classes that form the best relationships.
    """

    # Sort probabilities in descending order
    sorted_probabilities = np.sort(probabilities)[::-1]

    best_number_of_relationships = 1
    best_deviation = float("inf")
    for hypothesis in range(0, min(upper_bound, len(probabilities) + 1)):
        # The expected probabilities
        if hypothesis == 0:
            expected_probabilities = np.full(len(probabilities), 1 / len(probabilities))
        else:
            expected_probabilities = np.full(hypothesis, 1 / hypothesis)
            expected_probabilities = np.pad(
                expected_probabilities,
                (0, len(probabilities) - hypothesis),
                mode="constant",
                constant_values=0,
            )

        # Calculate the deviation from the expected probabilities
        deviation = np.sum(np.abs(sorted_probabilities - expected_probabilities))
        if deviation < best_deviation:
            best_deviation = deviation
            best_number_of_relationships = hypothesis

    # Get the indices of the best relationships
    best_relationships = np.argsort(probabilities)[::-1][:best_number_of_relationships]
    return best_relationships.tolist()
