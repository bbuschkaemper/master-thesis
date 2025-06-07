from typing import List
import numpy as np
import numpy.typing as npt
from scipy.stats import truncnorm


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


def hypothesis_relationships(
    probabilities: npt.NDArray[np.float64], upper_bound: int = 5
) -> List[int]:
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
    return best_relationships.tolist()  # type: ignore


def density_threshold_relationships(
    probabilities: npt.NDArray[np.float64],
    threshold: float = 0.5,
) -> List[int]:
    """Filter relationships based on a density threshold.

    Find the least number of relationships whose probabilities
    sum up to at least the given threshold.

    Parameters
    ----------
    probabilities : npt.NDArray[np.float64]
        An array of probabilities for each relationship.
    threshold : float, optional
        The minimum cumulative probability threshold to meet, by default 0.5

    Returns
    -------
    List[int]
        The indices of the source classes that form the best relationships.
    """

    # Sort probabilities in descending order
    sorted_probabilities = np.sort(probabilities)[::-1]

    cumulative_sum = 0.0
    selected_indices = []
    for index, prob in enumerate(sorted_probabilities):
        cumulative_sum += prob
        selected_indices.append(index)
        if cumulative_sum >= threshold:
            break

    # Get the original indices of the selected relationships
    original_indices = np.argsort(probabilities)[::-1][selected_indices]
    return original_indices.tolist()  # type: ignore
