from scipy.stats import truncnorm
import numpy as np
import numpy.typing as npt
from taxonomy import Taxonomy, DomainClass, Relationship


class DeviationClass(frozenset[np.intp]):
    """A set of concept indices that form a synthetic class.

    In synthetic taxonomy generation, a DeviationClass represents a group of atomic
    concepts (identified by integers) that are bundled together to form a class.
    For instance, a class "dog" might contain concepts like "furry", "four legs", "tail", etc.

    Implemented as an immutable frozenset for hashability and use as dictionary keys.
    """


class Deviation(frozenset[DeviationClass]):
    """A collection of related synthetic classes forming a domain.

    A Deviation represents a complete taxonomy domain containing multiple DeviationClasses.
    This structure simulates a real-world dataset domain (like CIFAR-100 or Caltech-101)
    with its class structure for simulation purposes.

    Implemented as an immutable frozenset for hashability and use in graph structures.
    """

    def to_mapping(self) -> dict[int, int]:
        """Convert the DeviationClass to a mapping of concepts to their deviation class index.

        Returns
        -------
        dict[int, int]
            A dictionary where keys are concept indices
            and values are their deviation class index.
        """
        mapping = {}
        for class_index, deviation_class in enumerate(self):
            for concept_index in deviation_class:
                mapping[int(concept_index)] = class_index

        return mapping


class SimulatedPredictions(tuple[np.intp, np.intp, npt.NDArray[np.float64]]):
    """Container for cross-domain prediction probabilities in synthetic taxonomy experiments.

    Represented as a tuple (source_domain_id, target_domain_id, probability_matrix) where:
    - source_domain_id: The domain ID of the model making predictions
    - target_domain_id: The domain ID of the dataset being predicted
    - probability_matrix: 2D array where element [i,j] represents the probability of
      predicting class j in the source domain for class i in the target domain
    """


class SyntheticTaxonomy(Taxonomy):
    """A taxonomy generator for synthetic domain classes with controlled relationships.

    This class creates artificial taxonomies with configurable parameters to simulate
    real-world domain relationships. It's useful for testing taxonomy algorithms with
    known ground truth relationships between classes.

    The synthetic taxonomy is built by generating multiple domains (deviations), each with
    a set of classes that are composed of atomic concept elements. Cross-domain relationships
    are established based on conceptual overlaps between classes.
    """

    def __init__(
        self,
        num_atomic_concepts: int,
        num_domains: int,
        domain_class_count_mean: float,
        domain_class_count_variance: float,
        concept_cluster_size_mean: float,
        concept_cluster_size_variance: float,
        random_seed: int = 42,
    ):
        """Create a synthetic taxonomy with randomly generated domains and relationships.

        Parameters
        ----------
        num_atomic_concepts : int
            The total number of atomic concepts available in the universe
        num_domains : int
            The number of domains (deviations) to generate
        domain_class_count_mean : float
            Mean number of classes per domain
        domain_class_count_variance : float
            Variance in the number of classes per domain
        concept_cluster_size_mean : float
            Mean number of concepts per class
        concept_cluster_size_variance : float
            Variance in the number of concepts per class
        random_seed : int, optional
            Seed for random number generation, by default 42

        Notes
        -----
        The initialization process:
        1. Generates synthetic domains with classes made of atomic concepts
        2. Simulates cross-domain predictions based on conceptual overlap
        3. Creates a taxonomy graph with relationships between domains
        """
        # Store parameters
        self.num_atomic_concepts = num_atomic_concepts
        self.num_domains = num_domains
        self.domain_class_count_mean = domain_class_count_mean
        self.domain_class_count_variance = domain_class_count_variance
        self.concept_cluster_size_mean = concept_cluster_size_mean
        self.concept_cluster_size_variance = concept_cluster_size_variance
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        # Generate synthetic domains (previously called deviations)
        self.domains = [self.__create_domain() for _ in range(num_domains)]

        # Calculate simulated prediction probabilities between all domain pairs
        # (excluding self-predictions)
        self.cross_domain_predictions: list[SimulatedPredictions] = []
        for source_domain_id, source_domain in enumerate(self.domains):
            for target_domain_id, target_domain in enumerate(self.domains):
                if source_domain_id == target_domain_id:
                    continue
                prediction_matrix = self.__simulate_predictions(
                    source_domain, target_domain
                )
                self.cross_domain_predictions.append(
                    SimulatedPredictions(
                        (
                            np.intp(source_domain_id),
                            np.intp(target_domain_id),
                            prediction_matrix,
                        )
                    )
                )

        # Create human-readable domain labels for visualization
        domain_labels = {}
        for domain_id, domain in enumerate(self.domains):
            class_labels = []
            for class_concepts in domain:
                # Format each class as a set of its concept indices
                class_labels.append("{" + ", ".join(map(str, class_concepts)) + "}")
            domain_labels[domain_id] = class_labels

        # Initialize the taxonomy with empty predictions and targets
        # Actual relationships will be built from simulated cross-domain predictions
        super().__init__(
            cross_domain_predictions=[], domain_targets=[], domain_labels=domain_labels
        )

        # Add relationships to the taxonomy graph based on simulated predictions
        for (
            source_domain_id,
            target_domain_id,
            predictions,
        ) in self.cross_domain_predictions:
            for target_class_id, class_predictions in enumerate(predictions):
                # Get the most likely prediction for this class
                source_class_id = np.argmax(class_predictions)
                prediction_confidence = float(class_predictions[source_class_id])

                # Create domain classes for source and target
                source_class = DomainClass(
                    (np.intp(source_domain_id), np.intp(source_class_id))
                )
                target_class = DomainClass(
                    (np.intp(target_domain_id), np.intp(target_class_id))
                )

                # Add the relationship to the taxonomy graph
                relationship = Relationship(
                    (target_class, source_class, prediction_confidence)
                )
                self._add_relationship(relationship)

    @staticmethod
    def __simulate_predictions(
        source_domain: Deviation,
        target_domain: Deviation,
    ) -> npt.NDArray[np.float64]:
        """Simulate how a model trained on source_domain would classify samples from target_domain.

        This method calculates prediction probabilities based on concept overlap between
        classes in different domains. The core idea is that a class can be partially recognized
        by another domain's model if they share some common concepts.

        Parameters
        ----------
        source_domain : Deviation
            The domain the classifier was trained on
        target_domain : Deviation
            The domain containing samples to be classified

        Returns
        -------
        npt.NDArray[np.float64]
            2D array of prediction probabilities where:
              - Each row corresponds to a class in the target domain
              - Each column corresponds to a class in the source domain
              - Value [i,j] is the probability of predicting source class j for target class i

        Notes
        -----
        The prediction simulation works as follows:
        1. For each target class, find which source classes share concepts with it
        2. Assign probabilities based on percentage of shared concepts
        3. Distribute remaining probability evenly across all classes

        For example, if target class {A, B} is classified by a source domain with
        classes {A, C} and {B, D}, it would have 50% overlap with each source class.
        """
        prediction_probabilities = []

        # For each class in the target domain
        for target_class in target_domain:
            # Extract the atomic concepts that make up this class
            target_concept_set = set(target_class)

            # Create probability distribution for this target class (initially all zeros)
            class_probabilities = [0.0] * len(source_domain)

            # Find and calculate overlapping concepts with each source class
            concept_overlaps = []
            for source_idx, source_class in enumerate(source_domain):
                source_concept_set = set(source_class)
                shared_concepts = target_concept_set.intersection(source_concept_set)

                # If there's overlap, calculate the proportion of target concepts covered
                if shared_concepts:
                    overlap_ratio = len(shared_concepts) / len(target_concept_set)
                    concept_overlaps.append((source_idx, overlap_ratio))

            # Assign probabilities based on concept overlap ratios
            for source_idx, overlap_ratio in concept_overlaps:
                class_probabilities[source_idx] = overlap_ratio

            # Calculate remaining probability to distribute
            remaining_probability = 1.0 - sum(class_probabilities)

            # Distribute remaining probability evenly across all classes
            if remaining_probability > 0 and len(source_domain) > 0:
                even_distribution = remaining_probability / len(source_domain)
                class_probabilities = [
                    p + even_distribution for p in class_probabilities
                ]

            prediction_probabilities.append(class_probabilities)

        return np.array(prediction_probabilities, dtype=np.float64)

    def __create_domain(self) -> Deviation:
        """Generate a synthetic domain with classes composed of atomic concepts.

        A domain (previously called deviation) consists of multiple classes, each
        containing a cluster of related atomic concepts.

        Returns
        -------
        Deviation
            A set of DeviationClasses, where each DeviationClass represents
            a class in this domain

        Notes
        -----
        The domain generation process:
        1. Determine how many classes to include in this domain
        2. Select unique atomic concepts to be used in this domain
        3. Group concepts into clusters (classes) of varying sizes
        """
        # Determine number of classes for this domain (bounded by total concept count)
        class_count = np.round(
            self.__sample_truncated_normal(
                mean=self.domain_class_count_mean,
                variance=self.domain_class_count_variance,
                upper_bound=self.num_atomic_concepts,
                lower_bound=1,
            )
        ).astype(np.intp)

        # Select which atomic concepts will be part of this domain
        available_concepts = range(self.num_atomic_concepts)
        selected_concepts = set(
            self.rng.choice(available_concepts, size=class_count, replace=False)
        )

        # Create the domain as a set of classes
        domain = set()

        # Group concepts into classes until all selected concepts are assigned
        while selected_concepts:
            # Handle last concept separately to avoid empty clusters
            if len(selected_concepts) == 1:
                cluster_size = 1
            else:
                # Determine class size based on specified distribution
                cluster_size = np.round(
                    self.__sample_truncated_normal(
                        mean=self.concept_cluster_size_mean,
                        variance=self.concept_cluster_size_variance,
                        lower_bound=1,
                        upper_bound=len(selected_concepts),
                    )
                ).astype(np.intp)

            # Randomly select concepts for this class
            class_concepts = frozenset(
                self.rng.choice(
                    list(selected_concepts),
                    size=cluster_size,
                    replace=False,
                ).astype(np.intp)
            )

            # Create a class from these concepts and add it to the domain
            domain_class = DeviationClass(class_concepts)
            domain.add(domain_class)

            # Remove assigned concepts
            selected_concepts -= class_concepts

        return Deviation(frozenset(domain))

    def __sample_truncated_normal(
        self,
        mean: float,
        variance: float,
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
                a=a, b=b, loc=mean, scale=np.sqrt(variance), random_state=self.rng
            )
        except ValueError:
            # Return 0 if parameters lead to invalid distribution
            return 0
