from typing import List, Tuple, Dict, TypeAlias, Literal
import numpy as np
import numpy.typing as npt
from .taxonomy import Taxonomy, DomainClass, Relationship
from .utils import (
    hypothesis_relationships,
    density_threshold_relationships,
    sample_truncated_normal,
)
from .types import Deviation, DeviationClass, SimulatedPredictions


_RelationshipFilteringMethod: TypeAlias = Literal[
    "mcfp",
    "hypothesis",
    "density_threshold",
    "threshold",
]
_SyntheticRelationshipFilteringMethod: TypeAlias = Literal[
    "mcfp",
    "true",
]


class CrossPredictionsTaxonomy(Taxonomy):
    """Class for building a taxonomy from cross-domain predictions.
    This taxonomy is constructed by analyzing how models trained on one domain
    classify samples from another domain, revealing conceptual relationships
    between classes across domains.
    """

    @classmethod
    def from_cross_domain_predictions(
        cls,
        cross_domain_predictions: List[Tuple[int, int, npt.NDArray[np.intp]]],
        domain_targets: List[Tuple[int, npt.NDArray[np.intp]]],
        domain_labels: Dict[int, List[str]] | None = None,
        relationship_type: _RelationshipFilteringMethod = "mcfp",
        threshold: float = 0.5,
        upper_bound: int = 5,
    ):
        """Creates a taxonomy object with an integrated graph structure.

        This taxonomy is built from cross-domain predictions between multiple domains.
        It analyzes how models trained on one domain classify samples from another domain,
        revealing conceptual relationships between classes across domains.

        Parameters
        ----------
        cross_domain_predictions : List[Tuple[int, int, npt.NDArray[np.intp]]]
            List of tuples where each tuple contains:
            - model_domain_id: The domain ID of the model used for predictions
            - dataset_domain_id: The domain ID of the dataset being predicted
            - predictions: Array of class predictions made by the model
        domain_targets : List[Tuple[int, npt.NDArray[np.intp]]]
            List of tuples where each tuple contains:
            - domain_id: The domain ID
            - targets: Array of ground truth class labels for that domain
        domain_labels : Dict[int, List[str]], optional
            Optional dictionary mapping domain IDs to lists of human-readable class labels.
            If provided, these labels will be used for domain classes instead of class IDs.
            If not provided, class IDs will be used as labels.
        relationship_type : _RELATIONSHIP_TYPE, optional
            The type of relationship to establish between domains. Defaults to "mcfp".
        threshold : float, optional
            The threshold for establishing relationships.
            This applies only to "threshold" and "density_threshold" relationship types.
            Defaults to 0.5.
        upper_bound : int, optional
            The upper bound for the number of relationships to keep.
            This applies only to "hypothesis" relationship type.
            Defaults to 5.
        """

        obj = cls(domain_labels=domain_labels)

        # Store targets for each domain in a dictionary
        targets = {}
        for domain_id, target_array in domain_targets:
            if domain_id in targets:
                raise ValueError(f"Duplicate domain ID {domain_id} found in targets.")
            targets[domain_id] = target_array

        # Process each cross-domain prediction
        for model_domain_id, dataset_domain_id, predictions in cross_domain_predictions:
            if dataset_domain_id not in targets:
                raise ValueError(
                    f"Dataset domain ID {dataset_domain_id} not found in targets."
                )

            # Validate that prediction arrays match their respective target arrays
            dataset_targets = targets[dataset_domain_id]
            if predictions.shape != dataset_targets.shape:
                raise ValueError(
                    f"Predictions of domain {model_domain_id} to domain "
                    f"{dataset_domain_id} must match targets in shape. "
                    f"Got {predictions.shape} vs {dataset_targets.shape}"
                )

            # Build correlation matrix between these domains
            correlations = obj._form_correlation_matrix(predictions, dataset_targets)

            # Normalize the correlation matrix to ensure it sums to 1 for each target class
            correlations = correlations.astype(np.float64)
            row_sums = correlations.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0  # Avoid division by zero
            correlations /= row_sums

            for target_class_idx in range(correlations.shape[0]):
                # Create a target domain class for this target class index
                target_class = DomainClass(
                    (np.intp(dataset_domain_id), np.intp(target_class_idx))
                )
                class_probabilities = correlations[target_class_idx]

                # Add the target class node to the taxonomy graph
                obj._add_node(target_class)

                if relationship_type == "mcfp":
                    source_class_idx = np.argmax(class_probabilities)
                    probability = class_probabilities[source_class_idx]

                    # Create source domain class (from the model's domain)
                    source_class = DomainClass(
                        (np.intp(model_domain_id), np.intp(source_class_idx))
                    )

                    # Skip if the probability is zero.
                    if probability == 0.0:
                        continue

                    # Add the relationship to the taxonomy graph with its confidence value
                    obj._add_relationship(
                        Relationship((target_class, source_class, probability))
                    )
                elif relationship_type == "hypothesis":
                    relationships = hypothesis_relationships(
                        class_probabilities, upper_bound
                    )

                    # Iterate through all source classes
                    for source_class_idx in relationships:
                        # Create a source domain class for this source class index
                        source_class = DomainClass(
                            (np.intp(model_domain_id), np.intp(source_class_idx))
                        )

                        confidence = float(class_probabilities[source_class_idx])

                        # Add the relationship to the taxonomy graph with its confidence value
                        obj._add_relationship(
                            Relationship((target_class, source_class, confidence))
                        )
                elif relationship_type == "density_threshold":
                    relationships = density_threshold_relationships(
                        class_probabilities, threshold=threshold
                    )

                    # Iterate through all source classes
                    for source_class_idx in relationships:
                        # Create a source domain class for this source class index
                        source_class = DomainClass(
                            (np.intp(model_domain_id), np.intp(source_class_idx))
                        )

                        confidence = float(class_probabilities[source_class_idx])

                        # Add the relationship to the taxonomy graph with its confidence value
                        obj._add_relationship(
                            Relationship((target_class, source_class, confidence))
                        )
                elif relationship_type == "threshold":
                    # Iterate through all source classes and their probabilities
                    for source_class_idx, confidence in enumerate(class_probabilities):
                        # Create a source domain class for this source class index
                        source_class = DomainClass(
                            (np.intp(model_domain_id), np.intp(source_class_idx))
                        )

                        # If below the threshold, skip the relationship
                        if confidence < threshold:
                            continue

                        # Add the relationship to the taxonomy graph with its confidence value
                        obj._add_relationship(
                            Relationship(
                                (target_class, source_class, float(confidence))
                            )
                        )
                else:
                    raise ValueError(
                        f"Unknown relationship type: {relationship_type}. "
                        "Supported types are 'mcfp' and 'threshold'."
                    )

        return obj

    @staticmethod
    def _form_correlation_matrix(
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


class SyntheticTaxonomy(Taxonomy):
    """Class for building a synthetic taxonomy from merging concepts.
    This taxonomy is constructed by merging atomic concepts into synthetic classes,
    simulating a real-world dataset domain.
    """

    def __init__(
        self,
        domains: List[Deviation],
        domain_labels: Dict[int, List[str]] | None = None,
    ):
        """Initialize the SyntheticTaxonomy with domains and their labels.
        Parameters
        ----------
        domains : List[Deviation]
            List of Deviation objects representing synthetic domains.
            Each Deviation contains multiple DeviationClasses, which are sets of atomic concepts.
        domain_labels : Dict[int, List[str]], optional
            Optional dictionary mapping domain IDs to lists of human-readable class labels.
            If provided, these labels will be used for domain classes instead of class IDs.
            If not provided, class IDs will be used as labels.
        """

        super().__init__(domain_labels=domain_labels)
        self.domains = domains

    @classmethod
    def create_synthetic_taxonomy(
        cls,
        num_atomic_concepts: int,
        num_domains: int,
        domain_class_count_mean: float,
        domain_class_count_variance: float,
        concept_cluster_size_mean: float,
        concept_cluster_size_variance: float,
        no_prediction_class: bool = False,
        relationship_type: _SyntheticRelationshipFilteringMethod = "true",
        atomic_concept_labels: List[str] | None = None,
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
        no_prediction_class : bool, optional
            Some datasets have a no-prediction class which means
            we need to distribute probabilities differently.
        relationship_type : _RELATIONSHIP_TYPE, optional
            The type of relationship to establish between domains. Defaults to "mcfp".
        atomic_concept_labels : List[str], optional
            Optional list of labels for atomic concepts.
            If provided, these labels will be used for atomic concepts
            instead of numeric indices.
        random_seed : int, optional
            Seed for random number generation, by default 42
        """

        rng = np.random.default_rng(random_seed)

        # Generate synthetic domains (previously called deviations)
        domains = [
            SyntheticTaxonomy._create_domain(
                domain_class_count_mean=domain_class_count_mean,
                domain_class_count_variance=domain_class_count_variance,
                num_atomic_concepts=num_atomic_concepts,
                rng=rng,
                concept_cluster_size_mean=concept_cluster_size_mean,
                concept_cluster_size_variance=concept_cluster_size_variance,
            )
            for _ in range(num_domains)
        ]

        # Calculate simulated prediction probabilities between all domain pairs
        # (excluding self-predictions)
        cross_domain_predictions: list[SimulatedPredictions] = []
        for source_domain_id, source_domain in enumerate(domains):
            for target_domain_id, target_domain in enumerate(domains):
                if source_domain_id == target_domain_id:
                    continue
                prediction_matrix = cls._simulate_predictions(
                    source_domain, target_domain, no_prediction_class
                )
                cross_domain_predictions.append(
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
        for domain_id, domain in enumerate(domains):
            class_labels = []
            for class_concepts in domain:
                concept_names = (
                    class_concepts
                    if atomic_concept_labels is None
                    else [atomic_concept_labels[i] for i in class_concepts]
                )

                # Format each class as a set of its concept names
                class_labels.append("{" + ", ".join(map(str, concept_names)) + "}")
            domain_labels[domain_id] = class_labels

        obj = cls(domains=domains, domain_labels=domain_labels)

        # Add relationships to the taxonomy graph based on simulated predictions
        for (
            source_domain_id,
            target_domain_id,
            predictions,
        ) in cross_domain_predictions:
            for target_class_id, class_predictions in enumerate(predictions):
                target_class = DomainClass(
                    (np.intp(target_domain_id), np.intp(target_class_id))
                )

                # Add target node for if no relationships are created
                obj._add_node(target_class)

                if relationship_type == "mcfp":
                    # Get the most likely prediction for this class
                    source_class_id = np.argmax(class_predictions)
                    prediction_confidence = float(class_predictions[source_class_id])

                    # Create domain classes for source and target
                    source_class = DomainClass(
                        (np.intp(source_domain_id), np.intp(source_class_id))
                    )

                    # If zero, skip the relationship
                    if prediction_confidence == 0.0:
                        continue

                    # Add the relationship to the taxonomy graph
                    relationship = Relationship(
                        (target_class, source_class, prediction_confidence)
                    )
                    obj._add_relationship(relationship)
                elif relationship_type == "true":
                    # Iterate through all source classes and their probabilities
                    for source_class_id, confidence in enumerate(class_predictions):
                        # Create domain classes for source and target
                        source_class = DomainClass(
                            (np.intp(source_domain_id), np.intp(source_class_id))
                        )

                        # If zero, skip the relationship
                        if confidence == 0.0:
                            continue

                        # Add the relationship to the taxonomy graph
                        relationship = Relationship(
                            (target_class, source_class, float(confidence))
                        )
                        obj._add_relationship(relationship)
                else:
                    raise ValueError(
                        f"Unknown relationship type: {relationship_type}. "
                        "Supported types are 'mcfp' and 'threshold'."
                    )

        return obj

    @staticmethod
    def _simulate_predictions(
        source_domain: Deviation,
        target_domain: Deviation,
        no_prediction_class: bool = False,
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
        no_prediction_class : bool, optional
            Whether the target domain has a no-prediction class
            that receives all probabilities not assigned to other classes.

        Returns
        -------
        npt.NDArray[np.float64]
            2D array of prediction probabilities where:
              - Each row corresponds to a class in the target domain
              - Each column corresponds to a class in the source domain
              - Value [i,j] is the probability of predicting source class j for target class i
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
            if (
                remaining_probability > 0
                and len(source_domain) > 0
                and not no_prediction_class
            ):
                even_distribution = remaining_probability / len(source_domain)
                class_probabilities = [
                    p + even_distribution for p in class_probabilities
                ]

            prediction_probabilities.append(class_probabilities)

        return np.array(prediction_probabilities, dtype=np.float64)

    @staticmethod
    def _create_domain(
        domain_class_count_mean: float,
        domain_class_count_variance: float,
        num_atomic_concepts: int,
        rng: np.random.Generator,
        concept_cluster_size_mean: float,
        concept_cluster_size_variance: float,
    ) -> Deviation:
        """Generate a synthetic domain with classes composed of atomic concepts.

        A domain (previously called deviation) consists of multiple classes, each
        containing a cluster of related atomic concepts.

        Returns
        -------
        Deviation
            A set of DeviationClasses, where each DeviationClass represents
            a class in this domain
        """
        # Determine number of classes for this domain (bounded by total concept count)
        class_count = np.round(
            sample_truncated_normal(
                mean=domain_class_count_mean,
                variance=domain_class_count_variance,
                upper_bound=num_atomic_concepts,
                lower_bound=1,
                rng=rng,
            )
        ).astype(np.intp)

        # Select which atomic concepts will be part of this domain
        available_concepts = range(num_atomic_concepts)
        selected_concepts = set(
            rng.choice(available_concepts, size=class_count, replace=False)
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
                    sample_truncated_normal(
                        mean=concept_cluster_size_mean,
                        variance=concept_cluster_size_variance,
                        lower_bound=1,
                        upper_bound=len(selected_concepts),
                        rng=rng,
                    )
                ).astype(np.intp)

            # Randomly select concepts for this class
            class_concepts = frozenset(
                rng.choice(
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
