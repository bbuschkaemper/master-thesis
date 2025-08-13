import numpy as np
import numpy.typing as npt


class DomainClass(tuple[np.intp, np.intp]):
    """A class from a specific domain represented as a tuple (domain_id, class_id).

    A DomainClass represents a single class within a particular data domain.
    For example, in multi-dataset taxonomy building, each dataset would be a different domain.

    Parameters
    ----------
    domain_id : np.intp
        Identifies which domain the class belongs to (e.g., CIFAR-100=0, Caltech-256=1)
    class_id : np.intp
        The identifier of the class within its domain (e.g., "dog"=5, "airplane"=0)
    """


class UniversalClass(frozenset[DomainClass]):
    """A class in the universal taxonomy represented as a set of domain classes.

    UniversalClass objects are created during the taxonomy building process to represent
    conceptual groupings that span across multiple domains. They contain one or more
    DomainClass objects that have been identified as semantically similar based on
    cross-domain prediction patterns.

    A UniversalClass is implemented as a frozenset (immutable set) of DomainClass objects,
    ensuring it can be used as a dictionary key or in other set operations.
    """


type Class = DomainClass | UniversalClass
"""Type alias representing either a domain-specific class or a universal class."""


class Relationship(tuple[Class, Class, float]):
    """A directional relationship between two classes with an associated confidence weight.

    In the taxonomy graph, relationships are represented as directed edges from the target
    class to the source class. The confidence weight indicates the strength or certainty
    of this relationship based on prediction patterns.

    Parameters
    ----------
    target_class : Class
        The originating class (e.g., the class being predicted)
    source_class : Class
        The destination class (e.g., the class that is predicted)
    weight : float
        The confidence/probability of the relationship (between 0 and 1)
    """


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
