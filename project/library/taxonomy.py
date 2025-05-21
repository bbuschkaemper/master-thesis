import pickle
from typing import List, Tuple, Dict
import numpy as np
import numpy.typing as npt
import networkx as nx
from pyvis.network import Network


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

    Examples
    --------
    >>> # Class 7 from domain 0 (e.g., "horse" class from CIFAR-100)
    >>> domain_class = DomainClass((0, 7))
    """


class UniversalClass(frozenset[DomainClass]):
    """A class in the universal taxonomy represented as a set of domain classes.

    UniversalClass objects are created during the taxonomy building process to represent
    conceptual groupings that span across multiple domains. They contain one or more
    DomainClass objects that have been identified as semantically similar based on
    cross-domain prediction patterns.

    A UniversalClass is implemented as a frozenset (immutable set) of DomainClass objects,
    ensuring it can be used as a dictionary key or in other set operations.

    Examples
    --------
    >>> # A universal class combining "dog" from CIFAR-100 and "dog" from Caltech-256
    >>> dog_class = UniversalClass(frozenset({
    >>>     DomainClass((0, 5)),  # "dog" in CIFAR-100
    >>>     DomainClass((1, 42))  # "dog" in Caltech-256
    >>> }))
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

    Notes
    -----
    The direction is target → source, which matches how predictions flow:
    a class from one domain (target) is being predicted as a class from another domain (source).

    Examples
    --------
    >>> # DomainClass(0,5) is predicted as DomainClass(1,3) with 0.8 confidence
    >>> relationship = Relationship((DomainClass((0,5)), DomainClass((1,3)), 0.8))
    """


class Taxonomy:
    """A class representing relationships between classes from different domains.

    The Taxonomy class builds a graph structure that captures relationships between
    classes from different domains (e.g., different datasets like CIFAR-100 and Caltech-256).
    It analyzes cross-domain predictions to identify conceptual similarities and builds
    a universal taxonomy that unifies classes across domains.

    The graph structure consists of:
    - Nodes: DomainClass and UniversalClass objects
    - Edges: Directed relationships with confidence weights

    This taxonomy can be visualized, serialized, and manipulated through various methods.
    """

    def __init__(
        self,
        cross_domain_predictions: List[Tuple[int, int, npt.NDArray[np.intp]]],
        domain_targets: List[Tuple[int, npt.NDArray[np.intp]]],
        domain_labels: Dict[int, List[str]] | None = None,
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
            Dictionary mapping domain IDs to human-readable labels for each class.
            This is used for visualization purposes. If not provided, class IDs are used.

        Notes
        -----
        The initialization process:
        1. Validates input shapes
        2. Builds correlation matrices between domains
        3. Extracts the most common cross-domain predictions
        4. Constructs initial relationships in the taxonomy graph
        """

        # Store domain labels for visualization
        self.domain_labels = domain_labels

        # Store domain targets in a dictionary for easier access
        self.targets = {}
        for domain_id, targets in domain_targets:
            self.targets[domain_id] = targets

        # Initialize the NetworkX graph for storing taxonomy relationships
        self.graph = nx.DiGraph()

        # Process each cross-domain prediction
        for model_domain_id, dataset_domain_id, predictions in cross_domain_predictions:
            if dataset_domain_id not in self.targets:
                raise ValueError(
                    f"Dataset domain ID {dataset_domain_id} not found in targets."
                )

            # Validate that prediction arrays match their respective target arrays
            dataset_targets = self.targets[dataset_domain_id]
            if predictions.shape != dataset_targets.shape:
                raise ValueError(
                    f"Predictions of domain {model_domain_id} to domain "
                    f"{dataset_domain_id} must match targets in shape. "
                    f"Got {predictions.shape} vs {dataset_targets.shape}"
                )

            # Build correlation matrix between these domains
            correlations = self.__form_correlation_matrix(predictions, dataset_targets)

            # Extract most common predictions and their confidence values
            most_common_classes, confidence_values = (
                self.__foreign_prediction_distributions(correlations)
            )

            # Build initial taxonomy relationships between these domains
            self.__build_initial_relationships(
                domain_id=model_domain_id,
                foreign_domain_id=dataset_domain_id,
                most_common_classes=most_common_classes,
                confidence_values=confidence_values,
            )

    # --------------------------------------
    # Graph construction and relationship handling
    # --------------------------------------

    def __build_initial_relationships(
        self,
        domain_id: int,
        foreign_domain_id: int,
        most_common_classes: npt.NDArray[np.intp],
        confidence_values: npt.NDArray[np.float32],
    ):
        """Builds initial relationships in the taxonomy graph from one domain to another.

        This method creates directed relationships between classes from two different domains
        based on prediction patterns. For each class in the target domain, it creates a
        relationship to the most commonly predicted class in the source domain.

        Parameters
        ----------
        domain_id : int
            ID of the source domain (the domain of the model making predictions)
        foreign_domain_id : int
            ID of the target domain (the domain of the dataset being predicted)
        most_common_classes : npt.NDArray[np.intp]
            For each class in the target domain, the index of the most commonly
            predicted class in the source domain
        confidence_values : npt.NDArray[np.float32]
            Confidence values (probabilities between 0 and 1) for each relationship

        Notes
        -----
        Relationships with zero confidence are skipped. This can happen when there are
        no successful predictions for a particular class.
        """
        for target_class_idx, source_class_idx in enumerate(most_common_classes):
            # Skip relationships with zero confidence
            if confidence_values[target_class_idx] == 0:
                continue

            # Create source domain class (from the model's domain)
            source_class = DomainClass((np.intp(domain_id), np.intp(source_class_idx)))

            # Create target domain class (from the dataset being predicted)
            target_class = DomainClass(
                (np.intp(foreign_domain_id), np.intp(target_class_idx))
            )

            # Add the relationship to the taxonomy graph with its confidence value
            relationship_confidence = confidence_values[target_class_idx]
            self._add_relationship(
                Relationship((target_class, source_class, relationship_confidence))
            )

    # --------------------------------------
    # Graph query methods
    # --------------------------------------

    def _add_relationship(self, relationship: Relationship):
        """Adds a relationship to the graph.

        Parameters
        ----------
        relationship : Relationship
            The relationship to add to the graph (target, source, weight)
        """
        target, source, weight = relationship

        # Add nodes if they don't exist
        if not self.graph.has_node(source):
            self.graph.add_node(source, node_obj=source)
        if not self.graph.has_node(target):
            self.graph.add_node(target, node_obj=target)

        # Add the edge with weight attribute
        self.graph.add_edge(target, source, weight=float(weight))

    def __remove_relationship(self, relationship: Relationship):
        """Removes a relationship from the graph.

        Parameters
        ----------
        relationship : Relationship
            The relationship to remove from the graph (target, source, weight)
        """
        target, source, _ = relationship
        if self.graph.has_edge(target, source):
            self.graph.remove_edge(target, source)

    def __get_relationships(self) -> list[Relationship]:
        """Returns all relationships in the graph.

        Returns
        -------
        list[Relationship]
            The relationships of the graph
        """
        relationships = []
        for u, v, data in self.graph.edges(data=True):
            relationships.append(Relationship((u, v, float(data["weight"]))))
        return relationships

    def __get_nodes(self) -> set[Class]:
        """Returns all nodes in the graph.

        Returns
        -------
        set[Class]
            All nodes in the graph
        """
        return set(self.graph.nodes())

    def __get_relationships_from(self, target: Class) -> list[Relationship]:
        """Returns all outgoing relationships from a node.

        Parameters
        ----------
        target : Class
            The node to get the outgoing relationships from

        Returns
        -------
        list[Relationship]
            Outgoing relationships from the node
        """
        relationships = []
        if target not in self.graph:
            return relationships

        for _, source in self.graph.out_edges(target):
            weight = self.graph.edges[target, source]["weight"]
            relationships.append(Relationship((target, source, float(weight))))
        return relationships

    def __get_relationships_to(self, source: Class) -> list[Relationship]:
        """Returns all incoming relationships to a node.

        Parameters
        ----------
        source : Class
            The node to get the incoming relationships to

        Returns
        -------
        list[Relationship]
            Incoming relationships to the node
        """
        relationships = []
        if source not in self.graph:
            return relationships

        for target, _ in self.graph.in_edges(source):
            weight = self.graph.edges[target, source]["weight"]
            relationships.append(Relationship((target, source, float(weight))))
        return relationships

    def __get_relationship(self, target: Class, source: Class) -> Relationship | None:
        """Returns the relationship between two nodes if it exists.

        Parameters
        ----------
        target : Class
            The source node of the relationship (where the edge starts)
        source : Class
            The target node of the relationship (where the edge ends)

        Returns
        -------
        Relationship | None
            The relationship if it exists, None otherwise
        """
        if self.graph.has_edge(target, source):
            weight = self.graph.edges[target, source]["weight"]
            return Relationship((target, source, float(weight)))
        return None

    # --------------------------------------
    # Data analysis and processing
    # --------------------------------------

    @staticmethod
    def __form_correlation_matrix(
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

        Notes
        -----
        This matrix is the foundation for identifying relationships between
        classes across domains. High counts in a cell indicate a strong relationship
        between those classes.
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

    @staticmethod
    def __foreign_prediction_distributions(
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

        Examples
        --------
        If class 5 in the target domain was predicted as class 2 in the source domain
        60% of the time, the most_common_classes array would have value 2 at index 5,
        and the confidence_values array would have value 0.6 at index 5.
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

    # --------------------------------------
    # Universal taxonomy building
    # --------------------------------------

    def build_universal_taxonomy(self):
        """Builds a universal taxonomy graph from the initial domain relationships.

        This method transforms the initial graph of domain-to-domain relationships
        into a graph where domain classes are connected to universal classes that
        represent shared concepts across domains. The algorithm iteratively applies
        a series of rules in a specific order until no more changes can be made:

        1. Isolated nodes: Create singleton universal classes for domain classes
           with no connections
        2. Bidirectional relationships: Merge domain classes with bidirectional
           relationships into a shared universal class
        3. Transitive cycles: Break cycles by removing weaker relationships
           that would create inconsistencies
        4. Unilateral domain relationships: Transform remaining domain-to-domain
           relationships into proper universal class hierarchies

        Each rule is applied until it can't make any more changes, then the next rule
        is tried. This process repeats until a complete iteration where no rule
        can make further changes.

        Notes
        -----
        The universal taxonomy represents a higher-level organization where classes
        from different domains that represent similar concepts are grouped into
        universal classes. This enables knowledge transfer and alignment between
        different classification systems.
        """
        while True:
            # Flag to track if any modifications were made in this iteration
            changes_made = False

            # Rule 1: Handle isolated domain nodes
            changes_made = self.__handle_isolated_nodes() or changes_made
            if changes_made:
                continue

            # Rule 2: Process bidirectional relationships
            changes_made = self.__handle_bidirectional_relationships() or changes_made
            if changes_made:
                continue

            # Rule 3: Handle problematic transitive relationships
            changes_made = self.__handle_transitive_cycles() or changes_made
            if changes_made:
                continue

            # Rule 4: Process unilateral domain-to-domain relationships
            changes_made = self.__handle_unilateral_relationships() or changes_made
            if changes_made:
                continue

            # If no changes were made in this iteration, we're done
            if not changes_made:
                break

    def __handle_isolated_nodes(self) -> bool:
        """Handle isolated domain nodes by creating singleton universal classes.

        This rule identifies domain classes that have no incoming or outgoing
        relationships (completely isolated nodes). For each such node, it creates
        a singleton universal class containing only that domain class.

        Example transformation:
        Before: DomainClass(A) (isolated)
        After:  DomainClass(A) → UniversalClass({A})

        Returns
        -------
        bool
            True if any changes were made, False otherwise
        """
        for node in self.__get_nodes():
            # Only process domain classes without any connections
            if (
                isinstance(node, DomainClass)
                and not self.__get_relationships_to(node)
                and not self.__get_relationships_from(node)
            ):
                # Create a new universal class containing just this node
                universal_class = UniversalClass(frozenset({node}))
                self._add_relationship(Relationship((node, universal_class, 1.0)))
                return True  # Changes were made
        return False

    def __handle_bidirectional_relationships(self) -> bool:
        """Process bidirectional relationships by creating shared universal classes.

        This rule identifies pairs of classes that have bidirectional mappings
        (A→B and B→A). These pairs likely represent the same concept across different
        domains and should be merged into a shared universal class.

        Example transformation:
        Before: DomainClass(A) → DomainClass(B)
                DomainClass(B) → DomainClass(A)
        After:  DomainClass(A) → UniversalClass({A,B})
                DomainClass(B) → UniversalClass({A,B})

        Returns
        -------
        bool
            True if any changes were made, False otherwise
        """
        for relationship in self.__get_relationships():
            class_a, class_b, _ = relationship

            # Check if there's a reverse relationship
            reverse_relationship = self.__get_relationship(class_b, class_a)
            if not reverse_relationship:
                continue

            # Create a universal class that combines both classes
            classes_from_b = set()
            classes_from_a = set()

            # Extract classes from class_B (could be domain class or universal class)
            if isinstance(class_b, UniversalClass):
                classes_from_b.update(class_b)
            else:
                classes_from_b.add(class_b)

            # Extract classes from class_A (could be domain class or universal class)
            if isinstance(class_a, UniversalClass):
                classes_from_a.update(class_a)
            else:
                classes_from_a.add(class_a)

            # Create new universal class with all contained classes
            combined_classes = classes_from_b.union(classes_from_a)
            universal_class = UniversalClass(frozenset(combined_classes))

            # Set weight as the average of the two relationships
            average_weight = (relationship[2] + reverse_relationship[2]) / 2.0

            # Add relationships from original nodes to the new universal class
            self._add_relationship(
                Relationship((class_b, universal_class, average_weight))
            )
            self._add_relationship(
                Relationship((class_a, universal_class, average_weight))
            )

            # Remove the bidirectional relationships
            self.__remove_relationship(relationship)
            self.__remove_relationship(reverse_relationship)

            return True  # Changes were made

        return False

    def __handle_transitive_cycles(self) -> bool:
        """Handle problematic transitive relationships that could create cycles.

        This rule identifies and resolves potential inconsistencies in the taxonomy.
        If we have a chain A→B→C where A and C are in the same domain, this creates
        a problematic situation because classes in the same domain must be disjoint.

        The method resolves this by comparing the confidence weights of the relationships
        and removing the weaker link to break the cycle.

        Example:
        If DomainClass(D1,1) → DomainClass(D2,5) → DomainClass(D1,7),
        and the weights are 0.7 and 0.9 respectively, the first relationship
        is removed since it has the lower confidence.

        Returns
        -------
        bool
            True if any changes were made, False otherwise
        """
        for relationship in self.__get_relationships():
            class_a, class_b, first_relationship_weight = relationship

            # Skip if the first class is not a domain class
            if not isinstance(class_a, DomainClass):
                continue

            # Get relationships originating from class_B
            next_relationships = self.__get_relationships_from(class_b)

            # Only consider relationships where the destination is a domain class
            next_relationships = [
                rel for rel in next_relationships if isinstance(rel[1], DomainClass)
            ]

            if not next_relationships:
                continue

            domain_id_a = class_a[0]  # Domain ID of the first class

            # Check for potential cycles where A→B→C and A and C are in the same domain
            for next_relationship in next_relationships:
                _, class_c, second_relationship_weight = next_relationship
                if not isinstance(class_c, DomainClass):
                    raise ValueError("Expected DomainClass")

                domain_id_c = class_c[0]  # Domain ID of the final class

                # Only break cycles when A and C are in the same domain
                if domain_id_c == domain_id_a:
                    # Remove the weaker relationship
                    if first_relationship_weight < second_relationship_weight:
                        self.__remove_relationship(relationship)
                    else:
                        self.__remove_relationship(next_relationship)

                    return True  # Changes were made

        return False

    def __handle_unilateral_relationships(self) -> bool:
        """Process unilateral domain-to-domain relationships into universal relationships.

        For unidirectional relationships between domain classes (A→B), this method creates:
        1. A shared universal class containing both domain classes
        2. A separate universal class containing only the source domain class

        This structure allows domain classes to be connected to the appropriate universal classes
        while preserving the hierarchical relationship implied by the original link.

        Example transformation:
        Before: DomainClass(A) → DomainClass(B)
        After:  DomainClass(A) → UniversalClass({A,B})
               DomainClass(A) → UniversalClass({A})
               DomainClass(B) → UniversalClass({A,B})

        Returns
        -------
        bool
            True if any changes were made, False otherwise
        """
        for relationship in self.__get_relationships():
            domain_class_a, domain_class_b, _ = relationship

            # Only process domain-to-domain relationships
            if not isinstance(domain_class_b, DomainClass) or not isinstance(
                domain_class_a, DomainClass
            ):
                continue

            # Create a universal class containing both domain classes
            shared_universal_class = UniversalClass(
                frozenset({domain_class_b, domain_class_a})
            )

            # Create a universal class containing only the source domain class
            source_universal_class = UniversalClass(frozenset({domain_class_b}))

            # Add relationships to the new universal classes
            self._add_relationship(
                Relationship((domain_class_b, shared_universal_class, relationship[2]))
            )
            self._add_relationship(
                Relationship((domain_class_a, shared_universal_class, relationship[2]))
            )
            self._add_relationship(
                Relationship((domain_class_b, source_universal_class, 1.0))
            )

            # Remove the original relationship
            self.__remove_relationship(relationship)

            return True  # Changes were made

        return False

    # --------------------------------------
    # Persistence and I/O
    # --------------------------------------

    def save(self, filepath: str):
        """Save the taxonomy graph to a file (pickle format).

        Parameters
        ----------
        filepath : str
            Path where to save the graph
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.graph, f)

    @classmethod
    def load(cls, filepath: str) -> "Taxonomy":
        """Load a taxonomy from a file.

        Parameters
        ----------
        filepath : str
            Path to the saved taxonomy

        Returns
        -------
        Taxonomy
            The loaded taxonomy
        """
        # Create an empty taxonomy
        taxonomy = cls(
            cross_domain_predictions=[],
            domain_targets=[],
        )

        # Load the graph directly
        with open(filepath, "rb") as f:
            taxonomy.graph = pickle.load(f)

        return taxonomy

    # --------------------------------------
    # Visualization
    # --------------------------------------

    def visualize_graph(
        self,
        title: str = "Universal Taxonomy Graph",
        height: int = 800,
        width: int = 1200,
    ) -> Network:
        """Visualizes the taxonomy graph using PyVis Network.

        Creates an interactive HTML visualization of the taxonomy graph with nodes colored
        by domain and edges showing relationship strengths.

        Parameters
        ----------
        title : str, optional
            Title to display on the visualization, by default "Universal Taxonomy Graph"
        height : int, optional
            Height of the visualization in pixels, by default 800
        width : int, optional
            Width of the visualization in pixels, by default 1200

        Returns
        -------
        Network
            PyVis Network object that can be displayed or saved to HTML

        Notes
        -----
        - Different domain nodes are colored with distinct colors
        - Universal class nodes are colored salmon
        - Edge weights are displayed as labels on the connections
        """

        # Step 1: Create human-readable labels for all nodes
        node_labels = self.__create_node_labels()

        # Step 2: Set node colors and groups based on domain or universal class
        node_colors, node_groups = self.__assign_node_colors_and_groups()

        # Step 3: Create and configure the PyVis network
        network = Network(height=height, width=width, directed=True)  # type: ignore
        network.heading = title

        # Step 4: Add nodes with their styling
        self.__add_nodes_to_visualization(
            network, node_labels, node_colors, node_groups
        )

        # Step 5: Add edges with weight labels
        self.__add_edges_to_visualization(network)

        # Enable physics for better layout
        network.toggle_physics(True)

        return network

    def __create_node_labels(
        self,
    ) -> dict:
        """Create human-readable labels for all nodes in the graph.

        Returns
        -------
        dict
            Dictionary mapping nodes to their display labels
        """
        node_labels = {}
        domain_labels = self.domain_labels or {}

        # First pass: Create labels for domain classes
        for node in self.graph.nodes():
            if isinstance(node, DomainClass):
                domain_id, class_id = node
                domain_id = int(domain_id)
                class_id = int(class_id)

                # Use provided human-readable labels if available
                if domain_id in domain_labels and class_id < len(
                    domain_labels[domain_id]
                ):
                    label = domain_labels[domain_id][class_id]
                else:
                    # Fall back to class ID if no label is available
                    label = f"{class_id}"

                node_labels[node] = f"D{domain_id}:{label}"

        # Second pass: Create labels for universal classes using domain class labels
        for node in self.graph.nodes():
            if isinstance(node, UniversalClass):
                # Join the labels of all domain classes in this universal class
                class_labels = []
                for domain_class in node:
                    if domain_class in node_labels:
                        class_labels.append(node_labels[domain_class])
                    else:
                        class_labels.append(str(domain_class))

                # Format as a set with curly braces
                node_labels[node] = "{" + ", ".join(class_labels) + "}"

        return node_labels

    def __assign_node_colors_and_groups(self) -> tuple[list, list]:
        """Assign colors and groups to nodes based on their domain or type.

        Returns
        -------
        tuple[list, list]
            Lists of colors and groups for each node
        """
        node_colors = []
        node_groups = []

        # Define a color palette for different domains
        domain_colors = [
            "skyblue",  # Domain 0
            "lightgreen",  # Domain 1
            "lightyellow",  # Domain 2
            "lightpink",  # Domain 3
            "lightcyan",  # Domain 4
            "thistle",  # Domain 5
            "peachpuff",  # Domain 6
            "lightcoral",  # Domain 7
            "lightsteelblue",  # Domain 8
            "palegreen",  # Domain 9
        ]

        # Process each node in the graph
        for node in self.graph.nodes():
            if isinstance(node, DomainClass):
                domain_id = node[0]
                # Get color for this domain (cycling through colors if needed)
                color = domain_colors[domain_id % len(domain_colors)]
                node_colors.append(color)
                node_groups.append(f"Domain {domain_id}")
            else:  # Universal class
                node_colors.append("salmon")
                node_groups.append("Universal")

        return node_colors, node_groups

    def __add_nodes_to_visualization(
        self,
        network: Network,
        node_labels: dict,
        node_colors: list,
        node_groups: list,
    ) -> None:
        """Add nodes to the PyVis network with appropriate styling.

        Parameters
        ----------
        network : Network
            PyVis network object
        node_labels : dict
            Dictionary of node labels
        node_colors : list
            List of colors for each node
        node_groups : list
            List of group names for each node
        """
        for i, node in enumerate(self.graph.nodes()):
            # Add the node with appropriate styling
            network.add_node(
                str(node),  # Node ID (needs to be a string for PyVis)
                label=node_labels[node],  # Human-readable label
                color=node_colors[i],  # Color based on domain/type
                title=node_labels[node],  # Tooltip text
                group=node_groups[i],  # Group for layout algorithms
            )

    def __add_edges_to_visualization(self, network: Network) -> None:
        """Add edges to the PyVis network with weight labels.

        Parameters
        ----------
        network : Network
            PyVis network object
        """
        for target, source, data in self.graph.edges(data=True):
            weight = data.get("weight", 1.0)
            # Format weight to 2 decimal places for display
            weight_label = f"{weight:.2f}"

            # Add the edge with the weight as both the label and tooltip
            network.add_edge(
                str(target),  # Target node ID
                str(source),  # Source node ID
                title=weight_label,  # Tooltip showing weight
                label=weight_label,  # Edge label showing weight
                value=weight,  # Numeric weight (affects edge thickness)
            )
