import pickle
from typing import List, Tuple, Dict
import numpy as np
import numpy.typing as npt
import networkx as nx
from pyvis.network import Network


class DomainClass(tuple[np.intp, np.intp]):
    """A class from a specific domain represented as a tuple (domain_id, class_id).

    The first element (domain_id) identifies which domain the class belongs to.
    The second element (class_id) is the identifier of the class within its domain.
    """


class UniversalClass(frozenset[DomainClass]):
    """A class in the universal taxonomy represented as a set of domain classes.

    Universal classes are created during the taxonomy building process and represent
    relationships between classes from different domains that share similar concepts.
    """


type Class = DomainClass | UniversalClass
"""Type alias representing either a domain-specific class or a universal class."""


class Relationship(tuple[Class, Class, float]):
    """A directional relationship between two classes with an associated confidence weight.
    In the graph, the edge goes from the target class to the source class.

    Represented as a tuple (source_class, target_class, weight) where:
    - source_class: The originating class
    - target_class: The destination class
    - weight: The confidence/probability of the relationship (between 0 and 1)
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

        Parameters
        ----------
        domain_id : int
            ID of the source domain (model making predictions)
        foreign_domain_id : int
            ID of the target domain (dataset being predicted)
        most_common_classes : npt.NDArray[np.intp]
            For each class in the target domain,
            the most commonly predicted class in the source domain
        confidence_values : npt.NDArray[np.float32]
            Confidence/probability values for each relationship
        """
        for target_class_idx, source_class_idx in enumerate(most_common_classes):
            # Create source and target domain classes
            source_class = DomainClass((np.intp(domain_id), np.intp(source_class_idx)))
            target_class = DomainClass(
                (np.intp(foreign_domain_id), np.intp(target_class_idx))
            )

            # Skip relationships with zero confidence
            if confidence_values[target_class_idx] == 0:
                continue

            # Add the relationship to the taxonomy graph
            self._add_relationship(
                Relationship(
                    (source_class, target_class, confidence_values[target_class_idx])
                )
            )

    def _add_relationship(self, relationship: Relationship):
        """Adds a relationship to the graph.

        Parameters
        ----------
        relationship : Relationship
            The relationship to add to the graph (source, target, weight)
        """
        source, target, weight = relationship

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
            The relationship to remove from the graph (source, target, weight)
        """
        source, target, _ = relationship
        if self.graph.has_edge(target, source):
            self.graph.remove_edge(target, source)

    def __redirect_incoming_relationships(self, old_source: Class, new_source: Class):
        """Redirect all incoming relationships from old_source to new_source.

        Parameters
        ----------
        old_source : Class
            The original source node
        new_source : Class
            The new source node to redirect relationships to
        """
        for rel in self.__get_relationships_to(old_source):
            self.__remove_relationship(rel)
            self._add_relationship(Relationship((new_source, rel[1], rel[2])))

    # --------------------------------------
    # Graph query methods
    # --------------------------------------

    def __get_relationships(self) -> list[Relationship]:
        """Returns all relationships in the graph.

        Returns
        -------
        list[Relationship]
            The relationships of the graph
        """
        relationships = []
        for u, v, data in self.graph.edges(data=True):
            relationships.append(Relationship((v, u, float(data["weight"]))))
        return relationships

    def __get_nodes(self) -> set[Class]:
        """Returns all nodes in the graph.

        Returns
        -------
        set[Class]
            All nodes in the graph
        """
        return set(self.graph.nodes())

    def __get_relationships_from(self, node: Class) -> list[Relationship]:
        """Returns all outgoing relationships from a node.

        Parameters
        ----------
        node : Class
            The node to get the relationships from

        Returns
        -------
        list[Relationship]
            Outgoing relationships from the node
        """
        relationships = []
        if node not in self.graph:
            return relationships

        for _, source in self.graph.out_edges(node):
            weight = self.graph.edges[node, source]["weight"]
            relationships.append(Relationship((source, node, float(weight))))
        return relationships

    def __get_relationships_to(self, node: Class) -> list[Relationship]:
        """Returns all incoming relationships to a node.

        Parameters
        ----------
        node : Class
            The node to get the relationships to

        Returns
        -------
        list[Relationship]
            Incoming relationships to the node
        """
        relationships = []
        if node not in self.graph:
            return relationships

        for target, _ in self.graph.in_edges(node):
            weight = self.graph.edges[target, node]["weight"]
            relationships.append(Relationship((node, target, float(weight))))
        return relationships

    def __get_relationship(
        self, from_node: Class, to_node: Class
    ) -> Relationship | None:
        """Returns the relationship between two nodes if it exists.

        Parameters
        ----------
        from_node : Class
            The starting node of the relationship
        to_node : Class
            The ending node of the relationship

        Returns
        -------
        Relationship | None
            The relationship if it exists, None otherwise
        """
        if self.graph.has_edge(from_node, to_node):
            weight = self.graph.edges[from_node, to_node]["weight"]
            return Relationship((to_node, from_node, float(weight)))
        return None

    # --------------------------------------
    # Data analysis and processing
    # --------------------------------------

    @staticmethod
    def __form_correlation_matrix(
        predictions: npt.NDArray[np.intp],
        targets: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.intp]:
        """Forms a correlation matrix for predictions on a foreign domain.

        Each row represents a true class in the foreign domain, and each column
        represents a predicted class from the model's domain. The value at position
        (i,j) indicates how many times class i was predicted as class j.

        Parameters
        ----------
        predictions : npt.NDArray[np.intp]
            Model predictions using own domain labels on foreign domain data
        targets : npt.NDArray[np.intp]
            True labels of the foreign domain

        Returns
        -------
        npt.NDArray[np.intp]
            The correlation matrix of shape (n_classes_foreign, n_classes_own)
        """
        # Find the number of unique classes in both predictions and targets
        n_target_classes = np.max(targets) + 1
        n_prediction_classes = np.max(predictions) + 1

        # Initialize correlation matrix with zeros
        correlations = np.zeros(
            (n_target_classes, n_prediction_classes),
            dtype=np.intp,
        )

        # Count occurrences of each (target, prediction) pair
        for i, pred in enumerate(predictions):
            correlations[targets[i], pred] += 1

        return correlations

    @staticmethod
    def __foreign_prediction_distributions(
        correlations: npt.NDArray[np.intp],
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]:
        """Calculates the most common prediction for each class and its confidence.

        For each class in the foreign domain, identifies which class from the model's
        domain is most commonly predicted, and calculates the confidence of that
        prediction.

        Parameters
        ----------
        correlations : npt.NDArray[np.intp]
            The correlation matrix where each element (i,j) represents how many times
            foreign class i was predicted as own-domain class j

        Returns
        -------
        tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]
            A tuple containing:
            - Array of most commonly predicted classes for each foreign class
            - Array of confidence values (probabilities) for those predictions
        """
        n_foreign_classes = correlations.shape[0]
        most_common_classes = np.zeros(n_foreign_classes, dtype=np.intp)
        confidence_values = np.zeros(n_foreign_classes, dtype=np.float32)

        # For each class in the foreign domain
        for foreign_class_idx in range(n_foreign_classes):
            # Find the most commonly predicted class
            prediction_counts = correlations[foreign_class_idx, :]
            most_common_classes[foreign_class_idx] = np.argmax(prediction_counts)

            # Calculate confidence as the proportion of predictions for this class
            total_predictions = np.sum(prediction_counts)
            if total_predictions > 0:
                max_count = prediction_counts[most_common_classes[foreign_class_idx]]
                confidence_values[foreign_class_idx] = max_count / total_predictions
            else:
                confidence_values[foreign_class_idx] = 0.0

        return most_common_classes, confidence_values

    # --------------------------------------
    # Universal taxonomy building
    # --------------------------------------

    def build_universal_taxonomy(self):
        """Builds a universal taxonomy graph from the initial domain relationships.

        This method transforms the initial graph of domain-to-domain relationships
        into a graph where all relationships are from domain classes to universal classes.
        The algorithm iteratively applies a series of rules to resolve relationships:

        1. Isolated nodes: Create singleton universal classes
        2. Bidirectional relationships: Merge classes into a shared universal class
        3. Transitive cycles: Break cycles by removing lower-weight relationships
        4. Unilateral domain relationships: Transform into proper universal relationships

        The process continues until no more changes can be made to the graph.

        Notes
        -----
        The universal taxonomy represents a higher-level organization where classes
        from different domains that represent similar concepts are grouped into
        universal classes.
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

            # If no changes were made we are done
            if not changes_made:
                break

    def __handle_isolated_nodes(self) -> bool:
        """Handle isolated domain nodes by creating singleton universal classes.

        This rule processes domain classes that have no relationships, creating
        a universal class for each isolated node.

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
                self._add_relationship(Relationship((universal_class, node, 1.0)))
                return True  # Changes were made
        return False

    def __handle_bidirectional_relationships(self) -> bool:
        """Process bidirectional relationships by creating shared universal classes.

        If two classes have bidirectional mappings (A→B and B→A), they likely
        represent the same concept and should be merged into a universal class.

        This rule is critical for identifying equivalent concepts across domains.

        Returns
        -------
        bool
            True if any changes were made, False otherwise
        """
        for relationship in self.__get_relationships():
            source_node, target_node, _ = relationship

            # Check if there's a reverse relationship
            reverse_rel = self.__get_relationship(source_node, target_node)
            if not reverse_rel:
                continue

            # Create a universal class that combines both classes
            source_classes = set()
            target_classes = set()

            # Extract classes from source node (could be domain class or universal class)
            if isinstance(source_node, UniversalClass):
                source_classes.update(source_node)
            else:
                source_classes.add(source_node)

            # Extract classes from target node (could be domain class or universal class)
            if isinstance(target_node, UniversalClass):
                target_classes.update(target_node)
            else:
                target_classes.add(target_node)

            # Create new universal class with all contained classes
            combined_classes = source_classes.union(target_classes)
            universal_class = UniversalClass(frozenset(combined_classes))

            # Add relationships from original nodes to the new universal class
            self._add_relationship(Relationship((universal_class, source_node, 1.0)))
            self._add_relationship(Relationship((universal_class, target_node, 1.0)))

            # Remove the bidirectional relationships
            self.__remove_relationship(relationship)
            self.__remove_relationship(reverse_rel)

            # Redirect incoming relationships to the new universal class
            self.__redirect_incoming_relationships(source_node, universal_class)
            self.__redirect_incoming_relationships(target_node, universal_class)

            return True  # Changes were made

        return False

    def __handle_transitive_cycles(self) -> bool:
        """Handle problematic transitive relationships that could create cycles.

        If we have A→B→C where A and C are in the same domain, this creates an invalid
        situation because classes in the same domain must be disjoint.
        We resolve this by removing the weaker relationship.

        This rule ensures consistency within domains.

        Returns
        -------
        bool
            True if any changes were made, False otherwise
        """
        for relationship in self.__get_relationships():
            source_node, target_node, source_weight = relationship

            # Skip if target is not a domain class
            if not isinstance(target_node, DomainClass):
                continue

            # Get relationships from source
            next_relationships = self.__get_relationships_from(source_node)

            # Only consider relationships between domain classes
            next_relationships = [
                rel for rel in next_relationships if isinstance(rel[0], DomainClass)
            ]

            if not next_relationships:
                continue

            # Extract the domain ID of the target (A)
            target_domain_id = target_node[0]

            # Check for potential cycles where A→B→C and A and C are in the same domain
            for next_rel in next_relationships:
                next_target, _, next_weight = next_rel
                if not isinstance(next_target, DomainClass):
                    raise ValueError("Expected DomainClass")

                source_domain_id = next_target[0]  # Domain ID of the final target node

                # Only break cycles when A and C are in the same domain
                if source_domain_id == target_domain_id:
                    # Remove the weaker relationship
                    if source_weight < next_weight:
                        self.__remove_relationship(relationship)
                    else:
                        self.__remove_relationship(next_rel)

                    return True  # Changes were made

        return False

    def __handle_unilateral_relationships(self) -> bool:
        """Process unilateral domain-to-domain relationships into universal relationships.

        For relationships like A→B between domain classes, create appropriate universal classes
        to represent the relationship hierarchy.

        This rule transforms the remaining domain-to-domain links into the proper
        universal taxonomy structure.

        Returns
        -------
        bool
            True if any changes were made, False otherwise
        """
        for relationship in self.__get_relationships():
            source_node, target_node, _ = relationship

            # Only process domain-to-domain relationships
            if not isinstance(source_node, DomainClass) or not isinstance(
                target_node, DomainClass
            ):
                continue

            # Create a universal class containing both classes
            shared_universal_class = UniversalClass(
                frozenset({source_node, target_node})
            )

            # Create a universal class containing only the second class
            source_universal_class = UniversalClass(frozenset({source_node}))

            # Add relationships to the new universal classes
            self._add_relationship(
                Relationship((shared_universal_class, source_node, 1.0))
            )
            self._add_relationship(
                Relationship((shared_universal_class, target_node, 1.0))
            )
            self._add_relationship(
                Relationship((source_universal_class, source_node, 1.0))
            )

            # Remove the original relationship
            self.__remove_relationship(relationship)

            # Redirect incoming relationships to the target to the appropriate universal class
            self.__redirect_incoming_relationships(target_node, shared_universal_class)

            # Redirect incoming relationships to the source to both universal classes
            for rel in self.__get_relationships_to(source_node):
                self.__remove_relationship(rel)
                self._add_relationship(
                    Relationship((shared_universal_class, rel[1], rel[2]))
                )
                self._add_relationship(
                    Relationship((source_universal_class, rel[1], rel[2]))
                )

            return True  # Changes were made

        return False

    # --------------------------------------
    # Persistence and I/O
    # --------------------------------------

    def save(self, filepath: str):
        """Save the taxonomy graph to a file.

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
