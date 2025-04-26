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


class Relationship(tuple[Class, Class, np.float32]):
    """A directional relationship between two classes with an associated confidence weight.

    Represented as a tuple (source_class, target_class, weight) where:
    - source_class: The originating class
    - target_class: The destination class
    - weight: The confidence/probability of the relationship (between 0 and 1)
    """


class Taxonomy:
    def __init__(
        self,
        cross_domain_predictions: List[Tuple[int, int, npt.NDArray[np.intp]]],
        domain_targets: List[Tuple[int, npt.NDArray[np.intp]]],
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

        Notes
        -----
        The initialization process:
        1. Validates input shapes
        2. Builds correlation matrices between domains
        3. Extracts the most common cross-domain predictions
        4. Constructs initial relationships in the taxonomy graph
        """
        # Store domain targets in a dictionary for easier access
        self.targets = {}
        for domain_id, targets in domain_targets:
            self.targets[domain_id] = targets

        # Initialize the NetworkX graph for storing taxonomy relationships
        self.graph = nx.DiGraph()

        # Process each cross-domain prediction
        for model_domain_id, dataset_domain_id, predictions in cross_domain_predictions:
            # Validate that prediction arrays match their respective target arrays
            dataset_targets = self.targets[dataset_domain_id]
            assert (
                predictions.shape == dataset_targets.shape
            ), f"""Predictions of domain {model_domain_id}
             to domain {dataset_domain_id} must match targets in shape"""

            # Build correlation matrix between these domains
            correlations = self.__form_correlation_matrix(predictions, dataset_targets)

            # Extract most common predictions and their confidence values
            most_common_classes, confidence_values = (
                self.__foreign_prediction_distributions(correlations)
            )

            # Build initial taxonomy relationships between these domains
            self.__build_initial_relationships(
                domain_id=dataset_domain_id,
                foreign_domain_id=model_domain_id,
                most_common_classes=most_common_classes,
                confidence_values=confidence_values,
            )

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
            ID of the source domain (0 for A, 1 for B)
        foreign_domain_id : int
            ID of the target domain (0 for A, 1 for B)
        most_common_classes : npt.NDArray[np.intp]
            For each class in the source domain,
            the most commonly predicted class in the target domain
        confidence_values : npt.NDArray[np.float32]
            Confidence/probability values for each relationship
        """
        for class_idx, target_class_idx in enumerate(most_common_classes):
            # Create source and target domain classes
            source_class = DomainClass((np.intp(domain_id), np.intp(class_idx)))
            target_class = DomainClass(
                (np.intp(foreign_domain_id), np.intp(target_class_idx))
            )

            # Skip relationships with zero confidence
            if confidence_values[class_idx] == 0:
                continue

            # Add the relationship to the taxonomy graph
            self.__add_relationship(
                (source_class, target_class, confidence_values[class_idx])
            )

    @staticmethod
    def __form_correlation_matrix(
        predictions: npt.NDArray[np.intp],
        targets: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.intp]:
        """Forms a correlation matrix for predictions on a foreign domain.

        Each row of the correlation matrix corresponds to a class in the foreign domain
        and each column corresponds to a class in the own domain.
        The value of cell (i, j) is the number of times a class in the foreign domain
        i was predicted as a class in the own domain j.
        The correlation matrix is of shape (n_classes_foreign, n_classes_own).

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
        correlations = np.zeros(
            (np.max(targets) + 1, np.max(predictions) + 1),
            dtype=np.intp,
        )

        for i, pred in enumerate(predictions):
            correlations[targets[i], pred] += 1

        return correlations

    @staticmethod
    def __foreign_prediction_distributions(
        correlations: npt.NDArray[np.intp],
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]:
        """Calculates a distribution for the predictions of each own domain class in the foreign
        domain.

        The result is a 1D array of shape (n_classes_foreign,) where the value at index i
        is the own domain class that was predicted the most times for the foreign domain class i
        together with the probability of the prediction.
        The probability is the number of times the class was predicted divided by the total
        number of predictions for that class.
        If there are no predictions for a class, the probability is 0.

        For convenience, the function returns a tuple of two 1D arrays:
        - The first element is a 1D array of shape (n_classes_foreign,) with the most common
          foreign class.
        - The second element is a 1D array of shape (n_classes_foreign,) with the
          probabilities of the predictions (i.e. the distribution).

        Parameters
        ----------
        correlations : npt.NDArray[np.intp]
            The correlation matrix indicating the predictions.

        Returns
        -------
        tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]
            The most common foreign predictions and their probabilities.
            The first element is a 1D array of shape (n_classes_foreign,) with the most common
            foreign class.
            The second element is a 1D array of shape (n_classes_foreign,) with the
            probabilities of the predictions.
        """

        values = np.zeros(correlations.shape[0], dtype=np.intp)
        probabilities = np.zeros(correlations.shape[0], dtype=np.float32)
        for i in range(correlations.shape[0]):
            values[i] = np.argmax(correlations[i, :])
            total_predictions = np.sum(correlations[i, :])
            probabilities[i] = (
                correlations[i, values[i]] / total_predictions
                if total_predictions > 0
                else 0.0
            )

        return values, probabilities

    def __add_relationship(self, relationship: Relationship):
        """Adds a relationship to the graph.

        Parameters
        ----------
        relationship : Relationship
            The relationship to add to the graph.
        """
        source, target, weight = relationship

        # Add nodes if they don't exist
        if not self.graph.has_node(source):
            self.graph.add_node(source, node_obj=source)
        if not self.graph.has_node(target):
            self.graph.add_node(target, node_obj=target)

        # Add the edge with weight attribute
        self.graph.add_edge(source, target, weight=float(weight))

    def __remove_relationship(self, relationship: Relationship):
        """Removes a relationship from the graph.

        Parameters
        ----------
        relationship : Relationship
            The relationship to remove from the graph.
        """
        source, target, _ = relationship
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)

    def __get_relationships(self) -> list[Relationship]:
        """Returns the relationships of the graph.

        Returns
        -------
        list[Relationship]
            The relationships of the graph.
        """
        relationships = []
        for u, v, data in self.graph.edges(data=True):
            relationships.append((u, v, np.float32(data["weight"])))
        return relationships

    def __get_nodes(self) -> set[Class]:
        """Returns the nodes of the graph.

        Returns
        -------
        set[Class]
            The nodes of the graph.
        """
        return set(self.graph.nodes())

    def __get_relationships_from(self, node: Class) -> list[Relationship]:
        """Returns the relationships from a node.

        Parameters
        ----------
        node : Class
            The node to get the relationships from.

        Returns
        -------
        list[Relationship]
            The relationships from the node.
        """
        relationships = []
        if node not in self.graph:
            return relationships

        for _, target in self.graph.out_edges(node):
            weight = self.graph.edges[node, target]["weight"]
            relationships.append((node, target, np.float32(weight)))
        return relationships

    def __get_relationships_to(self, node: Class) -> list[Relationship]:
        """Returns the relationships to a node.

        Parameters
        ----------
        node : Class
            The node to get the relationships to.

        Returns
        -------
        list[Relationship]
            The relationships to the node.
        """
        relationships = []
        if node not in self.graph:
            return relationships

        for source, _ in self.graph.in_edges(node):
            weight = self.graph.edges[source, node]["weight"]
            relationships.append((source, node, np.float32(weight)))
        return relationships

    def __get_relationship(
        self, from_node: Class, to_node: Class
    ) -> Relationship | None:
        """Checks if a relationship exists between two nodes.
        Returns the relationship if it exists, None otherwise.

        Parameters
        ----------
        from_node : Class
            The starting node of the relationship.
        to_node : Class
            The ending node of the relationship.

        Returns
        -------
        Relationship | None
            The relationship if it exists, None otherwise.
        """
        if self.graph.has_edge(from_node, to_node):
            weight = self.graph.edges[from_node, to_node]["weight"]
            return (from_node, to_node, np.float32(weight))
        return None

    def __is_finished(self) -> bool:
        """Checks if the universal taxonomy building process is complete.

        A universal taxonomy is considered complete when:
        1. All relationships are from domain classes to universal classes
        2. All universal classes have at least one incoming relationship

        Returns
        -------
        bool
            True if the taxonomy building is complete, False otherwise
        """
        # Check that all universal classes have at least one incoming relationship
        for node in self.__get_nodes():
            if isinstance(node, UniversalClass):
                if not self.graph.in_edges(node):
                    return False

        # Check that all relationships are from domain classes to universal classes
        for source, target in self.graph.edges():
            if not (
                isinstance(source, DomainClass) and isinstance(target, UniversalClass)
            ):
                return False

        return True

    def build_universal_taxonomy(self):
        """Builds a universal taxonomy graph from the initial domain relationships.

        This method transforms the initial graph of domain-to-domain relationships
        into a graph where all relationships are from domain classes to universal classes.
        The algorithm iteratively applies a series of rules to resolve relationships:

        1. Isolated nodes: Create singleton universal classes
        2. Bidirectional relationships: Merge classes into a shared universal class
        3. Transitive cycles: Break cycles by removing lower-weight relationships
        4. Unilateral domain relationships: Transform into proper universal relationships

        The process continues until all relationships follow the proper structure.
        """
        while not self.__is_finished():
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

            # If no changes were made but we're not finished, there might be a structural issue
            if not changes_made:
                print(
                    "Warning: Universal taxonomy building could not complete. "
                    "Graph structure may be invalid."
                )
                break

    def __handle_isolated_nodes(self) -> bool:
        """Handle isolated domain nodes by creating singleton universal classes.

        Returns
        -------
        bool
            True if any changes were made, False otherwise
        """
        for node in self.__get_nodes():
            # Check for nodes without incoming or outgoing relationships
            if (
                not isinstance(node, UniversalClass)
                and not self.__get_relationships_to(node)
                and not self.__get_relationships_from(node)
            ):

                # Create a new universal class containing just this node
                universal_class = UniversalClass(frozenset({node}))
                self.__add_relationship((node, universal_class, 1.0))
                return True  # Changes were made
        return False

    def __handle_bidirectional_relationships(self) -> bool:
        """Process bidirectional relationships by creating shared universal classes.

        If two classes have bidirectional mappings (A→B and B→A), they likely
        represent the same concept and should be merged into a universal class.

        Returns
        -------
        bool
            True if any changes were made, False otherwise
        """
        for relationship in self.__get_relationships():
            # Check if there's a reverse relationship
            reverse_rel = self.__get_relationship(relationship[1], relationship[0])
            if not reverse_rel:
                continue

            # Create a universal class that combines both classes
            source_classes = set()
            target_classes = set()

            # Extract classes from source node (could be domain class or universal class)
            if isinstance(relationship[0], UniversalClass):
                source_classes.update(relationship[0])
            else:
                source_classes.add(relationship[0])

            # Extract classes from target node (could be domain class or universal class)
            if isinstance(relationship[1], UniversalClass):
                target_classes.update(relationship[1])
            else:
                target_classes.add(relationship[1])

            # Create new universal class with all contained classes
            combined_classes = source_classes.union(target_classes)
            universal_class = UniversalClass(frozenset(combined_classes))

            # Add relationships from original nodes to the new universal class
            self.__add_relationship((relationship[0], universal_class, 1.0))
            self.__add_relationship((relationship[1], universal_class, 1.0))

            # Remove the bidirectional relationships
            self.__remove_relationship(relationship)
            self.__remove_relationship(reverse_rel)

            # Redirect incoming relationships to the new universal class
            self.__redirect_incoming_relationships(relationship[0], universal_class)
            self.__redirect_incoming_relationships(relationship[1], universal_class)

            return True  # Changes were made

        return False

    def __redirect_incoming_relationships(self, old_target: Class, new_target: Class):
        """Redirect all incoming relationships from old_target to new_target.

        Parameters
        ----------
        old_target : Class
            The original target node
        new_target : Class
            The new target node to redirect relationships to
        """
        for rel in self.__get_relationships_to(old_target):
            self.__remove_relationship(rel)
            self.__add_relationship((rel[0], new_target, rel[2]))

    def __handle_transitive_cycles(self) -> bool:
        """Handle problematic transitive relationships that could create cycles.

        If we have A→B→C where A and C are in the same domain, this creates an invalid
        situation because classes in the same domain must be disjoint.
        We resolve this by removing the weaker relationship.

        Returns
        -------
        bool
            True if any changes were made, False otherwise
        """
        for relationship in self.__get_relationships():
            # Skip if source is not a domain class
            if not isinstance(relationship[0], DomainClass):
                continue

            # Get relationships from the target of this relationship
            next_relationships = self.__get_relationships_from(relationship[1])

            # Only consider relationships between domain classes
            next_relationships = [
                rel for rel in next_relationships if isinstance(rel[1], DomainClass)
            ]

            if not next_relationships:
                continue

            # Check for potential cycles where A→B→C and A and C are in the same domain
            source_domain_id = relationship[0][0]  # Domain ID of the source node

            for next_rel in next_relationships:
                target_domain_id = next_rel[1][0]  # Domain ID of the final target node

                # Only break cycles when source and final target are in the same domain
                if source_domain_id == target_domain_id:
                    # Remove the weaker relationship
                    if relationship[2] < next_rel[2]:
                        self.__remove_relationship(relationship)
                    else:
                        self.__remove_relationship(next_rel)

                    return True  # Changes were made

        return False

    def __handle_unilateral_relationships(self) -> bool:
        """Process unilateral domain-to-domain relationships into universal relationships.

        For relationships like A→B between domain classes, create appropriate universal classes
        to represent the relationship hierarchy.

        Returns
        -------
        bool
            True if any changes were made, False otherwise
        """
        for relationship in self.__get_relationships():
            # Only process domain-to-domain relationships
            if not isinstance(relationship[0], DomainClass) or not isinstance(
                relationship[1], DomainClass
            ):
                continue

            # Create a universal class containing both classes
            shared_universal_class = UniversalClass(
                frozenset({relationship[0], relationship[1]})
            )

            # Create a universal class containing only the second class
            target_universal_class = UniversalClass(frozenset({relationship[1]}))

            # Add relationships to the new universal classes
            self.__add_relationship((relationship[0], shared_universal_class, 1.0))
            self.__add_relationship((relationship[1], shared_universal_class, 1.0))
            self.__add_relationship((relationship[1], target_universal_class, 1.0))

            # Remove the original relationship
            self.__remove_relationship(relationship)

            # Redirect incoming relationships to the appropriate universal classes
            self.__redirect_incoming_relationships(
                relationship[0], shared_universal_class
            )

            # Redirect incoming relationships to the target to both universal classes
            for rel in self.__get_relationships_to(relationship[1]):
                self.__remove_relationship(rel)
                self.__add_relationship((rel[0], shared_universal_class, rel[2]))
                self.__add_relationship((rel[0], target_universal_class, rel[2]))

            return True  # Changes were made

        return False

    def save(self, filepath: str):
        """Save the taxonomy graph to a file.

        Parameters
        ----------
        filepath : str
            Path where to save the graph.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.graph, f)

    @classmethod
    def load(cls, filepath: str) -> "Taxonomy":
        """Load a taxonomy from a file.

        Parameters
        ----------
        filepath : str
            Path to the saved taxonomy.

        Returns
        -------
        Taxonomy
            The loaded taxonomy.
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

    def to_networkx(self) -> nx.DiGraph:
        """Return the NetworkX graph object.

        Returns
        -------
        nx.DiGraph
            The NetworkX graph.
        """
        return self.graph

    def visualize_graph(
        self,
        domain_labels: Dict[int, List[str]] = None,
        title: str = "Universal Taxonomy Graph",
        height: int = 800,
        width: int = 1200,
    ) -> Network:
        """Visualizes the taxonomy graph using PyVis Network.

        Creates an interactive HTML visualization of the taxonomy graph with nodes colored
        by domain and edges showing relationship strengths.

        Parameters
        ----------
        domain_labels : Dict[int, List[str]], optional
            Dictionary mapping domain IDs to lists of human-readable labels for their classes
            For example: {0: ["dog", "cat", ...], 1: ["canine", "feline", ...]}
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
        # Get the graph to visualize
        graph = self.to_networkx()

        # Step 1: Create human-readable labels for all nodes
        node_labels = self.__create_node_labels(graph, domain_labels)

        # Step 2: Set node colors and groups based on domain or universal class
        node_colors, node_groups = self.__assign_node_colors_and_groups(graph)

        # Step 3: Create and configure the PyVis network
        network = Network(height=height, width=width, directed=True)
        network.heading = title

        # Step 4: Add nodes with their styling
        self.__add_nodes_to_visualization(
            network, graph, node_labels, node_colors, node_groups
        )

        # Step 5: Add edges with weight labels
        self.__add_edges_to_visualization(network, graph)

        # Enable physics for better layout
        network.toggle_physics(True)

        return network

    def __create_node_labels(
        self,
        graph: nx.DiGraph,
        domain_labels: Dict[int, List[str]] = None,
    ) -> dict:
        """Create human-readable labels for all nodes in the graph.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph containing nodes to label
        domain_labels : Dict[int, List[str]], optional
            Dictionary mapping domain IDs to lists of human-readable labels for their classes

        Returns
        -------
        dict
            Dictionary mapping nodes to their display labels
        """
        node_labels = {}
        domain_labels = domain_labels or {}

        # First pass: Create labels for domain classes
        for node in graph.nodes():
            if isinstance(node, DomainClass):
                domain_id, class_id = node

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
        for node in graph.nodes():
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

    def __assign_node_colors_and_groups(self, graph: nx.DiGraph) -> tuple[list, list]:
        """Assign colors and groups to nodes based on their domain or type.

        Parameters
        ----------
        graph : nx.DiGraph
            The graph containing nodes to color and group

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

        # Keep track of which domains exist in the graph
        domains_seen = set()

        for node in graph.nodes():
            if isinstance(node, DomainClass):
                domain_id = node[0]
                domains_seen.add(domain_id)

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
        graph: nx.DiGraph,
        node_labels: dict,
        node_colors: list,
        node_groups: list,
    ) -> None:
        """Add nodes to the PyVis network with appropriate styling.

        Parameters
        ----------
        network : Network
            PyVis network object
        graph : nx.DiGraph
            NetworkX graph containing the nodes
        node_labels : dict
            Dictionary of node labels
        node_colors : list
            List of colors for each node
        node_groups : list
            List of group names for each node
        """
        for i, node in enumerate(graph.nodes()):
            # Add the node with appropriate styling
            network.add_node(
                str(node),  # Node ID (needs to be a string for PyVis)
                label=node_labels[node],  # Human-readable label
                color=node_colors[i],  # Color based on domain/type
                title=node_labels[node],  # Tooltip text
                group=node_groups[i],  # Group for layout algorithms
            )

    def __add_edges_to_visualization(self, network: Network, graph: nx.DiGraph) -> None:
        """Add edges to the PyVis network with weight labels.

        Parameters
        ----------
        network : Network
            PyVis network object
        graph : nx.DiGraph
            NetworkX graph containing the edges with weights
        """
        for source, target, data in graph.edges(data=True):
            weight = data.get("weight", 1.0)
            # Format weight to 2 decimal places for display
            weight_label = f"{weight:.2f}"

            # Add the edge with the weight as both the label and tooltip
            network.add_edge(
                str(source),  # Source node ID
                str(target),  # Target node ID
                title=weight_label,  # Tooltip showing weight
                label=weight_label,  # Edge label showing weight
                value=weight,  # Numeric weight (affects edge thickness)
            )
