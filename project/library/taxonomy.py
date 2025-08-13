import pickle
from typing import List, Dict, Tuple
import numpy as np
import numpy.typing as npt
import networkx as nx
import plotly.graph_objects as go
from pyvis.network import Network
from .types import (
    DomainClass,
    UniversalClass,
    Class,
    Relationship,
)


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

    def __init__(self, domain_labels: Dict[int, List[str]] | None = None):
        """Initializes an empty taxonomy with no relationships or classes.

        Parameters
        ----------
        domain_labels : Dict[int, List[str]], optional
            Optional dictionary mapping domain IDs to lists of human-readable class labels.
            If provided, these labels will be used for domain classes instead of class IDs.
            If not provided, class IDs will be used as labels.
        """

        self.domain_labels = domain_labels or {}
        self.graph = nx.DiGraph()

    def _add_node(self, node: Class):
        """Adds a node to the graph.

        Parameters
        ----------
        node : Class
            The node to add to the graph (DomainClass or UniversalClass)
        """
        if not self.graph.has_node(node):
            self.graph.add_node(node, node_obj=node)

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

    def _remove_relationship(self, relationship: Relationship):
        """Removes a relationship from the graph.

        Parameters
        ----------
        relationship : Relationship
            The relationship to remove from the graph (target, source, weight)
        """
        target, source, _ = relationship
        if self.graph.has_edge(target, source):
            self.graph.remove_edge(target, source)

    def _get_relationships(self) -> list[Relationship]:
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

    def get_nodes(self) -> set[Class]:
        """Returns all nodes in the graph.

        Returns
        -------
        set[Class]
            All nodes in the graph
        """
        return set(self.graph.nodes())

    def get_relationships_from(self, target: Class) -> list[Relationship]:
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

    def _get_relationships_to(self, source: Class) -> list[Relationship]:
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

    def _get_relationship(self, target: Class, source: Class) -> Relationship | None:
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
        for node in self.get_nodes():
            # Only process domain classes without any connections
            if (
                isinstance(node, DomainClass)
                and not self._get_relationships_to(node)
                and not self.get_relationships_from(node)
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
        for relationship in self._get_relationships():
            class_a, class_b, _ = relationship

            # Check if there's a reverse relationship
            reverse_relationship = self._get_relationship(class_b, class_a)
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
            self._remove_relationship(relationship)
            self._remove_relationship(reverse_relationship)

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
        for relationship in self._get_relationships():
            class_a, class_b, first_relationship_weight = relationship

            # Skip if the first class is not a domain class
            if not isinstance(class_a, DomainClass):
                continue

            # Get relationships originating from class_B
            next_relationships = self.get_relationships_from(class_b)

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
                        self._remove_relationship(relationship)
                    else:
                        self._remove_relationship(next_relationship)

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
        for relationship in self._get_relationships():
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
            self._remove_relationship(relationship)

            return True  # Changes were made

        return False

    def save(self, filepath: str):
        """Save the taxonomy graph to a file (pickle format).

        Parameters
        ----------
        filepath : str
            Path where to save the graph
        """
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "graph": self.graph,
                    "domain_labels": self.domain_labels,
                },
                f,
            )

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
        taxonomy = cls()

        # Load the graph directly
        with open(filepath, "rb") as f:
            obj = pickle.load(f)

        taxonomy.graph = obj["graph"]
        taxonomy.domain_labels = obj["domain_labels"]

        return taxonomy

    def visualize_graph(
        self,
        title: str = "Universal Taxonomy Graph",
        height: int = 900,
        width: int = 1800,
    ) -> Network:
        """Visualizes the taxonomy graph using PyVis Network.

        Creates an interactive HTML visualization of the taxonomy graph with nodes colored
        by domain and edges showing relationship strengths.

        Parameters
        ----------
        title : str, optional
            Title to display on the visualization, by default "Universal Taxonomy Graph"
        height : int, optional
            Height of the visualization in pixels, by default 900
        width : int, optional
            Width of the visualization in pixels, by default 1800

        Returns
        -------
        Network
            PyVis Network object that can be displayed or saved to HTML
        """

        # Create human-readable labels for all nodes
        node_labels = self.__create_node_labels()

        # Set node colors and groups based on domain or universal class
        node_colors, node_groups = self.__assign_node_colors_and_groups()

        # Create and configure the PyVis network
        network = Network(height=height, width=width, directed=True)  # type: ignore
        network.heading = title

        # Add nodes with their styling
        self.__add_nodes_to_visualization(
            network, node_labels, node_colors, node_groups
        )

        # Add edges with weight labels
        self.__add_edges_to_visualization(network)

        # Enable physics for better layout
        network.toggle_physics(True)

        return network

    def visualize_3d_graph(
        self,
        show_labels: bool = True,
    ) -> go.Figure:
        """Visualizes the taxonomy graph in 3D using Plotly.

        Creates a 3D scatter plot of the taxonomy graph with nodes colored by domain
        and edges showing relationships.

        Parameters
        ----------
        show_labels : bool, optional
            Whether to show human-readable labels for nodes, by default True

        Returns
        -------
        go.Figure
            Plotly Figure object containing the 3D visualization
        """

        # Create human-readable labels for all nodes
        node_labels = self.__create_node_labels()

        # Assign colors and groups to nodes
        node_colors, _ = self.__assign_node_colors_and_groups()

        # Encode positions of nodes in 3D space
        pos = nx.spring_layout(self.graph, dim=3, seed=42)

        # Extract node positions
        x = [pos[node][0] for node in self.graph.nodes()]
        y = [pos[node][1] for node in self.graph.nodes()]
        z = [pos[node][2] for node in self.graph.nodes()]

        # Create a scatter plot for nodes
        if show_labels:
            node_trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers+text",
                marker=dict(
                    size=10,
                    color=node_colors,
                    line=dict(width=2),
                ),
                text=[node_labels[node] for node in self.graph.nodes()],
                hoverinfo="text",
            )
        else:
            node_trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=10,
                    color=node_colors,
                    line=dict(width=2),
                ),
                hoverinfo="none",
            )

        # Create a list of edges for the 3D plot
        edge_x = []
        edge_y = []
        edge_z = []
        for source, target in self.graph.edges():
            x0, y0, z0 = pos[source]
            x1, y1, z1 = pos[target]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]

        # Create a scatter plot for edges
        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(width=1, color="black"),
        )

        # Create the final figure with nodes and edges
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                zaxis=dict(showgrid=False, zeroline=False, visible=False),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            width=1000,
            height=1000,
        )
        fig.update_traces(marker=dict(sizemode="diameter", opacity=0.8))

        return fig

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

    def _adjacency_matrix(self) -> npt.NDArray[np.float64]:
        """Generate the adjacency matrix of the taxonomy graph.

        The adjacency matrix is a square matrix where each element (i, j)
        represents the weight of the relationship between domain class i and
        domain class j. Assumes that no universal classes are present.

        Returns
        -------
        npt.NDArray[np.float64]
            The adjacency matrix representing relationships between domain classes
        """

        assert all(
            not isinstance(node, UniversalClass) for node in self.get_nodes()
        ), "Adjacency matrix is only defined for taxonomies without universal classes."

        # Get all domain classes in the graph
        domain_classes = [
            node for node in self.get_nodes() if isinstance(node, DomainClass)
        ]
        domain_classes = sorted(domain_classes, key=lambda x: (x[0], x[1]))
        num_classes = len(domain_classes)

        # Build the adjacency matrix
        adjacency_matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
        for relationship in self._get_relationships():
            target, source, weight = relationship
            if isinstance(target, DomainClass) and isinstance(source, DomainClass):
                target_index = domain_classes.index(target)
                source_index = domain_classes.index(source)
                adjacency_matrix[target_index, source_index] = weight

        # If a node has no incoming or outgoing relationships,
        # add a self-loop with weight 1.0
        for i, domain_class in enumerate(domain_classes):
            if not self._get_relationships_to(
                domain_class
            ) and not self.get_relationships_from(domain_class):
                adjacency_matrix[i, i] = 1.0

        return adjacency_matrix

    def edge_difference_ratio(self, other: "Taxonomy") -> np.float64:
        """Calculate the edge difference ratio between this taxonomy and another.
        Assumes that both taxonomies have no universal classes.

        Parameters
        ----------
        other : Taxonomy
            The other taxonomy to compare against

        Returns
        -------
        np.float64
            The edge difference ratio between the two taxonomies
        """

        adj1 = self._adjacency_matrix()
        adj2 = other._adjacency_matrix()  # pylint: disable=protected-access

        # Validate that matrices have the same dimensions
        if adj1.shape != adj2.shape:
            raise ValueError(
                f"Taxonomies have incompatible domain class counts: "
                f"{adj1.shape} vs {adj2.shape}. "
                f"Both taxonomies must have the same domain classes for comparison."
            )

        # Calculate element-wise differences between the two matrices
        diff = adj1 - adj2

        # Calculate element-wise maximum for normalization
        max_val = np.maximum(adj1, adj2)

        # Both matrices are completely zero => perfect match
        if not np.any(max_val):
            return np.float64(0.0)

        return np.sum(np.abs(diff)) / np.sum(max_val)

    def precision_recall_f1(
        self, other: "Taxonomy"
    ) -> Tuple[np.float64, np.float64, np.float64]:
        """Calculates the precision, recall, and F1 score between this taxonomy and another.
        Assumes that both taxonomies have no universal classes.

        Parameters
        ----------
        other : Taxonomy
            The other taxonomy to compare against

        Returns
        -------
        Tuple[np.float64, np.float64, np.float64]
            The precision, recall, and F1 score between the two taxonomies
        """

        adj1 = self._adjacency_matrix()
        adj2 = other._adjacency_matrix()  # pylint: disable=protected-access

        # Validate that matrices have the same dimensions
        if adj1.shape != adj2.shape:
            raise ValueError(
                f"Taxonomies have incompatible domain class counts: "
                f"{adj1.shape} vs {adj2.shape}. "
                f"Both taxonomies must have the same domain classes for comparison."
            )

        # Binarize the adjacency matrices
        adj1 = (adj1 > 0).astype(np.float64)
        adj2 = (adj2 > 0).astype(np.float64)

        # Calculate true positives, false positives, and false negatives
        true_positives = np.sum(np.logical_and(adj1, adj2))
        false_positives = np.sum(np.logical_and(adj1, np.logical_not(adj2)))
        false_negatives = np.sum(np.logical_and(np.logical_not(adj1), adj2))

        # Calculate precision, recall, and F1 score
        precision = np.float64(
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = np.float64(
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1_score = np.float64(
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return precision, recall, f1_score
