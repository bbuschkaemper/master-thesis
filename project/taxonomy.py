import numpy as np
import numpy.typing as npt


class DomainClass(tuple[np.intp, np.intp]):
    """Class a from domain A represented as a tuple (A, a)."""


class UniversalClass(frozenset[DomainClass]):
    """A class a from the universal class domain represented as a set of domain classes."""


type Class = DomainClass | UniversalClass
"""A class a from domain A represented as a tuple (A, a) or a class from the universal domain
    represented as a set of domain classes."""


class Relationship(tuple[Class, Class, np.float32]):
    """A unilateral relationship A:a -> B:b represented as a tuple of two classes and a weight.
    The weight is the probability of the relationship."""


class DirectedWeightedUniversalTaxonomyGraph:
    def __init__(self):
        self.relationships: set[Relationship] = set()
        self.nodes: set[Class] = set()

    def add_relationship(self, relationship: Relationship):
        """Adds a relationship to the graph.

        Parameters
        ----------
        relationship : Relationship
            The relationship to add to the graph.
        """
        self.relationships.add(relationship)

        # Add the nodes to the graph
        self.nodes.add(relationship[0])
        self.nodes.add(relationship[1])

    def remove_relationship(self, relationship: Relationship):
        """Removes a relationship from the graph.

        Parameters
        ----------
        relationship : Relationship
            The relationship to remove from the graph.
        """
        self.relationships.remove(relationship)

    def get_relationships(self) -> list[Relationship]:
        """Returns the relationships of the graph.

        Returns
        -------
        list[Relationship]
            The relationships of the graph.
        """
        return self.relationships

    def get_nodes(self) -> set[Class]:
        """Returns the nodes of the graph.

        Returns
        -------
        set[Class]
            The nodes of the graph.
        """
        return self.nodes

    def get_domain_relationship_from(self, node: Class) -> Relationship | None:
        """Returns the relationship from a node towards a domain class
        or None if the node has no outgoing relationship to a domain class.

        Parameters
        ----------
        node : Class
            The node to get the relationships from.

        Returns
        -------
        Relationship | None
            The relationship from the node.
            None if the node has no outgoing relationship.
        """
        relationships = [
            relationship
            for relationship in self.relationships
            if relationship[0] == node and isinstance(relationship[1], DomainClass)
        ]

        if len(relationships) == 0:
            return None

        if len(relationships) > 1:
            raise ValueError(
                f"Node {node} has more than one outgoing relationship: {relationships}"
            )
        return relationships[0]

    def get_relationships_from(self, node: Class) -> list[Relationship]:
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
        return [
            relationship
            for relationship in self.relationships
            if relationship[0] == node
        ]

    def get_relationships_to(self, node: Class) -> list[Relationship]:
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
        return [
            relationship
            for relationship in self.relationships
            if relationship[1] == node
        ]

    def get_relationship(self, from_node: Class, to_node: Class) -> Relationship | None:
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
        for relationship in self.relationships:
            if relationship[0] == from_node and relationship[1] == to_node:
                return relationship
        return None

    def is_finished(self) -> bool:
        """Checks if the graph is finished.
        A graph is finished if all mappings are from domain classes to universal classes
        and all universal classes have at least one incoming relationship."""

        # Check that all universal classes have at least one incoming relationship
        for node in self.get_nodes():
            if isinstance(node, UniversalClass):
                if len(self.get_relationships_to(node)) == 0:
                    return False

        # Check that all relationships are from domain classes to universal classes
        for relationship in self.relationships:
            if not (
                isinstance(relationship[0], DomainClass)
                and isinstance(relationship[1], UniversalClass)
            ):
                return False

        return True


class Taxonomy:
    def __init__(
        self,
        a_to_b_predictions: npt.NDArray[np.intp],
        b_to_a_predictions: npt.NDArray[np.intp],
        a_targets: npt.NDArray[np.intp],
        b_targets: npt.NDArray[np.intp],
    ):
        """Creates a taxonomy object.
        The taxonomy takes the predictions of a model A trained for domain A
        that predicts on a foreign domain B and the predictions of a model B trained for
        domain B that predicts on a foreign domain A. The targets are the true labels of
        the respective domains.
        The predictions use labels of their own domain on the foreign domain.

        Parameters
        ----------
        a_to_b_predictions : npt.NDArray[np.intp]
            Predictions of model A (using domain A labels) on domain B
        b_to_a_predictions : npt.NDArray[np.intp]
            Predictions of model B (using domain B labels) on domain A
        a_targets : npt.NDArray[np.intp]
            True labels for domain A
        b_targets : npt.NDArray[np.intp]
            True labels for domain B
        """

        assert a_to_b_predictions.shape == b_targets.shape
        assert b_to_a_predictions.shape == a_targets.shape

        self.a_to_b_predictions = a_to_b_predictions
        self.b_to_a_predictions = b_to_a_predictions
        self.a_targets = a_targets
        self.b_targets = b_targets

    @staticmethod
    def form_correlation_matrix(
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
    def most_common_foreign_predictions(
        correlations: npt.NDArray[np.intp],
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]:
        """Calculates the most common foreign predictions for each class in the foreign
        domain. The most common foreign prediction is the class in the own domain that
        was predicted the most times for a class in the foreign domain.

        The result is a 1D array of shape (n_classes_foreign,) where the value at index i
        is the own domain class that was predicted the most times for the foreign domain class i
        together with the probability of the prediction.
        The probability is the number of times the class was predicted divided by the total
        number of predictions for that class.
        If there are no predictions for a class, the probability is 0.

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

    def build_universal_taxonomy(self) -> DirectedWeightedUniversalTaxonomyGraph:
        """Builds a universal taxonomy graph from the predictions and targets.

        Returns
        -------
        DirectedTaxonomyGraph
            The universal taxonomy graph.
        """

        # Build the correlation matrices
        a_to_b_correlations = self.form_correlation_matrix(
            self.a_to_b_predictions, self.b_targets
        )
        b_to_a_correlations = self.form_correlation_matrix(
            self.b_to_a_predictions, self.a_targets
        )

        # Get the most common foreign predictions
        a_to_b_foreign_predictions = self.most_common_foreign_predictions(
            a_to_b_correlations
        )
        b_to_a_foreign_predictions = self.most_common_foreign_predictions(
            b_to_a_correlations
        )

        # Build the taxonomy graph
        # A relationship goes from ground truth foreign class to most common own class
        graph = DirectedWeightedUniversalTaxonomyGraph()
        for i, pred in enumerate(a_to_b_foreign_predictions[0]):
            from_class = DomainClass((1, i))
            to_class = DomainClass((0, pred))
            graph.add_relationship(
                (from_class, to_class, a_to_b_foreign_predictions[1][i])
            )
        for i, pred in enumerate(b_to_a_foreign_predictions[0]):
            from_class = DomainClass((0, i))
            to_class = DomainClass((1, pred))
            graph.add_relationship(
                (from_class, to_class, b_to_a_foreign_predictions[1][i])
            )

        while not graph.is_finished():
            added = False

            # If we have a node A:a without a relationship,
            # we create a new relationship A:a -> {A:a} to a new universal class.
            for node in graph.get_nodes():
                if (
                    len(graph.get_relationships_to(node)) == 0
                    and len(graph.get_relationships_from(node)) == 0
                ):
                    # Create a new universal class
                    universal_class = UniversalClass(frozenset({node}))
                    graph.add_relationship((node, universal_class, 1.0))
                    added = True
                    break
            if added:
                continue

            # If two classes A:a, B:a have bi-directional mappings A:a -> B:a, B:a -> A:a,
            # it suggests that they are the same class.
            # We then map: A:a -> {A:a, B:a}, B:a -> {A:a, B:a}
            # and remove the old bi-directional mapping.
            # We redirect incoming relationships of old classes to the new universal class.
            # E.g. A:b -> A:a => A:b -> {A:a, B:a}
            for relationship in graph.get_relationships():
                rel = graph.get_relationship(relationship[1], relationship[0])
                if rel is None:
                    continue

                # Create a universal class.
                # If the relationships contain a universal class,
                # add their nodes to the new universal class.
                universal_class = set()
                if isinstance(relationship[0], UniversalClass):
                    universal_class.update(relationship[0])
                else:
                    universal_class.add(relationship[0])
                if isinstance(relationship[1], UniversalClass):
                    universal_class.update(relationship[1])
                else:
                    universal_class.add(relationship[1])
                universal_class = UniversalClass(frozenset(universal_class))

                # Add the new universal class to the graph that contains both classes.
                graph.add_relationship((relationship[0], universal_class, 1.0))
                graph.add_relationship((relationship[1], universal_class, 1.0))

                # Remove the bi-directional relationships.
                graph.remove_relationship(relationship)
                graph.remove_relationship(rel)

                # Redirect incoming relationships of old classes to the new universal class.
                for rel in graph.get_relationships_to(relationship[0]):
                    graph.remove_relationship(rel)
                    graph.add_relationship((rel[0], universal_class, rel[2]))
                for rel in graph.get_relationships_to(relationship[1]):
                    graph.remove_relationship(rel)
                    graph.add_relationship((rel[0], universal_class, rel[2]))
                added = True
                break
            if added:
                continue

            # If we have a triplet of relationships
            # A:a -> B:a, B:a -> A:b, it would suggest that A:a is a subclass of B:a is a subclass of A:b,
            # which would mean that A:a is a subclass of A:b.
            # This is impossible since A:a and A:b are in the same domain and automatically disjoint.
            # Therefore, we remove the relationship with the lower weight.
            for relationship in graph.get_relationships():
                relationships_from = [
                    relationship
                    for relationship in graph.get_relationships_from(relationship[1])
                    if isinstance(relationship[1], DomainClass)
                ]
                if len(relationships_from) < 1:
                    continue

                # Remove the relationship with the lower weight
                if relationship[2] < relationships_from[0][2]:
                    graph.remove_relationship(relationship)
                else:
                    graph.remove_relationship(relationships_from[0])
                added = True
                break
            if added:
                continue

            # If we have a unilateral relationship A:a -> B:a between domain classes,
            # it suggests that A:a is a subclass of B:a.
            # We then map: A:a -> {A:a, B:a}, B:a -> {A:a, B:a}, B:a -> {B:a}
            # and remove the old relationship.
            # We redirect incoming relationships as well:
            # E.g. A:b -> A:a => A:b -> {A:a, B:a}
            # E.g. A:c -> B:a => A:c -> {A:a, B:a}, A:c -> {B:a}
            for relationship in graph.get_relationships():
                if isinstance(relationship[0], UniversalClass) or isinstance(
                    relationship[1], UniversalClass
                ):
                    continue

                # Add the new universal class to the graph that contains both classes.
                universal_class = set()
                if isinstance(relationship[0], UniversalClass):
                    universal_class.update(relationship[0])
                else:
                    universal_class.add(relationship[0])
                if isinstance(relationship[1], UniversalClass):
                    universal_class.update(relationship[1])
                else:
                    universal_class.add(relationship[1])
                universal_class = UniversalClass(frozenset(universal_class))
                graph.add_relationship((relationship[0], universal_class, 1.0))
                graph.add_relationship((relationship[1], universal_class, 1.0))

                # Add the new universal class to the graph that contains the second class.
                universal_class2 = set()
                if isinstance(relationship[1], UniversalClass):
                    universal_class2.update(relationship[1])
                else:
                    universal_class2.add(relationship[1])
                universal_class2 = UniversalClass(frozenset(universal_class2))
                graph.add_relationship((relationship[1], universal_class2, 1.0))
                graph.add_relationship((relationship[1], universal_class2, 1.0))

                # Remove the old relationship.
                graph.remove_relationship(relationship)

                # Redirect incoming relationships to A:a to the first new universal class.
                for rel in graph.get_relationships_to(relationship[0]):
                    graph.remove_relationship(rel)
                    graph.add_relationship((rel[0], universal_class, rel[2]))

                # Redirect incoming relationships to B:a to both new universal classes.
                for rel in graph.get_relationships_to(relationship[1]):
                    graph.remove_relationship(rel)
                    graph.add_relationship((rel[0], universal_class, rel[2]))
                    graph.add_relationship((rel[0], universal_class2, rel[2]))
                added = True
                break
            if added:
                continue

        return graph
