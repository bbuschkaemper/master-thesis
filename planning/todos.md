# Todos

## Thesis writing

Empty

## Graph correctness metrics

- We treat each universal taxonomy class as an edge between two classes of different domains that have an incoming edge to the universal class
- Every combination of two classes from different domains is one edge
- Average precision:
  - <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html>
  - See for reference implementation
- Graph edit distance:
  - Create adjacency matrix from the edges
  - Use XOR to find number of incorrect edges and use that for metric
- Edges have weights that are not yet respected in the metrics
- The Missing Link paper for reference implementation that uses edge weights

## Conceptual taxonomy

- Instead of mcfp, we use all predictions above a certain threshold (using a single mcfp means we cannot represent multiple concept relationships)
- Only create universal class on bidirectional edges
- Unilateral edges do not have a true meaning here (if it were a subset of the other, it would have a bidirectional edge to some degree)
- Find some method to calculate the threshold (it should depend on how likely it is to get a random class prediction)

## Thresholding

- Most common foreign prediction does not care about probability (e.g. mcfp could be 0.51 vs 0.49 or 0.9 vs 0.1 and have same result)
- We already use normalised probabilities as weights for edges
- Introduce threshold for edge weights:
  - Discard edges with weights below threshold
  - Use domain class as its own universal class
- Use deviation to ground truth taxonomy as metric to find optimal threshold
- We have no-threshold as baseline for comparison

## Universal taxonomy model training

- We train a single new model with universal taxonomy as output (num universal classes equals output layer size)
- We train on domain datasets:
  - Mapping of training data target class to universal classes with edge weights as target output distribution
  - Loss function is cross entropy between predicted and target output distribution
- Compare accuracy of universal taxonomy model on domain datasets with baseline models
- Question: How are taxonomy correctness and model accuracy on domain datasets related?
