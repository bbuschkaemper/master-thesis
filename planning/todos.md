# Todos

## Thesis writing

Empty

## Code refactoring

- Write save/load for synthetic taxonomy
- Instead of modifying dataloader, create dataset class that directly has the correct targets
- Use a single taxonomy class that defines different constructors for each method of taxonomy/synthetic taxonomy

## Taxonomy model simulation

- Can we simulate a model with prediction probability to get a plot (model accuracy vs. taxonomy accuracy) instead of manually training models?

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
