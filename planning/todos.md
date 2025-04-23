# Todos

## Synthetic ground truth universal taxonomy

- Pick dataset as starting point
- Build synthetic datasets with deviations:
  - Discard some classes from original dataset
  - Merge multiple classes from original dataset into a single new dataset class
- This creates a ground truth taxonomy with multiple domains (i.e. deviating datasets)

## Multi-domain algorithm

- Right now algorithm only works for two domains
- Use iterative approach to build universal taxonomy for n domains:
  - Use same algorithm rules for editing edges but apply them for n domains
- Compare against ground truth taxonomy:
  - Find metric for deviation of universal taxonomy from ground truth taxonomy

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
