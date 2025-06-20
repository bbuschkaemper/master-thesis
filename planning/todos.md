# Todos

## Thesis

- Hypothesis for the thesis: "Can we create a universal model based on universal taxonomy that performs similar to baseline models trained on domain datasets?"

## Taxonomy generation

- Test taxonomy generation on cifar100-caltech101 to see if it works and write a lot about it in the results section (we can have a lot of examples/visualizations here)

## Compare filtering methods

- Use 2 different datasets for metrics (maybe ImageNet as second?)
- Multiple synthetic dataset deviations with different complexities
- Find two digit datasets to have a known mapping between them (e.g. one with color and one with black and white)
- Write about filtering methods in method section and compare them in results section
- Find optimal parameters for the filtering methods by comparing their metrics on the same datasets (do we need to do this for each dataset or is tuning on one dataset enough?)
- Precision/Recall curves with parameters for filtering methods (run on multiple datasets)

## Conceptual universal taxonomy

- Create universal class only on bidirectional edges or for isolated nodes
- Unilateral edges have no meaning when using thresholding that creates multiple outgoing edges per class (subset hypothesis would not hold)
- Add to method section as second algorithm
- Performance can only be compared on final trained model, since taxonomy algorithms are also applied to ground truth (write this in method section!)

## Universal taxonomy model training

- New model with universal classes as output layer
- We train on domain datasets:
  - Map prediction from universal classes to domain classes by using relationship probabilities
  - Loss function is cross-entropy between predicted mapping and ground truth domain classes
- Compare accuracy of universal taxonomy model on domain datasets with baseline models trained only on the domain datasets
- Question: How are taxonomy correctness and model accuracy on domain datasets related?
- Question: How does the model accuracy on domain datasets change with different universal taxonomy algorithms?
