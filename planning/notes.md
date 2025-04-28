# Notes for writing thesis

## Training fine-tuned models

- Used SGD for smaller caltech dataset and Adam for larger cifar dataset
- More complicated ResNet architecture for larger cifar dataset
- Decrease torch matmul accuracy for more speed
- Weight decay for cifar since model overfits
- Use paperswithcode leaderboard to quote and compare baseline models per dataset
- Cite cutout paper for learning rate changes selection
- Also explain why we used dropout

## Improvements to original universal taxonomy method

- Method is based on both papers
- We used unilateral relationship that targets universal class instead of discarding it (in paper example: palm -> palm but not palm -> {A-tree, V-vegetation})
- We used probabilities instead of most common foreign prediction (which means we didn't need improved naive concatenation)
- We used individual models for each domain instead of a shared model with individual heads

## Multi-domain taxonomy

- Rules can be applied exactly the same way as in the original paper
- We need to look for transitive relationships if A and C are the same domain

## Synthetic taxonomy

- We simulate the models running on the dataset deviations
- Some edge cases have equal probabilities between merged classes which means that the argmax simply uses the first class (it is not realistic for real-world datasets so we can ignore it)
- We use truncated normal distribution to determine clusters and deviation dataset sizes

### Synthetic Taxonomy Algorithm Details

- **DeviationClass**: Implemented as a frozenset of class IDs representing classes that are merged/clustered in a deviation
- **Deviation**: Collection of DeviationClasses that form a complete synthetic dataset/domain
- **Deviation Generation Algorithm**:
  1. Use truncated normal distribution to determine number of classes per deviation
  2. Randomly select classes from original dataset to be part of this deviation
  3. Form clusters by:
     - Determining cluster size using truncated normal distribution
     - Randomly assigning classes to clusters until all classes are assigned
  
- **Prediction Simulation Algorithm**:
  1. For each pair of deviations (excluding self-predictions):
  - The original classes we redesign as concepts that are then shared between the deviations
  - Compare each target deviation class with each dataset deviation class
  - Calculate overlap between classes in the clusters
  - Assign prediction probabilities based on overlap ratio
  - Distribute remaining percentages after overlap evenly among all foreign classes
