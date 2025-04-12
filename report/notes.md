# Notes

## Training models

- Use SGD for smaller caltech dataset and Adam for larger cifar dataset
- More complicated ResNet architecture for larger cifar dataset
- Decrease torch matmul accuracy for more speed
- Weight decay for cifar since model overfits
- Use paperswithcode leaderboard to quote and compare baseline models per dataset
- Cite paper for selected learning rate changes: <https://arxiv.org/abs/1708.04552v2>
- Also explain dropout

## Improvements to original

- Use probability instead of improved naive concatenation
- Use unilateral relationship that targets universal class instead of discarding it (in paper: palm -> palm but not palm -> {A-tree, V-vegetation})
- Use probability threshold to discard unsure relationships and instead put them into a universal class directly
