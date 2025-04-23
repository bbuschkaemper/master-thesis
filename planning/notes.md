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
