import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.manifold import TSNE
from typing import List, Tuple, Optional, Union
from torch.utils.data import DataLoader
import pandas as pd


class UniversalModelAnalyzer:
    """
    Analyzer class for extracting and visualizing representations from Universal ResNet models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the analyzer with a trained universal model.

        Parameters
        ----------
        model : torch.nn.Module
            The trained universal ResNet model
        device : str
            Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Hook to capture features from the layer before taxonomy mapping
        self.features = []
        self.hook_handle = None

    def _feature_hook(self, module, input_tensor, output):
        """Hook function to capture intermediate features."""
        # Ignore unused parameters - they're required by PyTorch's hook signature
        del module, input_tensor  # Suppress unused variable warnings
        self.features.append(output.detach().cpu().numpy())

    def _register_hook(self):
        """Register hook to capture features from the penultimate layer."""
        # For ResNet architectures, we want the features before the final classification layer
        if hasattr(self.model, "model") and hasattr(self.model.model, "fc"):
            # For the universal ResNet, we want features before the fc layer
            if isinstance(self.model.model.fc, torch.nn.Sequential):
                # Hook to the first layer of the sequential (before taxonomy mapping)
                self.hook_handle = self.model.model.fc[0].register_forward_hook(
                    self._feature_hook
                )
            else:
                # Hook to the layer before fc
                self.hook_handle = self.model.model.avgpool.register_forward_hook(
                    self._feature_hook
                )
        elif hasattr(self.model, "model") and hasattr(self.model.model, "classifier"):
            # For EfficientNet architectures
            if isinstance(self.model.model.classifier, torch.nn.Sequential):
                # Hook to the first layer after dropout
                self.hook_handle = self.model.model.classifier[1].register_forward_hook(
                    self._feature_hook
                )
            else:
                self.hook_handle = self.model.model.features.register_forward_hook(
                    self._feature_hook
                )
        else:
            raise ValueError(
                "Unable to identify the appropriate layer for feature extraction"
            )

    def _unregister_hook(self):
        """Remove the registered hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def extract_universal_features(
        self, dataloader: DataLoader, max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], List[str]]:
        """
        Extract features from the universal output layer (before taxonomy mapping).

        This method specifically extracts features from the final layer of the model
        that outputs universal class activations, before they are converted to domain-specific
        predictions via the taxonomy mapping.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader containing the data to analyze
        max_samples : Optional[int]
            Maximum number of samples to process (None for all samples)

        Returns
        -------
        Tuple[np.ndarray, List[Tuple[int, int]], List[str]]
            - Universal features array of shape (n_samples, n_universal_classes)
            - List of (domain_id, class_id) tuples for each sample
            - List of class names for each sample
        """
        universal_features = []
        labels = []
        class_names = []

        with torch.no_grad():
            sample_count = 0

            for inputs, targets in dataloader:
                if max_samples is not None and sample_count >= max_samples:
                    break

                inputs = inputs.to(self.device)

                # Get universal class outputs directly from the model
                universal_outputs = self.model(inputs)
                universal_features.append(universal_outputs.cpu().numpy())

                # Process targets - handle different formats
                if isinstance(targets, list) and len(targets) == 2:
                    # Targets are [domain_ids_tensor, class_ids_tensor] from CombinedDataModule
                    domain_ids_tensor, class_ids_tensor = targets
                    batch_labels = list(
                        zip(domain_ids_tensor.tolist(), class_ids_tensor.tolist())
                    )
                elif isinstance(targets[0], tuple):
                    # Targets are already (domain_id, class_id) tuples
                    batch_labels = targets
                else:
                    # Convert other formats if needed
                    batch_labels = [(int(t[0]), int(t[1])) for t in targets]

                labels.extend(batch_labels)

                # Generate class names
                for domain_id, class_id in batch_labels:
                    class_names.append(f"Domain_{domain_id}_Class_{class_id}")

                sample_count += len(batch_labels)

                if max_samples is not None and sample_count >= max_samples:
                    # Trim to exact number of samples
                    excess = sample_count - max_samples
                    if excess > 0:
                        labels = labels[:-excess]
                        class_names = class_names[:-excess]
                        universal_features[-1] = universal_features[-1][:-excess]
                    break

        # Concatenate all features
        if not universal_features:
            raise ValueError(
                "No features were extracted. Check your dataloader and model."
            )

        all_features = np.concatenate(universal_features, axis=0)

        return all_features, labels, class_names

    def analyze_universal_output_tsne(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = 1000,
        perplexity: float = 30.0,
        color_by: str = "domain",
        save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Figure]:
        """
        Analyze the universal output layer with t-SNE (before taxonomy mapping).

        This is the main method for analyzing the output layer of the universal ResNet model
        that contains the universal class activations before they are mapped to domain-specific
        predictions.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader containing the data to analyze
        max_samples : Optional[int]
            Maximum number of samples to process
        perplexity : float
            t-SNE perplexity parameter
        color_by : str
            What to color by in visualization
        save_path : Optional[str]
            Path to save the visualization (optional)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Figure]
            - Universal features from output layer
            - t-SNE embeddings
            - Matplotlib figure
        """
        # Extract universal features directly from output layer
        features, labels, class_names = self.extract_universal_features(
            dataloader, max_samples
        )

        # Apply t-SNE
        embeddings = self.apply_tsne(features, perplexity=perplexity)

        # Visualize
        fig = self.visualize_tsne(embeddings, labels, class_names, color_by=color_by)
        fig.suptitle(
            "t-SNE Analysis of Universal Output Layer (Before Taxonomy Mapping)",
            fontsize=14,
            y=0.98,
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Universal output layer visualization saved to {save_path}")

        return features, embeddings, fig

    def extract_features(
        self, dataloader: DataLoader, max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], List[str]]:
        """
        Extract features from the penultimate layer of the model.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader containing the data to analyze
        max_samples : Optional[int]
            Maximum number of samples to process (None for all samples)

        Returns
        -------
        Tuple[np.ndarray, List[Tuple[int, int]], List[str]]
            - Features array of shape (n_samples, n_features)
            - List of (domain_id, class_id) tuples for each sample
            - List of class names for each sample
        """
        self.features = []
        labels = []
        class_names = []

        self._register_hook()

        try:
            with torch.no_grad():
                sample_count = 0

                for inputs, targets in dataloader:
                    if max_samples is not None and sample_count >= max_samples:
                        break

                    inputs = inputs.to(self.device)

                    # Forward pass to trigger the hook
                    _ = self.model(inputs)

                    # Process targets - handle different formats
                    if isinstance(targets, list) and len(targets) == 2:
                        # Targets are [domain_ids_tensor, class_ids_tensor] from CombinedDataModule
                        domain_ids_tensor, class_ids_tensor = targets
                        batch_labels = list(
                            zip(domain_ids_tensor.tolist(), class_ids_tensor.tolist())
                        )
                    elif isinstance(targets[0], tuple):
                        # Targets are already (domain_id, class_id) tuples
                        batch_labels = targets
                    else:
                        # Convert other formats if needed
                        batch_labels = [(int(t[0]), int(t[1])) for t in targets]

                    labels.extend(batch_labels)

                    # Generate class names (you might want to customize this based on your taxonomy)
                    for domain_id, class_id in batch_labels:
                        class_names.append(f"Domain_{domain_id}_Class_{class_id}")

                    sample_count += len(batch_labels)

                    if max_samples is not None and sample_count >= max_samples:
                        # Trim to exact number of samples
                        excess = sample_count - max_samples
                        if excess > 0:
                            labels = labels[:-excess]
                            class_names = class_names[:-excess]
                            self.features[-1] = self.features[-1][:-excess]
                        break

        finally:
            self._unregister_hook()

        # Concatenate all features
        if not self.features:
            raise ValueError(
                "No features were extracted. Check your dataloader and model."
            )

        all_features = np.concatenate(self.features, axis=0)

        # Flatten features if they have spatial dimensions
        if len(all_features.shape) > 2:
            all_features = all_features.reshape(all_features.shape[0], -1)

        return all_features, labels, class_names

    def apply_tsne(
        self,
        features: np.ndarray,
        perplexity: float = 30.0,
        n_components: int = 2,
        learning_rate: Union[str, float] = "auto",
        n_iter: int = 1000,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Apply t-SNE dimensionality reduction to features.

        Parameters
        ----------
        features : np.ndarray
            Feature array of shape (n_samples, n_features)
        perplexity : float
            The perplexity parameter for t-SNE
        n_components : int
            Number of components for t-SNE output
        learning_rate : str or float
            Learning rate for t-SNE
        n_iter : int
            Number of iterations for optimization
        random_state : int
            Random state for reproducibility

        Returns
        -------
        np.ndarray
            t-SNE embedding of shape (n_samples, n_components)
        """
        print(
            f"Applying t-SNE to {features.shape[0]} samples with {features.shape[1]} features..."
        )

        # Handle learning_rate type conversion for scikit-learn compatibility
        if isinstance(learning_rate, str) and learning_rate != "auto":
            raise ValueError("learning_rate string must be 'auto'")

        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=n_iter,
            random_state=random_state,
            verbose=1,
        )

        embeddings = tsne.fit_transform(features)
        print(f"t-SNE completed. Final KL divergence: {tsne.kl_divergence_:.4f}")

        return embeddings

    def visualize_tsne(
        self,
        embeddings: np.ndarray,
        labels: List[Tuple[int, int]],
        class_names: List[str],
        color_by: str = "domain",
        figsize: Tuple[int, int] = (12, 8),
        alpha: float = 0.7,
        s: int = 20,
    ) -> Figure:
        """
        Visualize t-SNE embeddings with color coding.

        Parameters
        ----------
        embeddings : np.ndarray
            t-SNE embeddings of shape (n_samples, 2)
        labels : List[Tuple[int, int]]
            List of (domain_id, class_id) tuples
        class_names : List[str]
            List of class names
        color_by : str
            What to color by: 'domain', 'class', or 'domain_class'
        figsize : Tuple[int, int]
            Figure size
        alpha : float
            Point transparency
        s : int
            Point size

        Returns
        -------
        Figure
            The matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Prepare data for plotting
        df = pd.DataFrame(
            {
                "x": embeddings[:, 0],
                "y": embeddings[:, 1],
                "domain_id": [label[0] for label in labels],
                "class_id": [label[1] for label in labels],
                "class_name": class_names,
            }
        )

        if color_by == "domain":
            # Color by domain
            unique_domains = df["domain_id"].unique()
            colors = plt.cm.get_cmap("Set1")(np.linspace(0, 1, len(unique_domains)))

            for i, domain in enumerate(unique_domains):
                domain_data = df[df["domain_id"] == domain]
                ax.scatter(
                    domain_data["x"],
                    domain_data["y"],
                    c=[colors[i]],
                    label=f"Domain {domain}",
                    alpha=alpha,
                    s=s,
                )

        elif color_by == "class":
            # Color by class within domain
            df["class_label"] = (
                df["domain_id"].astype(str) + "_" + df["class_id"].astype(str)
            )
            unique_classes = df["class_label"].unique()

            if len(unique_classes) <= 20:
                colors = plt.cm.get_cmap("tab20")(
                    np.linspace(0, 1, len(unique_classes))
                )
            else:
                colors = plt.cm.get_cmap("hsv")(np.linspace(0, 1, len(unique_classes)))

            for i, class_label in enumerate(unique_classes):
                class_data = df[df["class_label"] == class_label]
                domain_id, class_id = class_label.split("_")
                ax.scatter(
                    class_data["x"],
                    class_data["y"],
                    c=[colors[i]],
                    label=f"D{domain_id}C{class_id}",
                    alpha=alpha,
                    s=s,
                )

        elif color_by == "domain_class":
            # Use different markers for domains and colors for classes
            domains = df["domain_id"].unique()
            markers = ["o", "s", "^", "v", "D", "P", "*", "X"]

            for domain_idx, domain in enumerate(domains):
                domain_data = df[df["domain_id"] == domain]
                unique_classes_in_domain = domain_data["class_id"].unique()

                if len(unique_classes_in_domain) <= 10:
                    colors = plt.cm.get_cmap("tab10")(
                        np.linspace(0, 1, len(unique_classes_in_domain))
                    )
                else:
                    colors = plt.cm.get_cmap("hsv")(
                        np.linspace(0, 1, len(unique_classes_in_domain))
                    )

                for class_idx, class_id in enumerate(unique_classes_in_domain):
                    class_data = domain_data[domain_data["class_id"] == class_id]
                    marker = markers[domain_idx % len(markers)]

                    ax.scatter(
                        class_data["x"],
                        class_data["y"],
                        c=[colors[class_idx]],
                        marker=marker,
                        label=f"D{domain}C{class_id}",
                        alpha=alpha,
                        s=s,
                    )

        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.set_title(f"t-SNE Visualization (colored by {color_by})")

        # Add legend with proper handling for many classes
        _, legend_labels = ax.get_legend_handles_labels()
        if len(legend_labels) <= 20:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            print(
                f"Too many classes ({len(legend_labels)}) for legend. Legend omitted."
            )

        plt.tight_layout()
        return fig

    def analyze_and_visualize(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = 1000,
        perplexity: float = 30.0,
        color_by: str = "domain",
        save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Figure]:
        """
        Complete pipeline: extract features, apply t-SNE, and visualize.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader containing the data to analyze
        max_samples : Optional[int]
            Maximum number of samples to process
        perplexity : float
            t-SNE perplexity parameter
        color_by : str
            What to color by in visualization
        save_path : Optional[str]
            Path to save the visualization (optional)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Figure]
            - Extracted features
            - t-SNE embeddings
            - Matplotlib figure
        """
        # Extract features
        features, labels, class_names = self.extract_features(dataloader, max_samples)

        # Apply t-SNE
        embeddings = self.apply_tsne(features, perplexity=perplexity)

        # Visualize
        fig = self.visualize_tsne(embeddings, labels, class_names, color_by=color_by)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to {save_path}")

        return features, embeddings, fig


def analyze_universal_model_tsne(
    model: torch.nn.Module,
    dataloader: DataLoader,
    max_samples: int = 1000,
    perplexity: float = 30.0,
    color_by: str = "domain",
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Figure]:
    """
    Convenience function to analyze a universal model with t-SNE.

    Parameters
    ----------
    model : torch.nn.Module
        The trained universal ResNet model
    dataloader : DataLoader
        DataLoader containing the data to analyze
    max_samples : int
        Maximum number of samples to process
    perplexity : float
        t-SNE perplexity parameter
    color_by : str
        What to color by: 'domain', 'class', or 'domain_class'
    save_path : Optional[str]
        Path to save the visualization (optional)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Figure]
        - Extracted features
        - t-SNE embeddings
        - Matplotlib figure
    """
    analyzer = UniversalModelAnalyzer(model)
    return analyzer.analyze_and_visualize(
        dataloader=dataloader,
        max_samples=max_samples,
        perplexity=perplexity,
        color_by=color_by,
        save_path=save_path,
    )


def analyze_universal_output_layer_tsne(
    model: torch.nn.Module,
    dataloader: DataLoader,
    max_samples: int = 1000,
    perplexity: float = 30.0,
    color_by: str = "domain",
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Figure]:
    """
    Convenience function specifically for analyzing the universal output layer with t-SNE.

    This function analyzes the final output layer of the universal ResNet model that contains
    universal class activations before they are converted to domain-specific predictions
    via the taxonomy mapping.

    Parameters
    ----------
    model : torch.nn.Module
        The trained universal ResNet model
    dataloader : DataLoader
        DataLoader containing the data to analyze
    max_samples : int
        Maximum number of samples to process
    perplexity : float
        t-SNE perplexity parameter
    color_by : str
        What to color by: 'domain', 'class', or 'domain_class'
    save_path : Optional[str]
        Path to save the visualization (optional)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Figure]
        - Universal features from output layer
        - t-SNE embeddings
        - Matplotlib figure
    """
    analyzer = UniversalModelAnalyzer(model)
    return analyzer.analyze_universal_output_tsne(
        dataloader=dataloader,
        max_samples=max_samples,
        perplexity=perplexity,
        color_by=color_by,
        save_path=save_path,
    )
