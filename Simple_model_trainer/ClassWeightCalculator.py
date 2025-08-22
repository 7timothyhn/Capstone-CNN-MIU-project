import os
import math
from typing import Optional, Dict
import numpy as np
from sklearn.utils import class_weight


class ClassWeightCalculator:
    def calculate_class_weights(
        self,
        image_dataset_directory: str,
        method: str = "skBalanced",
        class_indices: Optional[Dict[str, int]] = None,
    ) -> Optional[Dict[int, float]]:
        """
        Calculates appropriate class weights for the given dataset. Images with higher occurrence will get a lower
        weight, than classes with only a few instances.

        Args:
            image_dataset_directory: Path to the dataset directory (should contain 'training' subfolder)
            method: Weight calculation method:
                - None: no class weights
                - "simple": 1/sqrt(n) weight per class
                - "skBalanced": sklearn balanced weights
            class_indices: Dictionary mapping class names to their indices

        Returns:
            Dictionary with class indices as keys and weights as values (includes all classes), or None if method is None
        """
        if method is None:
            return None

        if method not in ["simple", "skBalanced"]:
            raise ValueError(
                "Method must either be None, or one of the strings 'simple' or 'skBalanced', "
                f"but provided was {method}."
            )

        # Use training subdirectory for weight calculation
        train_dir = os.path.join(image_dataset_directory, "training")
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")

        # Get all class names from directory structure
        all_classes = sorted(
            [
                class_name
                for class_name in os.listdir(train_dir)
                if os.path.isdir(os.path.join(train_dir, class_name))
            ]
        )

        if not all_classes:
            return None

        # Count samples per class
        class_counts = {}
        for class_name in all_classes:
            class_dir = os.path.join(train_dir, class_name)
            count = len(
                [
                    f
                    for f in os.listdir(class_dir)
                    if os.path.isfile(os.path.join(class_dir, f))
                ]
            )
            class_counts[class_name] = count

        # If class_indices not provided, create default 0-based indices
        if class_indices is None:
            class_indices = {
                class_name: idx for idx, class_name in enumerate(all_classes)
            }

        # Calculate weights based on method
        if method == "simple":
            # Improved simple weighting with sqrt(total_samples/(n_classes * count))
            total_samples = sum(class_counts.values())
            num_classes = len(class_counts)

            weights = {
                class_name: (
                    math.sqrt(total_samples / (num_classes * count))
                    if count > 0
                    else 1.0
                )
                for class_name, count in class_counts.items()
            }
        else:  # skBalanced
            # sklearn's balanced weights
            classes = list(class_counts.keys())
            y = np.repeat(classes, list(class_counts.values()))
            weights = dict(
                zip(
                    classes,
                    class_weight.compute_class_weight("balanced", classes=classes, y=y),
                )
            )

        # Create weights dictionary with all class indices, defaulting to 1.0 for missing classes
        class_weights = {
            idx: weights.get(class_name, 1.0)
            for class_name, idx in class_indices.items()
        }

        # Debug prints
        print("\nClass distribution:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1]):
            print(f"{class_name}: {count}")

        print("\nCalculated weights:")
        for class_name, weight in sorted(weights.items(), key=lambda x: x[1]):
            print(f"{class_name}: {weight:.4f}")

        print("\nFinal class weights (with indices):")
        for idx, weight in sorted(class_weights.items(), key=lambda x: x[0]):
            class_name = [k for k, v in class_indices.items() if v == idx][0]
            print(f"{idx} ({class_name}): {weight:.4f}")

        return class_weights


if __name__ == "__main__":
    # Example usage
    calculator = ClassWeightCalculator()

    # Example class indices (would normally come from your data generator)
    example_indices = {"class1": 0, "class2": 1, "class3": 2}

    try:
        print("\n=== Simple Weights ===")
        simple_weights = calculator.calculate_class_weights(
            "data/images", "simple", example_indices
        )

        print("\n=== Balanced Weights ===")
        balanced_weights = calculator.calculate_class_weights(
            "data/images", "skBalanced", example_indices
        )
    except Exception as e:
        print(f"\nError calculating weights: {e}")
