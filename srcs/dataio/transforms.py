#!/usr/bin/env python3
"""
Helper functions for creating transformation pipelines for data loading.
"""
from __future__ import annotations

from typing import Optional, Tuple

from srcs.cli.Transformation import create_transform_function


def create_training_transform(
    config_path: str = "transform/config.yaml",
    transform_types: Optional[Tuple[str, ...]] = None,
    apply_augmentation: bool = True,
):
    """
    Créer une fonction de transformation optimisée pour l'entraînement.

    Args:
        config_path: Chemin vers la configuration des transformations
        transform_types: Types de transformations (par défaut: ["Blur", "Mask"])
        apply_augmentation: Activer l'augmentation de données

    Returns:
        Fonction de transformation compatible avec ManifestSequence

    Example:
        ```python
        # Transformation standard pour l'entraînement
        transform = create_training_transform()

        # Transformation avec masquage et détection de maladie
        transform = create_training_transform(
            transform_types=("Mask", "Brown"),
            apply_augmentation=True
        )

        # Utilisation avec ManifestSequence
        sequence = ManifestSequence(
            items=train_items,
            label2idx=label_mapping,
            img_size=224,
            batch_size=32,
            shuffle=True,
            seed=42,
            transform=transform
        )
        ```
    """
    # Laisser None pour que la pipeline choisisse l'ordre optimal (DEFAULT_TYPES)

    return create_transform_function(
        config_path=config_path,
        transform_types=transform_types,
        apply_augmentation=apply_augmentation,
    )


def create_inference_transform(
    config_path: str = "transform/config.yaml",
    transform_types: Optional[Tuple[str, ...]] = None,
):
    """
    Créer une fonction de transformation pour l'inférence (sans augmentation).

    Args:
        config_path: Chemin vers la configuration des transformations
        transform_types: Types de transformations

    Returns:
        Fonction de transformation pour l'inférence
    """
    # Laisser None pour que la pipeline choisisse l'ordre optimal (DEFAULT_TYPES)

    return create_transform_function(
        config_path=config_path,
        transform_types=transform_types,
        apply_augmentation=False,
    )


def create_minimal_transform():
    """
    Créer une transformation minimale (redimensionnement seulement).

    Returns:
        Fonction de transformation minimale
    """
    return create_transform_function(
        config_path=None,  # Pas de configuration -> fallback simple
        transform_types=(),
        apply_augmentation=False,
    )
