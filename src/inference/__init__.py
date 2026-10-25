"""Inference utilities: full-volume segmentation shared by the CLI and the web app."""

from src.inference.volume_segmenter import MODALITIES, load_model, segment_volume

__all__ = ["MODALITIES", "load_model", "segment_volume"]
