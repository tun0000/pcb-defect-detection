"""Data preparation pipeline: download, VOC->YOLO conversion, anti-leakage split.

This subpackage depends on stdlib + Pillow + PyYAML only (no torch/ultralytics),
so tests and CI stay fast.
"""
