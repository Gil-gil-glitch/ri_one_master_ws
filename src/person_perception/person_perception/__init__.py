"""
Person Perception Package
Research-Grade Person Perception System for RoboCup JapanOpen

This package provides:
- Vision: YOLOv8-based person detection with RealSense depth
- Identity: InsightFace-based face recognition with embedding comparison
- Research: Entropy-based uncertainty calculation for active perception
"""

__version__ = '1.0.0'
__author__ = 'Jonathan Setiawan'

from . import core
# from . import nodes
