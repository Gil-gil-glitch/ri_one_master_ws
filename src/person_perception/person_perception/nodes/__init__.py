"""
ROS 2 Node Implementations

Node Architecture (Person Learning System):
  /vision_node -> /person_tracker -> /task_planner
"""

from .person_node import PersonNode          # Legacy monolithic node
from .vision_node import VisionNode          # Person detection
from .person_tracker_node import PersonTrackerNode  # Identity tracking
from .task_planner_node import TaskPlannerNode      # Coordinator

__all__ = [
    'PersonNode',
    'VisionNode',
    'PersonTrackerNode',
    'TaskPlannerNode',
]
