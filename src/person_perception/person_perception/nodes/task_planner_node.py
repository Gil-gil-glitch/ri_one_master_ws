"""
Task Planner Node: Coordinator for Person Learning System
===========================================================
Central coordinator node that bridges Computer Vision and NLP subsystems.
Subscribes to /tracked_person and /person_profile, orchestrates actions.

Part of the 3-node Person Learning System architecture:
  /vision_node -> /person_tracker -> /task_planner
                                          ^
  /nlp_node (person_profile) -------------|

Responsibilities:
  - Receive structured person info from NLP (name, attributes)
  - Associate profile data with tracked person IDs
  - Store learned persons in persistent memory (JSON database)
  - Trigger confirmation behaviors (Dialogue, Navigation, Memory Update)
"""

import json
import os
import time
from typing import Optional, Dict, List

# Conditional import for ROS 2
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object
    String = object  # Placeholder for type annotations


class PersonDatabase:
    """
    Simple JSON-based persistent person database.
    
    Schema per person:
    {
        "name": "Jonathan",
        "track_id": 1,
        "first_seen": 1709500000,
        "last_seen": 1709500100,
        "total_interactions": 5,
        "biometrics": {"age": 22, "gender": "Male"},
        "attributes": ["Red_Shirt", "Glasses"],
        "status": "known"  // known | learning | unknown
    }
    """
    
    def __init__(self, db_path: str = 'person_database.json'):
        self.db_path = db_path
        self.persons: Dict[str, Dict] = {}
        self._load()
    
    def _load(self):
        """Load database from disk."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    self.persons = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.persons = {}
    
    def save(self):
        """Save database to disk."""
        try:
            os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
            with open(self.db_path, 'w') as f:
                json.dump(self.persons, f, indent=2)
        except IOError as e:
            print(f'[PersonDB] Failed to save: {e}')
    
    def get_person(self, name: str) -> Optional[Dict]:
        """Look up a person by name."""
        return self.persons.get(name)
    
    def upsert_person(
        self,
        name: str,
        track_id: Optional[int] = None,
        biometrics: Optional[Dict] = None,
        attributes: Optional[List[str]] = None,
        status: str = 'known'
    ):
        """Insert or update a person record."""
        now = int(time.time())
        
        if name in self.persons:
            # Update existing
            record = self.persons[name]
            record['last_seen'] = now
            record['total_interactions'] = record.get('total_interactions', 0) + 1
            if track_id is not None:
                record['track_id'] = track_id
            if biometrics:
                record['biometrics'].update(biometrics)
            if attributes:
                record['attributes'] = attributes
            record['status'] = status
        else:
            # Insert new
            self.persons[name] = {
                'name': name,
                'track_id': track_id,
                'first_seen': now,
                'last_seen': now,
                'total_interactions': 1,
                'biometrics': biometrics or {},
                'attributes': attributes or [],
                'status': status
            }
        
        self.save()
    
    def get_all_known(self) -> List[Dict]:
        """Get all known persons."""
        return [
            p for p in self.persons.values()
            if p.get('status') == 'known'
        ]


class TaskPlannerNode(Node):
    """
    ROS 2 Coordinator Node for the Person Learning System.
    
    Responsibility:
    - Receive tracked person data from /person_tracker
    - Receive person profile info from NLP (/person_profile)
    - Associate NLP-provided names with tracked person IDs
    - Store learned persons in persistent database
    - Decide and publish actions (Dialogue, Navigation, Memory Update)
    
    Subscribes:
        /tracked_person  (from /person_tracker)
        /person_profile  (from NLP team)
    
    Publishes:
        /task_planner/actions
    """
    
    def __init__(self):
        if ROS2_AVAILABLE:
            super().__init__('task_planner')
            
            self.declare_parameter('db_path', 'person_database.json')
            self.declare_parameter('show_log', True)
            
            db_path = self.get_parameter('db_path').value
            self.show_log = self.get_parameter('show_log').value
        else:
            db_path = 'person_database.json'
            self.show_log = True
        
        # Person database
        self.db = PersonDatabase(db_path)
        self._log(f'Database loaded: {len(self.db.persons)} known persons')
        
        # Pending learning requests: track_id -> pending info
        self._pending_learns: Dict[int, Dict] = {}
        
        # Latest tracked person data (for association)
        self._latest_tracked: Dict[int, Dict] = {}
        
        # Action publisher (ROS 2 only)
        self.action_publisher = None
        if ROS2_AVAILABLE:
            self.action_publisher = self.create_publisher(
                String,
                '/task_planner/actions',
                10
            )
            
            # Subscribe to tracked persons from CV
            self.tracked_sub = self.create_subscription(
                String,
                '/tracked_person',
                self._tracked_person_callback,
                10
            )
            
            # Subscribe to person profiles from NLP
            self.profile_sub = self.create_subscription(
                String,
                '/person_profile',
                self._person_profile_callback,
                10
            )
        
        self._log('Task Planner Node started!')
    
    def _log(self, msg: str, level: str = 'info'):
        """Log via ROS 2 or print."""
        if ROS2_AVAILABLE and hasattr(self, 'get_logger'):
            logger = self.get_logger()
            if level == 'error':
                logger.error(msg)
            elif level == 'warn':
                logger.warn(msg)
            else:
                logger.info(msg)
        else:
            print(f'[TaskPlanner] [{level.upper()}] {msg}')
    
    def _tracked_person_callback(self, msg: String):
        """Handle /tracked_person messages from person_tracker."""
        try:
            data = json.loads(msg.data)
            self.process_tracked_persons(data)
        except json.JSONDecodeError as e:
            self._log(f'Invalid JSON from /tracked_person: {e}', level='warn')
    
    def _person_profile_callback(self, msg: String):
        """
        Handle /person_profile messages from NLP team.
        
        Expected schema:
        {
            "track_id": 1,
            "name": "Jonathan",
            "confirmed": true
        }
        """
        try:
            data = json.loads(msg.data)
            self.process_person_profile(data)
        except json.JSONDecodeError as e:
            self._log(f'Invalid JSON from /person_profile: {e}', level='warn')
    
    def process_tracked_persons(self, data: Dict) -> List[Dict]:
        """
        Process tracked person data and decide actions.
        
        Args:
            data: Message from /tracked_person containing tracked_persons list
            
        Returns:
            List of action dicts to be published/executed
        """
        tracked_persons = data.get('tracked_persons', [])
        actions = []
        
        for person in tracked_persons:
            track_id = person.get('track_id')
            name = person.get('name', 'Unknown')
            action = person.get('action', 'OBSERVE')
            similarity = person.get('similarity', 0.0)
            uncertainty = person.get('uncertainty', 1.0)
            biometrics = person.get('biometrics', {})
            attributes = person.get('attributes', [])
            
            # Store latest tracked data for association
            self._latest_tracked[track_id] = person
            
            # Process based on action from person_tracker
            action_data = self._decide_action(
                track_id=track_id,
                name=name,
                action=action,
                similarity=similarity,
                uncertainty=uncertainty,
                biometrics=biometrics,
                attributes=attributes
            )
            
            if action_data:
                actions.append(action_data)
                
                # Publish action
                if self.action_publisher is not None:
                    msg = String()
                    msg.data = json.dumps(action_data)
                    self.action_publisher.publish(msg)
                
                if self.show_log:
                    self._log_action(action_data)
        
        return actions
    
    def _decide_action(
        self,
        track_id: int,
        name: str,
        action: str,
        similarity: float,
        uncertainty: float,
        biometrics: Dict,
        attributes: List[str]
    ) -> Optional[Dict]:
        """
        Decide what action to take based on tracked person state.
        
        Implements the "Learning a Person" flow from the presentation:
        1. GREET: Known person -> greet by name, update DB
        2. LEARN: Unknown person -> request introduction from NLP
        3. ASK_CLARIFICATION: Uncertain -> request clarification
        4. OBSERVE: No specific action needed
        """
        timestamp = int(time.time() * 1000)
        
        if action == 'GREET' and name != 'Unknown':
            # Known person — update database and greet
            self.db.upsert_person(
                name=name,
                track_id=track_id,
                biometrics=biometrics,
                attributes=attributes,
                status='known'
            )
            
            return {
                'timestamp': timestamp,
                'action_type': 'GREET',
                'track_id': track_id,
                'target_name': name,
                'dialogue': f'Hello {name}! Nice to see you again.',
                'memory_update': True,
                'details': {
                    'similarity': round(similarity, 4),
                    'biometrics': biometrics,
                    'total_interactions': self.db.persons.get(name, {}).get(
                        'total_interactions', 1
                    )
                }
            }
        
        elif action == 'LEARN':
            # Unknown person — initiate learning flow
            # Add to pending learns (NLP will provide the name)
            if track_id not in self._pending_learns:
                self._pending_learns[track_id] = {
                    'started': timestamp,
                    'biometrics': biometrics,
                    'attributes': attributes
                }
            
            return {
                'timestamp': timestamp,
                'action_type': 'LEARN',
                'track_id': track_id,
                'target_name': None,
                'dialogue': 'I don\'t think we\'ve met. What is your name?',
                'memory_update': False,
                'details': {
                    'request': 'introduction',
                    'awaiting_name': True
                }
            }
        
        elif action == 'ASK_CLARIFICATION':
            return {
                'timestamp': timestamp,
                'action_type': 'ASK_CLARIFICATION',
                'track_id': track_id,
                'target_name': name if name != 'Unknown' else None,
                'dialogue': f'Sorry, are you {name}?' if name != 'Unknown'
                    else 'I\'m not sure who you are. Could you tell me your name?',
                'memory_update': False,
                'details': {
                    'uncertainty': round(uncertainty, 4),
                    'best_guess': name
                }
            }
        
        # OBSERVE — no action needed
        return None
    
    def process_person_profile(self, profile: Dict):
        """
        Process person profile from NLP team.
        Associates the provided name with a tracked person ID.
        
        Args:
            profile: Dict with 'track_id', 'name', and 'confirmed'
        """
        track_id = profile.get('track_id')
        name = profile.get('name')
        confirmed = profile.get('confirmed', False)
        
        if not track_id or not name:
            self._log('Invalid profile: missing track_id or name', level='warn')
            return
        
        if confirmed:
            # Get biometrics/attributes from pending learn or latest track
            bio = {}
            attrs = []
            
            if track_id in self._pending_learns:
                pending = self._pending_learns.pop(track_id)
                bio = pending.get('biometrics', {})
                attrs = pending.get('attributes', [])
            elif track_id in self._latest_tracked:
                latest = self._latest_tracked[track_id]
                bio = latest.get('biometrics', {})
                attrs = latest.get('attributes', [])
            
            # Save to database
            self.db.upsert_person(
                name=name,
                track_id=track_id,
                biometrics=bio,
                attributes=attrs,
                status='known'
            )
            
            self._log(f'Person learned: {name} (track_id={track_id})')
            
            # Publish confirmation action
            if self.action_publisher is not None:
                action_data = {
                    'timestamp': int(time.time() * 1000),
                    'action_type': 'CONFIRM_LEARN',
                    'track_id': track_id,
                    'target_name': name,
                    'dialogue': f'Nice to meet you, {name}! I\'ll remember you.',
                    'memory_update': True
                }
                msg = String()
                msg.data = json.dumps(action_data)
                self.action_publisher.publish(msg)
    
    def _log_action(self, action: Dict):
        """Pretty-print an action for debugging."""
        action_type = action.get('action_type', '?')
        track_id = action.get('track_id', '?')
        name = action.get('target_name', 'Unknown')
        dialogue = action.get('dialogue', '')
        
        self._log(
            f'[ACTION] {action_type} | Track:{track_id} | '
            f'Target:{name} | "{dialogue}"'
        )
    
    def shutdown(self):
        """Clean shutdown."""
        self.db.save()
        self._log(f'Database saved: {len(self.db.persons)} persons')
        
        if ROS2_AVAILABLE:
            rclpy.shutdown()


def main(args=None):
    """Entry point for the task planner node."""
    if not ROS2_AVAILABLE:
        print("=" * 60)
        print("WARNING: ROS 2 not detected — running in standalone mode")
        print("Use tools/run_live_system.py for full pipeline testing")
        print("=" * 60)
        return
    
    rclpy.init(args=args)
    node = TaskPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received')
    finally:
        node.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()
