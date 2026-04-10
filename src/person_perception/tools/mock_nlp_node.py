#!/usr/bin/env python3
"""
Mock NLP Node: The "Golden Standard" Tool
==========================================
Subscribes to /perception/person_info and pretty-prints the data
for debugging and integration testing with the NLP team.
"""

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MockNLPNode(Node):
    """
    Mock NLP subscriber that displays perception data in a human-readable format.
    
    This node acts as a stand-in for the actual NLP system during development
    and testing, providing clear visual feedback of the perception pipeline output.
    """
    
    def __init__(self):
        super().__init__('mock_nlp_node')
        
        # Subscribe to person perception topic
        self.subscription = self.create_subscription(
            String,
            '/perception/person_info',
            self.perception_callback,
            10
        )
        
        self.get_logger().info('Mock NLP Node started - listening to /perception/person_info')
        self.get_logger().info('=' * 60)
    
    def perception_callback(self, msg: String):
        """
        Process and display perception data in a pretty format.
        
        Expected JSON schema:
        {
            "is_human": true,
            "id": "Jonathan",
            "uncertainty": 0.1,
            "attributes": ["Male", "22", "Red_Shirt", "Backpack"],
            "action": "GREET"
        }
        """
        try:
            data = json.loads(msg.data)
            
            # Skip if no human detected
            if not data.get('is_human', False):
                return
            
            # Extract data
            user_id = data.get('id', 'Unknown')
            uncertainty = data.get('uncertainty', 1.0)
            attributes = data.get('attributes', [])
            action = data.get('action', 'OBSERVE')
            
            # Parse biometrics from attributes
            gender = None
            age = None
            features = []
            
            for attr in attributes:
                if attr in ['Male', 'Female']:
                    gender = attr
                elif attr.isdigit():
                    age = attr
                else:
                    features.append(attr)
            
            # Build biometric string
            bio_parts = []
            if gender:
                bio_parts.append(gender)
            if age:
                bio_parts.append(age)
            bio_str = ', '.join(bio_parts) if bio_parts else 'N/A'
            
            # Print formatted output
            print()
            print(f"[NLP MOCK] Saw User: {user_id} ({bio_str})")
            print(f"[NLP MOCK] Features: {features}")
            print(f"[NLP MOCK] Action Triggered: {action}")
            print(f"[NLP MOCK] Uncertainty: {uncertainty:.3f}")
            print('-' * 50)
            
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'Invalid JSON received: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing perception data: {e}')


def main(args=None):
    """Entry point for the mock NLP node."""
    rclpy.init(args=args)
    
    node = MockNLPNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Mock NLP Node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
