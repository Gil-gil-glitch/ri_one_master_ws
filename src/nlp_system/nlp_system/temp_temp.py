"""
temp_temp.py — Mock Person Tracker Node
========================================
Simulates /tracked_person messages from person_tracker via keypress.
Use this to test task_planner and NLP without a camera or CV dependencies.

Controls:
  1 → GREET           (known person, high similarity)
  2 → LEARN           (unknown person)
  3 → ASK_CLARIFICATION (uncertain match)
  4 → OBSERVE         (no person detected)
  q → quit

Publishes to: /tracked_person  (same schema as person_tracker_node)
Node name:    /temp_temp
"""

import json
import time
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


PAYLOADS = {
    '1': {
        'description': 'GREET — known person (Jonathan)',
        'data': {
            'timestamp': None,
            'tracked_persons': [
                {
                    'track_id': 1,
                    'name': 'Jonathan',
                    'bbox': [100, 80, 300, 420],
                    'confidence': 0.91,
                    'similarity': 0.91,
                    'uncertainty': 0.08,
                    'biometrics': {
                        'age': 22,
                        'gender': 'Male'
                    },
                    'attributes': ['Blue_Shirt'],
                    'action': 'GREET',
                    'frames_seen': 42
                }
            ]
        }
    },
    '2': {
        'description': 'LEARN — unknown person',
        'data': {
            'timestamp': None,
            'tracked_persons': [
                {
                    'track_id': 2,
                    'name': 'Unknown',
                    'bbox': [120, 90, 310, 430],
                    'confidence': 0.0,
                    'similarity': 0.0,
                    'uncertainty': 0.95,
                    'biometrics': {
                        'age': 25,
                        'gender': 'Female'
                    },
                    'attributes': ['Red_Shirt'],
                    'action': 'LEARN',
                    'frames_seen': 5
                }
            ]
        }
    },
    '3': {
        'description': 'ASK_CLARIFICATION — uncertain match',
        'data': {
            'timestamp': None,
            'tracked_persons': [
                {
                    'track_id': 3,
                    'name': 'Jonathan',
                    'bbox': [110, 85, 305, 425],
                    'confidence': 0.55,
                    'similarity': 0.55,
                    'uncertainty': 0.52,
                    'biometrics': {
                        'age': 22,
                        'gender': 'Male'
                    },
                    'attributes': ['Black_Shirt'],
                    'action': 'ASK_CLARIFICATION',
                    'frames_seen': 12
                }
            ]
        }
    },
    '4': {
        'description': 'OBSERVE — no person detected',
        'data': {
            'timestamp': None,
            'tracked_persons': []
        }
    }
}


class TempTemp(Node):

    def __init__(self):
        super().__init__('temp_temp')

        self.publisher = self.create_publisher(
            String,
            '/tracked_person',
            10
        )

        self.get_logger().info('=' * 50)
        self.get_logger().info('temp_temp mock CV node ready')
        self.get_logger().info('Publishing to: /tracked_person')
        self.get_logger().info('=' * 50)
        self.get_logger().info('Press 1 → GREET (known person)')
        self.get_logger().info('Press 2 → LEARN (unknown person)')
        self.get_logger().info('Press 3 → ASK_CLARIFICATION (uncertain)')
        self.get_logger().info('Press 4 → OBSERVE (no person)')
        self.get_logger().info('Press q → quit')
        self.get_logger().info('=' * 50)

    def publish_scenario(self, key: str):
        if key not in PAYLOADS:
            return

        scenario = PAYLOADS[key]
        payload = scenario['data'].copy()

        # Inject live timestamp
        payload['timestamp'] = int(time.time() * 1000)

        msg = String()
        msg.data = json.dumps(payload)
        self.publisher.publish(msg)

        self.get_logger().info(
            f'Published: {scenario["description"]}'
        )
        self.get_logger().info(
            f'  → {msg.data}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = TempTemp()

    print('\nReady — type a key and press Enter:')

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            # Non-blocking input check
            import select
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.readline().strip().lower()

                if key == 'q':
                    print('Quitting...')
                    break
                elif key in PAYLOADS:
                    node.publish_scenario(key)
                else:
                    print(f'Unknown key: "{key}" — use 1, 2, 3, 4, or q')

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()