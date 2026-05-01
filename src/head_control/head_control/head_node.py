import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from dynamixel_sdk import *
import time

# ===== SETTINGS =====
DEVICENAME = '/dev/ttyACM0'   # change if needed
BAUDRATE = 1000000
PROTOCOL_VERSION = 1.0

PAN_ID = 1
TILT_ID = 2

ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
# ====================

class HeadController(Node):
    def __init__(self):
        super().__init__('head_controller')

        self.portHandler = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        # Open port
        if not self.portHandler.openPort():
            self.get_logger().error("Failed to open port")
            return

        # 🔥 CRITICAL: wait for OpenCR reset
        self.get_logger().info("Waiting for OpenCR to initialize...")
        time.sleep(2)

        # Set baudrate
        if not self.portHandler.setBaudRate(BAUDRATE):
            self.get_logger().error("Failed to set baudrate")
            return

        # Enable torque
        for dxl_id in [PAN_ID, TILT_ID]:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE
            )

            if dxl_comm_result != COMM_SUCCESS:
                self.get_logger().error(f"Torque enable failed for ID {dxl_id}")
            else:
                self.get_logger().info(f"Torque enabled on ID {dxl_id}")

        time.sleep(0.1)

        # 🔥 Test movement (optional but very useful)
        self.get_logger().info("Testing motors...")
        self.packetHandler.write2ByteTxRx(
            self.portHandler, PAN_ID, ADDR_GOAL_POSITION, 492
        )
        self.packetHandler.write2ByteTxRx(
            self.portHandler, TILT_ID, ADDR_GOAL_POSITION, 512
        )

        # Subscriber
        self.subscription = self.create_subscription(
            Int32MultiArray,
            '/head_cmd',
            self.callback,
            10
        )

        self.get_logger().info("Head controller ready")

    def callback(self, msg):
        if len(msg.data) < 2:
            self.get_logger().warning("Invalid message length")
            return

        pan = int(msg.data[0])
        tilt = int(msg.data[1])

        # Clamp for safety
        pan = max(0, min(1023, pan))
        tilt = max(0, min(1023, tilt))

        self.get_logger().info(f"Pan: {pan}, Tilt: {tilt}")

        # Send to pan motor
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
            self.portHandler, PAN_ID, ADDR_GOAL_POSITION, pan
        )

        if dxl_comm_result != COMM_SUCCESS:
            self.get_logger().error(f"Pan write failed: {dxl_comm_result}")

        # Send to tilt motor
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
            self.portHandler, TILT_ID, ADDR_GOAL_POSITION, tilt
        )

        if dxl_comm_result != COMM_SUCCESS:
            self.get_logger().error(f"Tilt write failed: {dxl_comm_result}")

def main():
    rclpy.init()
    node = HeadController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # Disable torque on shutdown
    for dxl_id in [PAN_ID, TILT_ID]:
        node.packetHandler.write1ByteTxRx(
            node.portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
        )

    node.portHandler.closePort()
    node.destroy_node()
    rclpy.shutdown()