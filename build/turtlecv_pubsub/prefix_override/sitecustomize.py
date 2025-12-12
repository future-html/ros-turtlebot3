import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/future/ros2_turtlecv/install/turtlecv_pubsub'
