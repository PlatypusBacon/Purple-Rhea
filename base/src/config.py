# Central configuration — edit here, nowhere else

SERIAL_PORT    = "COM15"       # change to /dev/ttyUSB0 on Linux/Pi
BAUD_RATE      = 115200
TOTAL_FRAMES   = 36
STEP_DEGREES   = 10.0
NOMINAL_RADIUS = 0.2          # metres — measure from your physical rig
DEBUG_MODE = True


RUN_SURFACE_RECON = True
SURFACE_METHOD    = "ball_pivot"   # "ball_pivot" or "poisson"
# Image resolution — must match ESP32-CAM firmware framesize setting:
#   FRAMESIZE_QVGA  ->  320 x 240  (current firmware default)
#   FRAMESIZE_SVGA  ->  800 x 600  (recommended for reconstruction)
#   FRAMESIZE_XGA   -> 1024 x 768
IMAGE_WIDTH  = 756
IMAGE_HEIGHT = 1008
LIMIT_MEM=True

MQTT_BROKER    = "localhost"
MQTT_PORT      = 1883
TOPIC_TRIGGER  = "rhea/camera/trigger"
TOPIC_IMAGE    = "rhea/camera/image"
TOPIC_LOCATION = "rhea/tracking/location"
TOPIC_RESULT   = "rhea/pc/result"

OUTPUT_DIR  = "output"
IMAGE_CACHE = "output/frames"