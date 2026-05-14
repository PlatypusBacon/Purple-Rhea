# Central configuration — edit here, nowhere else

SERIAL_PORT    = "COM15"       # change to /dev/ttyUSB0 on Linux/Pi
BAUD_RATE      = 115200
TOTAL_FRAMES   = 36
STEP_DEGREES   = 10.0
NOMINAL_RADIUS = 0.2          # metres — measure from your physical rig
DEBUG_MODE = True
# Image resolution — must match ESP32-CAM firmware framesize setting:
#   FRAMESIZE_QVGA  ->  320 x 240  (current firmware default)
#   FRAMESIZE_SVGA  ->  800 x 600  (recommended for reconstruction)
#   FRAMESIZE_XGA   -> 1024 x 768
IMAGE_WIDTH  = 3024
IMAGE_HEIGHT = 4032

MQTT_BROKER    = "localhost"
MQTT_PORT      = 1883
TOPIC_TRIGGER  = "rhea/camera/trigger"
TOPIC_IMAGE    = "rhea/camera/image"
TOPIC_LOCATION = "rhea/tracking/location"
TOPIC_RESULT   = "rhea/pc/result"

OUTPUT_DIR  = "output"
IMAGE_CACHE = "output/frames"