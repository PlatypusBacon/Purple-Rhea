# Central configuration — edit here, nowhere else

SERIAL_PORT    = "/dev/cu.usbserial-A5069RR4"       # change to /dev/ttyUSB0 on Linux/Pi
BAUD_RATE      = 921600

# XIAO BLE Sense pose tracker on USB CDC-ACM. Set to None to disable and fall
# back to servo angle / inline LOCATION lines from the camera serial.
TRACKER_SERIAL_PORT = None                          # e.g. "/dev/ttyACM0"
TRACKER_BAUD_RATE   = 115200                        # CDC-ACM ignores baud
TOTAL_FRAMES   = 36
STEP_DEGREES   = 10.0
NOMINAL_RADIUS = 0.20          # metres — measure from your physical rig
CAMERA_HEIGHT = 0.20  # metres above turntable plane
DEBUG_MODE = True
DEBUG_VIZ = True   # set False to skip all debug image saving
WHITE = False


RUN_SURFACE_RECON = True
SURFACE_METHOD    = "poisson"#ball_pivot"  "ball_pivot" or "poisson"
POISSON_DEPTH               = 9           # 8–10; higher = finer detail
POISSON_SCALE               = 1.1
POISSON_LINEAR_FIT          = False
POISSON_DENSITY_KEEP_PERCENTILE = 15
OUTLIER_NB_NEIGHBORS        = 20
OUTLIER_STD_RATIO           = 2.0
VOXEL_SIZE                  = None        # None = auto
SMOOTH_ITERATIONS           = 0           # Taubin passes; 0 = off
CAMERA_CENTROID             = None    
# Image resolution — must match ESP32-CAM firmware framesize setting:
#   FRAMESIZE_QVGA  ->  320 x 240  (current firmware default)
#   FRAMESIZE_SVGA  ->  800 x 600  (recommended for reconstruction)
#   FRAMESIZE_XGA   -> 1024 x 768
IMAGE_WIDTH  = 1600
IMAGE_HEIGHT = 1200
LIMIT_MEM=True

MQTT_BROKER    = "localhost"
MQTT_PORT      = 1883
TOPIC_TRIGGER  = "rhea/camera/trigger"
TOPIC_IMAGE    = "rhea/camera/image"
TOPIC_LOCATION = "rhea/tracking/location"
TOPIC_RESULT   = "rhea/pc/result"

OUTPUT_DIR  = "output"
IMAGE_CACHE = "../../output/frames"