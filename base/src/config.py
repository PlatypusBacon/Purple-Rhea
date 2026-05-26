# ── Scan geometry ──────────────────────────────────────────────────────────────
TOTAL_FRAMES   = 36
STEP_DEGREES   = 10.0
NOMINAL_RADIUS = 0.20        # metres — fallback only, used if h computation fails

# ── Rig physical constants (for compute_camera_distance formula) ───────────────
RIG_BASE_LENGTH   = 0.13    # r — horizontal base length (metres)
RIG_MIDDLE_LENGTH = 0.25    # x — middle segment length (metres)
RIG_FINAL_LENGTH  = 0.09    # y — final segment length (metres)
CAMERA_HEIGHT     = 0.12    # H — vertical height of camera above ground (metres)

# ── Voxel grid bounds (used in visual_hull.py AND surface.py) ─────────────────
VOXEL_XY_EXTENT = 0.12      # ±metres in XY — set to slightly more than object radius
VOXEL_Z_MIN     = -0.02     # metres below turntable surface
VOXEL_Z_MAX     = 0.22      # metres above turntable surface

# ── Camera / MQTT ─────────────────────────────────────────────────────────────
IMAGE_WIDTH  = 1600
IMAGE_HEIGHT = 1200
MQTT_BROKER  = "10.48.233.146"
MQTT_PORT    = 1883

# ── Pipeline ──────────────────────────────────────────────────────────────────
NOMINAL_RADIUS              = 0.20
RUN_SURFACE_RECON           = True
SURFACE_METHOD              = "poisson"
POISSON_DEPTH               = 9
POISSON_SCALE               = 1.1
POISSON_LINEAR_FIT          = False
POISSON_DENSITY_KEEP_PERCENTILE = 15
OUTLIER_NB_NEIGHBORS        = 20
OUTLIER_STD_RATIO           = 2.0
VOXEL_SIZE                  = None
SMOOTH_ITERATIONS           = 0
import os as _os
_PROJECT_ROOT               = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")
MESH_DIR                  = _os.path.join(_PROJECT_ROOT, "output/mesh")
OUTPUT_DIR                  = _os.path.join(_PROJECT_ROOT, "output")
POINT_DIR                  = _os.path.join(_PROJECT_ROOT, "output/point")
IMAGE_CACHE                 = _os.path.join(_PROJECT_ROOT, "output", "frames")
DEBUG_MODE                  = True
DEBUG_VIZ                   = True
WHITE                       = False
