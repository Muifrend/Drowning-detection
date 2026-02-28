"""Configuration values for local drowning detection edge app."""

# Pool dimensions (meters)
POOL_W = 25.0
POOL_L = 50.0

# Speeds
DECK_SPEED = 3.0   # m/s
SWIM_SPEED = 1.0   # m/s

# Lifeguard positions in pool coords (meters from top-left)
LIFEGUARD_A = (2.0, 25.0)   # left side midpoint
LIFEGUARD_B = (23.0, 25.0)  # right side midpoint

# Agent thresholds
ALERT_THRESHOLD = 0.6      # p_distress to trigger ALERT
DISPATCH_THRESHOLD = 0.75  # p_distress to trigger DISPATCH
ESCALATE_THRESHOLD = 0.9   # p_distress to trigger ESCALATE
UNRESPONSIVE_SECONDS = 5   # seconds above threshold to escalate
TEMPORAL_WINDOW = 10       # frames to smooth p_distress over

# Display
FRAME_W = 1280
FRAME_H = 720
MINIMAP_W = 400
MINIMAP_H = 300
INFERENCE_EVERY = 5        # run PaliGemma every N frames

# Source
SOURCE = 0  # 0 = webcam, or path to video file
