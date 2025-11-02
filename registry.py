# registry.py
import pose.leap_binder as pose_binder
import detect.leap_binder as detect_binder

TASK_BINDERS = {
    "pose": pose_binder,
    "detect": detect_binder,
}
