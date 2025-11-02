from ultralytics.tensorleap_folder.get_cfg import cfg
task=cfg.task


if task=="pose":
    import pose.leap_binder as leap_binder
    import ultralytics.tensorleap_folder.pose.global_params as global_params
    import ultralytics.tensorleap_folder.pose.utils as utils

else:
    import detect.leap_binder as leap_binder
    import ultralytics.tensorleap_folder.detect.global_params as global_params
    import ultralytics.tensorleap_folder.detect.utils as utils