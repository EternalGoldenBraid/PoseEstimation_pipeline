import pyrealsense2 as rs
import numpy as np
import cv2

# 创建RealSense管道
pipeline = rs.pipeline()

# 创建配置对象
config = rs.config()

# 启用深度流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 启动管道
pipeline.start(config)

try:
    while True:
        # 等待一帧数据
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # 将深度数据转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())

        # 将深度图像转换为8位灰度图像以便显示
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 显示深度图像
        cv2.imshow('RealSense Depth', depth_colormap)

        # 按下'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 停止管道
    pipeline.stop()

# 关闭所有窗口
cv2.destroyAllWindows()
