import open3d as o3d
import numpy as np
from PIL import Image
import os
import tempfile
import matplotlib.pyplot as plt
import cv2
from loguru import logger
from pathlib import Path
import time


class PointCloudAnimator:
    """点云动画生成器类
    
    用于生成点云旋转的GIF或视频，提供统一的接口和参数控制。
    支持两种旋转模式：
    1. 相机旋转（默认）：保持点云静止，旋转相机视角
    2. 物体旋转：保持相机静止，旋转点云物体
    """
    
    def __init__(self, ply_path):
        """初始化点云动画生成器
        
        Args:
            ply_path (str): 点云文件路径
        """
        logger.info(f"初始化点云动画生成器，加载点云文件: {ply_path}")
        self.ply_path = ply_path
        self.pcd = o3d.io.read_point_cloud(ply_path)
        
        # 默认设置
        self.start_angle = -90
        self.end_angle = 90
        self.zoom_level = 0.1
        self.front = [0, 0, -1]
        self.lookat = [0, 0, 1]
        self.up = [0, -1, 0]
        self.point_size = 1.0  # 默认点大小
        self.width = 1280  # 默认宽度
        self.height = 720  # 默认高度
        self.rotation_mode = 'camera'  # 默认旋转模式：'camera' 或 'object'
        
    def set_view_params(self, front=None, lookat=None, up=None, zoom=None, point_size=None):
        """设置视角参数
        
        Args:
            front (list, optional): 视角前方向
            lookat (list, optional): 视角焦点
            up (list, optional): 视角上方向
            zoom (float, optional): 缩放级别
            point_size (float, optional): 点云点大小
        """
        if front is not None:
            self.front = front
        if lookat is not None:
            self.lookat = lookat
        if up is not None:
            self.up = up
        if zoom is not None:
            self.zoom_level = zoom
        if point_size is not None:
            self.point_size = point_size
        logger.debug(f"更新视角参数: front={self.front}, lookat={self.lookat}, up={self.up}, zoom={self.zoom_level}, point_size={self.point_size}")
    
    def set_frame_size(self, width, height):
        """设置输出帧的大小
        
        Args:
            width (int): 帧宽度
            height (int): 帧高度
        """
        self.width = width
        self.height = height
        logger.debug(f"设置帧大小: {width}x{height}")
        
    def set_angle_range(self, start_angle, end_angle):
        """设置旋转角度范围
        
        Args:
            start_angle (float): 开始角度
            end_angle (float): 结束角度
        """
        self.start_angle = start_angle
        self.end_angle = end_angle
        logger.debug(f"设置角度范围: {start_angle}° 到 {end_angle}°")
        
    def set_rotation_mode(self, mode):
        """设置旋转模式
        
        Args:
            mode (str): 'camera' 或 'object'
        """
        if mode not in ['camera', 'object']:
            raise ValueError("旋转模式必须是 'camera' 或 'object'")
        self.rotation_mode = mode
        logger.info(f"设置旋转模式: {mode}")
        
    def _generate_frames(self, num_frames, axis='horizontal'):
        """生成点云旋转的帧序列
        
        Args:
            num_frames (int): 帧数量
            axis (str, optional): 旋转轴，'horizontal'或'vertical'
        
        Returns:
            list: 帧列表
        """
        logger.info(f"开始生成帧序列: 帧数={num_frames}, 旋转轴={axis}, 旋转模式={self.rotation_mode}")
        frames = []
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=self.width, height=self.height)
        
        # 创建点云副本用于旋转
        pcd = o3d.geometry.PointCloud(self.pcd)
        vis.add_geometry(pcd)
        
        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.point_size = self.point_size
        
        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_front(self.front)
        ctr.set_lookat(self.lookat)
        ctr.set_up(self.up)
        ctr.set_zoom(self.zoom_level)
        
        # 计算角度范围和每帧旋转角度
        total_angle = self.end_angle - self.start_angle
        angle_per_frame = total_angle / (num_frames - 1) if num_frames > 1 else 0
        
        # 设置初始视角或旋转
        if self.rotation_mode == 'camera':
            if axis == 'horizontal':
                ctr.rotate(self.start_angle, 0, 0)
            else:
                ctr.rotate(0, self.start_angle, 0)
        else:  # object rotation
            if axis == 'horizontal':
                pcd.rotate(pcd.get_rotation_matrix_from_xyz([0, np.radians(self.start_angle), 0]))
            else:
                pcd.rotate(pcd.get_rotation_matrix_from_xyz([0, np.radians(self.start_angle), 0]))
            vis.update_geometry(pcd)
        
        # 生成帧
        for i in range(num_frames):
            logger.debug(f"生成第 {i+1}/{num_frames} 帧")
            current_angle = self.start_angle + (i * angle_per_frame)
            
            if self.rotation_mode == 'camera':
                # 更新视角
                if i < num_frames - 1:
                    if axis == 'horizontal':
                        ctr.rotate(angle_per_frame, 0, 0)
                    else:
                        ctr.rotate(0, angle_per_frame, 0)
            else:  # object rotation
                # 更新点云旋转
                if i < num_frames - 1:
                    if axis == 'horizontal':
                        pcd.rotate(pcd.get_rotation_matrix_from_xyz([0, np.radians(angle_per_frame), 0]))
                    else:
                        pcd.rotate(pcd.get_rotation_matrix_from_xyz([0, np.radians(angle_per_frame), 0]))
                    vis.update_geometry(pcd)
            
            # 渲染当前帧
            vis.poll_events()
            vis.update_renderer()
            
            # 捕获当前帧
            image = vis.capture_screen_float_buffer(do_render=True)
            image = np.asarray(image)
            image = (image * 255).astype(np.uint8)
            frames.append(image)
        
        # 清理
        vis.destroy_window()
        logger.info("帧序列生成完成")
        
        return frames
    
    def generate_gif(self, output_path, num_frames=30, duration=100, loop=0, axis='horizontal'):
        """生成GIF动画
        
        Args:
            output_path (str): 输出GIF文件路径
            num_frames (int, optional): 帧数量
            duration (int, optional): 每帧持续时间(ms)
            loop (int, optional): 循环次数，0表示无限循环
            axis (str, optional): 旋转轴，'horizontal'或'vertical'
        """
        start_time = time.time()
        logger.info(f"开始生成GIF: {output_path}")
        
        frames = self._generate_frames(num_frames, axis)
        
        # 转换为PIL图像
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # 保存GIF
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=loop
        )
        
        total_time = time.time() - start_time
        logger.success(f"GIF生成完成: {output_path}")
        logger.info(f"总帧数: {num_frames}, 每帧时长: {duration}ms, 总时长: {(num_frames * duration)/1000:.2f}秒")
        logger.info(f"角度范围: 从{self.start_angle}°到{self.end_angle}°, 每帧旋转: {(self.end_angle - self.start_angle)/(num_frames-1 if num_frames > 1 else 1):.2f}°")
        logger.info(f"总耗时: {total_time:.2f}秒")
        
    def generate_video(self, output_path, fps=30, duration=None, num_frames=None, axis='horizontal'):
        """生成视频
        
        Args:
            output_path (str): 输出视频文件路径
            fps (int, optional): 视频帧率
            duration (float, optional): 视频时长(秒)，与num_frames二选一
            num_frames (int, optional): 指定帧数量，与duration二选一
            axis (str, optional): 旋转轴，'horizontal'或'vertical'
        """
        start_time = time.time()
        logger.info(f"开始生成视频: {output_path}")
        
        # 计算帧数
        if num_frames is None and duration is not None:
            num_frames = int(fps * duration)
        elif num_frames is None:
            num_frames = 90  # 默认值
            
        frames = self._generate_frames(num_frames, axis)
        
        # 创建视频写入器
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 写入帧
        for frame in frames:
            # 转换为BGR格式
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        # 释放资源
        out.release()
        
        total_time = time.time() - start_time
        logger.success(f"视频生成完成: {output_path}")
        logger.info(f"总帧数: {num_frames}, 帧率: {fps}fps, 总时长: {num_frames/fps:.2f}秒")
        logger.info(f"角度范围: 从{self.start_angle}°到{self.end_angle}°, 每帧旋转: {(self.end_angle - self.start_angle)/(num_frames-1 if num_frames > 1 else 1):.2f}°")
        logger.info(f"总耗时: {total_time:.2f}秒")


def generate_point_cloud_gif(ply_path, output_path, num_frames=30, duration=100, **kwargs):
    """生成点云GIF的便捷函数
    
    Args:
        ply_path (str): 点云文件路径
        output_path (str): 输出GIF文件路径
        num_frames (int, optional): 帧数量
        duration (int, optional): 每帧持续时间(ms)
        **kwargs: 其他参数传递给PointCloudAnimator
            - start_angle (float): 起始角度
            - end_angle (float): 结束角度
            - zoom (float): 缩放级别
            - point_size (float): 点云点大小
            - width (int): 帧宽度
            - height (int): 帧高度
            - loop (int): 循环次数，0表示无限循环
            - axis (str): 旋转轴，'horizontal'或'vertical'
            - rotation_mode (str): 旋转模式，'camera'或'object'
    """
    logger.info(f"开始生成点云GIF: {ply_path} -> {output_path}")
    
    animator = PointCloudAnimator(ply_path)
    
    # 设置自定义参数
    if 'start_angle' in kwargs and 'end_angle' in kwargs:
        animator.set_angle_range(kwargs['start_angle'], kwargs['end_angle'])
    
    # 视角参数
    view_params = {}
    if 'zoom' in kwargs:
        view_params['zoom'] = kwargs['zoom']
    if 'point_size' in kwargs:
        view_params['point_size'] = kwargs['point_size']
    if view_params:
        animator.set_view_params(**view_params)
    
    # 设置帧大小
    if 'width' in kwargs and 'height' in kwargs:
        animator.set_frame_size(kwargs['width'], kwargs['height'])
        
    # 设置旋转模式
    if 'rotation_mode' in kwargs:
        animator.set_rotation_mode(kwargs['rotation_mode'])
    
    animator.generate_gif(output_path, num_frames, duration, 
                         loop=kwargs.get('loop', 0),
                         axis=kwargs.get('axis', 'horizontal'))


def generate_point_cloud_video(ply_path, output_path, fps=30, duration=10, **kwargs):
    """生成点云视频的便捷函数
    
    Args:
        ply_path (str): 点云文件路径
        output_path (str): 输出视频文件路径
        fps (int, optional): 视频帧率
        duration (float, optional): 视频时长(秒)
        **kwargs: 其他参数传递给PointCloudAnimator
            - start_angle (float): 起始角度
            - end_angle (float): 结束角度
            - zoom (float): 缩放级别
            - point_size (float): 点云点大小
            - width (int): 帧宽度
            - height (int): 帧高度
            - num_frames (int): 指定帧数量
            - axis (str): 旋转轴，'horizontal'或'vertical'
            - rotation_mode (str): 旋转模式，'camera'或'object'
    """
    logger.info(f"开始生成点云视频: {ply_path} -> {output_path}")
    
    animator = PointCloudAnimator(ply_path)
    
    # 设置自定义参数
    if 'start_angle' in kwargs and 'end_angle' in kwargs:
        animator.set_angle_range(kwargs['start_angle'], kwargs['end_angle'])
    
    # 视角参数
    view_params = {}
    if 'zoom' in kwargs:
        view_params['zoom'] = kwargs['zoom']
    if 'point_size' in kwargs:
        view_params['point_size'] = kwargs['point_size']
    if view_params:
        animator.set_view_params(**view_params)
    
    # 设置帧大小
    if 'width' in kwargs and 'height' in kwargs:
        animator.set_frame_size(kwargs['width'], kwargs['height'])
        
    # 设置旋转模式
    if 'rotation_mode' in kwargs:
        animator.set_rotation_mode(kwargs['rotation_mode'])
    
    animator.generate_video(output_path, fps, duration, 
                           num_frames=kwargs.get('num_frames'),
                           axis=kwargs.get('axis', 'horizontal'))


if __name__ == "__main__":
    # 配置日志
    # logger.add("pcd_animation_{time}.log", rotation="500 MB")
    
    ply_path = r"C:\Users\Andy\DCIM\demo_video.ply"
    zoom = 0.8
    point_size = 0.8
    width, height = 1920, 1080
    
    # # 示例1：使用便捷函数生成GIF (相机旋转模式)
    # generate_point_cloud_gif(
    #     ply_path, 
    #     "output_camera_rotation.gif", 
    #     num_frames=60,      # 帧数
    #     duration=50,        # 每帧持续50ms
    #     start_angle=-90,    # 起始角度
    #     end_angle=90,       # 结束角度
    #     zoom=zoom,          # 缩放级别
    #     point_size=point_size,  # 点大小
    #     width=width,        # 帧宽度
    #     height=height,      # 帧高度
    #     rotation_mode='camera'  # 相机旋转模式
    # )
    
    # 示例2：使用便捷函数生成GIF (物体旋转模式)
    generate_point_cloud_gif(
        ply_path, 
        "output_object_rotation.gif", 
        num_frames=60,      # 帧数
        duration=100,        # 每帧持续50ms
        start_angle=0,    # 起始角度
        end_angle=360,       # 结束角度
        zoom=zoom,          # 缩放级别
        point_size=point_size,  # 点大小
        width=width,        # 帧宽度
        height=height,      # 帧高度
        rotation_mode='object'  # 物体旋转模式
    )
    
    # # 示例3：使用便捷函数生成视频 (相机旋转模式)
    # generate_point_cloud_video(
    #     ply_path, 
    #     "output_camera_rotation.mp4", 
    #     fps=30,             # 帧率
    #     duration=3,         # 总时长3秒
    #     start_angle=-90,    # 起始角度
    #     end_angle=90,       # 结束角度
    #     zoom=zoom,          # 缩放级别
    #     point_size=point_size,  # 点大小
    #     width=width,        # 帧宽度
    #     height=height,      # 帧高度
    #     rotation_mode='camera'  # 相机旋转模式
    # )
    
    # 示例4：使用便捷函数生成视频 (物体旋转模式)
    generate_point_cloud_video(
        ply_path, 
        "output_object_rotation.mp4", 
        fps=30,             # 帧率
        duration=15,         # 总时长3秒
        start_angle=0,    # 起始角度
        end_angle=360,       # 结束角度
        zoom=zoom,          # 缩放级别
        point_size=point_size,  # 点大小
        width=width,        # 帧宽度
        height=height,      # 帧高度
        rotation_mode='object'  # 物体旋转模式
    )
        