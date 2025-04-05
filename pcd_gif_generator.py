import open3d as o3d
import numpy as np
from PIL import Image
import os
import tempfile
import matplotlib.pyplot as plt
import cv2


class PointCloudAnimator:
    """点云动画生成器类
    
    用于生成点云旋转的GIF或视频，提供统一的接口和参数控制。
    """
    
    def __init__(self, ply_path):
        """初始化点云动画生成器
        
        Args:
            ply_path (str): 点云文件路径
        """
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
    
    def set_frame_size(self, width, height):
        """设置输出帧的大小
        
        Args:
            width (int): 帧宽度
            height (int): 帧高度
        """
        self.width = width
        self.height = height
        
    def set_angle_range(self, start_angle, end_angle):
        """设置旋转角度范围
        
        Args:
            start_angle (float): 开始角度
            end_angle (float): 结束角度
        """
        self.start_angle = start_angle
        self.end_angle = end_angle
        
    def _generate_frames(self, num_frames, axis='horizontal'):
        """生成点云旋转的帧序列
        
        Args:
            num_frames (int): 帧数量
            axis (str, optional): 旋转轴，'horizontal'或'vertical'
        
        Returns:
            list: 帧列表
        """
        frames = []
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=self.width, height=self.height)
        vis.add_geometry(self.pcd)
        
        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.point_size = self.point_size  # 设置点大小
        
        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_front(self.front)
        ctr.set_lookat(self.lookat)
        ctr.set_up(self.up)
        ctr.set_zoom(self.zoom_level)
        
        # 计算角度范围和每帧旋转角度
        total_angle = self.end_angle - self.start_angle
        angle_per_frame = total_angle / (num_frames - 1) if num_frames > 1 else 0
        
        # 设置初始视角
        if axis == 'horizontal':
            ctr.rotate(self.start_angle, 0, 0)
        else:
            ctr.rotate(0, self.start_angle, 0)
        
        # 生成帧
        for i in range(num_frames):
            # 计算当前角度
            current_angle = self.start_angle + (i * angle_per_frame)
            
            # 渲染当前帧
            vis.poll_events()
            vis.update_renderer()
            
            # 捕获当前帧
            image = vis.capture_screen_float_buffer(do_render=True)
            image = np.asarray(image)
            image = (image * 255).astype(np.uint8)
            frames.append(image)
            
            # 更新视角（除了最后一帧）
            if i < num_frames - 1:
                if axis == 'horizontal':
                    ctr.rotate(angle_per_frame, 0, 0)
                else:
                    ctr.rotate(0, angle_per_frame, 0)
        
        # 清理
        vis.destroy_window()
        
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
        
        print(f"GIF已保存到: {output_path}")
        print(f"总帧数: {num_frames}, 每帧时长: {duration}ms, 总时长: {(num_frames * duration)/1000:.2f}秒")
        print(f"角度范围: 从{self.start_angle}°到{self.end_angle}°, 每帧旋转: {(self.end_angle - self.start_angle)/(num_frames-1 if num_frames > 1 else 1):.2f}°")
        
    def generate_video(self, output_path, fps=30, duration=None, num_frames=None, axis='horizontal'):
        """生成视频
        
        Args:
            output_path (str): 输出视频文件路径
            fps (int, optional): 视频帧率
            duration (float, optional): 视频时长(秒)，与num_frames二选一
            num_frames (int, optional): 指定帧数量，与duration二选一
            axis (str, optional): 旋转轴，'horizontal'或'vertical'
        """
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
        
        print(f"视频已保存到: {output_path}")
        print(f"总帧数: {num_frames}, 帧率: {fps}fps, 总时长: {num_frames/fps:.2f}秒")
        print(f"角度范围: 从{self.start_angle}°到{self.end_angle}°, 每帧旋转: {(self.end_angle - self.start_angle)/(num_frames-1 if num_frames > 1 else 1):.2f}°")


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
    """
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
    """
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
    
    animator.generate_video(output_path, fps, duration, 
                           num_frames=kwargs.get('num_frames'),
                           axis=kwargs.get('axis', 'horizontal'))


if __name__ == "__main__":
    ply_path = "/Users/andyliu/workspace/0327_1/105508922583/cloud.ply"
    zoom = 0.08
    point_size = 0.6
    width, hight = 1280, 960
    
    # 示例1：使用便捷函数生成GIF (带有小点大小)
    generate_point_cloud_gif(
        ply_path, 
        "output_basic.gif", 
        num_frames=60,      # 帧数
        duration=50,        # 每帧持续50ms
        start_angle=-90,    # 起始角度
        end_angle=90,       # 结束角度
        zoom=zoom,           # 缩放级别
        point_size=point_size,     # 设置更小的点大小
        width=width,         # 设置帧宽度为1280
        height=hight          # 设置帧高度为720
    )
    
    # 示例2：使用便捷函数生成视频 (带有小点大小)
    generate_point_cloud_video(
        ply_path, 
        "output_basic.mp4", 
        fps=30,             # 帧率
        duration=3,         # 总时长3秒
        start_angle=-90,    # 起始角度
        end_angle=90,       # 结束角度
        zoom=zoom,           # 缩放级别
        point_size=point_size,     # 设置更小的点大小
        width=width,         # 设置帧宽度为1280
        height=hight          # 设置帧高度为720
    )
    
    # # 示例3：使用类接口进行更多自定义
    # animator = PointCloudAnimator(ply_path)
    
    # # 自定义视角
    # animator.set_view_params(
    #     front=[0, 0, -1],
    #     lookat=[0, 0, 1],
    #     up=[0, -1, 0],
    #     zoom=0.15,
    #     point_size=0.3    # 设置非常小的点大小
    # )
    
    # # 设置旋转范围
    # animator.set_angle_range(-120, 120)
    
    # # 生成垂直方向旋转的GIF
    # animator.generate_gif(
    #     "output_vertical.gif",
    #     num_frames=45,
    #     duration=80,
    #     axis='vertical'  # 垂直方向旋转
    # )
    
    # # 生成水平方向旋转的视频
    # animator.generate_video(
    #     "output_custom.mp4",
    #     fps=24,
    #     num_frames=72  # 指定帧数而不是持续时间
    # )
        