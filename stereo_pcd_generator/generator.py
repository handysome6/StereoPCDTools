from .stereo_client import generate_pcd_from_rect_stereo_pair
from .prep.adjust_y import AdjustImagePairAuto, adjust_image_coordinates
from .prep.rectify import rectify_images_pair
from .prep.model import CameraModel
import os
import dotenv
from pathlib import Path
from .prep.y_diff_analyzer import ImagePairAnalyzer
from .Utils import setup_logger
from loguru import logger

dotenv.load_dotenv(override=True)
SERVER_URL = os.getenv("server_url")


def generae_pcd_raw_images(
        raw_left_path, 
        raw_right_path, 
        camera_model_path, 
        output_dir, 
        scale=0.35,
        adjust_image=False
    ):
    """
    从原始立体图像对生成点云

    该函数处理一对原始立体图像,进行矫正、调整和点云生成。

    Args:
        raw_left_path (str): 左相机原始图像路径
        raw_right_path (str): 右相机原始图像路径  
        camera_model_path (str): 相机模型文件路径,包含相机内外参数
        output_dir (str): 输出目录路径,用于保存中间结果和最终点云
        scale (float, optional): 点云生成时的缩放比例,默认为0.35
        adjust_image (bool, optional): 是否进行图像对自动调整的Flag,默认为False

    处理流程:
        1. 根据相机模型对原始图像对进行矫正
        2. 如果启用adjust_image,分析并调整图像对的y轴差异
        3. 从相机模型生成内参矩阵文件
        4. 调用点云生成服务生成最终点云

    输出:
        - 在output_dir下生成矫正后的图像对
        - 如果进行了调整,在output_dir/adjusted下生成调整后的图像对
        - 在output_dir下生成相机内参文件K.txt
        - 在output_dir下生成最终点云文件
    """
    output_dir = Path(output_dir)

    # 设置日志记录器
    setup_logger(output_dir)

    # 矫正图像对
    left_path, right_path = rectify_images_pair(
        camera_model_path=camera_model_path,
        raw_left_path=raw_left_path,
        raw_right_path=raw_right_path,
        output_dir=output_dir
    )

    # 自动调整图像对
    adjuster = AdjustImagePairAuto(
        left_path=left_path,
        right_path=right_path,
        output_dir=Path(output_dir) / "adjusted"
    )
    y_diff, needs_adjustment = adjuster.analyze_y_diff()
    if adjust_image and needs_adjustment:
        left_path, right_path = adjuster.adjust_images(y_diff)
    else:
        left_path, right_path = left_path, right_path
    

    # 生成相机内参文件
    cm = CameraModel.load_model(camera_model_path)
    K_file_path = Path(output_dir) / "K.txt"
    with open(K_file_path, 'w') as f:
        f.write(cm.generate_K_from_cm())

    # 生成点云
    generate_pcd_from_rect_stereo_pair(
        left_image_path=left_path,
        right_image_path=right_path,
        intrinsic_file=K_file_path,
        output_dir=output_dir,
        server_url=SERVER_URL,
        scale=scale
    )


def generate_pcd_dir(raw_dir, camera_model_path, output_dir, scale=0.35, adjust_image=False):
    raw_dir = Path(raw_dir)
    # determine the postfix of images (file format)
    suffix = list(raw_dir.glob("A_*"))[0].suffix
    logger.info(f"detected image suffix: {suffix}")
    for left_path in raw_dir.glob(f"A_*{suffix}"):
        right_path = raw_dir / left_path.name.replace(f"A_", f"D_")
        assert right_path.exists(), f"right image {right_path} not found"
        this_output_dir = Path(output_dir) / left_path.stem
        generae_pcd_raw_images(left_path, right_path, camera_model_path, this_output_dir, scale, adjust_image)


if __name__ == "__main__":
    # generae_pcd_raw_images(
    #     raw_left_path=r"C:\Users\Andy\DCIM\0411_rebar_data\raw\A_5796865372.jpg",
    #     raw_right_path=r"C:\Users\Andy\DCIM\0411_rebar_data\raw\D_5796865372.jpg",
    #     camera_model_path=r"C:\Users\Andy\DCIM\0411_rebar_data\raw\camera_model.json",
    #     output_dir=r"C:\Users\Andy\DCIM\0411_rebar_data\pcd_result\A_5796865372",
    #     scale=0.35
    # )
    generate_pcd_dir(
        raw_dir='/Users/andyliu/Downloads/展台重建',
        camera_model_path='/Users/andyliu/Downloads/展台重建/camera_model.json',
        output_dir='/Users/andyliu/Downloads/展台重建/pcd_result',
        scale=0.35
    )

