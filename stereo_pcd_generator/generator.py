from .stereo_client import generate_pcd_from_rect_stereo_pair
from .prep.adjust_y import adjust_image_pair, adjust_image_coordinates
from .prep.rectify import rectify_images_pair
from .prep.model import CameraModel
import os
import dotenv
from pathlib import Path

dotenv.load_dotenv()
SERVER_URL = os.getenv("server_url")


def generae_pcd_raw_images(raw_left_path, raw_right_path, camera_model_path, output_dir, scale=0.35):
    output_dir = Path(output_dir)

    # 矫正图像对
    left, right = rectify_images_pair(
        camera_model_path=camera_model_path,
        raw_left_path=raw_left_path,
        raw_right_path=raw_right_path,
        output_dir=output_dir
    )

    # 自动调整图像对
    # left, right = adjust_image_pair(
    #     left_path=left,
    #     right_path=right,
    #     output_dir=Path(output_dir) / "adjusted"
    # )
    # 手动调整图像对
    # (output_dir / "adjusted").mkdir(parents=True, exist_ok=True)
    # adjust_image_coordinates(right, output_dir / "adjusted" / right.name, 0, -15)
    # shutil.copy(left, output_dir / "adjusted" / left.name)
    # left = output_dir / "adjusted" / left.name
    # right = output_dir / "adjusted" / right.name


    # 生成相机内参文件
    cm = CameraModel.load_model(camera_model_path)
    K_file_path = Path(output_dir) / "K.txt"
    with open(K_file_path, 'w') as f:
        f.write(cm.generate_K_from_cm())

    # 生成点云
    generate_pcd_from_rect_stereo_pair(
        left_image_path=left,
        right_image_path=right,
        intrinsic_file=K_file_path,
        output_dir=output_dir,
        server_url=SERVER_URL,
        scale=scale
    )


def generate_pcd_dir(raw_dir, camera_model_path, output_dir, scale=0.35):
    raw_dir = Path(raw_dir)
    for left_path in raw_dir.glob("A_*.jpg"):
        right_path = raw_dir / left_path.name.replace("A_", "D_")
        assert right_path.exists(), f"right image {right_path} not found"
        this_output_dir = Path(output_dir) / left_path.stem
        generae_pcd_raw_images(left_path, right_path, camera_model_path, this_output_dir, scale)


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

