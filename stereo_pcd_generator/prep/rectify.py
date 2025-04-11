import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from loguru import logger

from .model import CameraModel


class StereoRectify():
    def __init__(self, camera) -> None:
        """
        Construct rectifier.
        params:
            camera: calibrated CameraModel object
        """
        self.camera = camera
        self.width = self.camera.image_size[0]
        self.height = self.camera.image_size[1]

        # init Q and maps
        self.Q = None
        self.leftMapX = self.leftMapY = None
        self.rightMapX = self.rightMapY = None

    def rectify_camera(self, roi_ratio = 0, new_image_ratio = 1):
        """
        Switch to call diff rectify method 
        roi_ratio: Determine how much black edge is preserved
                    roi_ratio = 0: None black area is preserved
                    roi_ratio = 1: all black area is preserved
        new_image_ratio: Determine the new imagesize 
        """
        # rectify parameters
        roi_ratio = roi_ratio
        newImageSize = np.array(self.camera.image_size) * new_image_ratio
        newImageSize = np.array(newImageSize, dtype=np.int32)

        if not self.camera.is_calibrated():
            print("No calib_data found. \nPlease calib camera before rectify")
            exit()
        if self.camera.is_fisheye:
            logger.error("Fisheye camera not supported")
        else:
            self._stereo_rectify_vanilla(roi_ratio, newImageSize)

    def is_rectified(self):
        """Check if this rectifier is rectified"""
        if  self.Q is None or\
            self.leftMapX  is None or self.leftMapY  is None or \
            self.rightMapX is None or self.rightMapY is None:
            return False
        else:
            return True

    def rectify_image(self, img_left=None, img_right=None, sbs_img=None):
        """ 
        Rectify single sbs image using maps
        img_left: left img
        img_right: right img
        sbs_img: single sbs image
        """
        # ensure rectify parameters exist
        if not self.is_rectified():
            print("Rectifier not rectified, rectifying first...")
            self.rectify_camera()
        
        if img_left is not None and img_right is not None:
            img_left = img_left
            img_right = img_right
        elif sbs_img is not None:
            # split
            img_left = sbs_img [:,          0:   self.width]
            img_right = sbs_img [:, self.width: 2*self.width]
        else:
            raise Exception("At least one pair of img should be provided. "
                "Either sbs_img or img_left/right.")
        
        # rectify the given image
        left_rect = cv2.remap(
            img_left, self.leftMapX, self.leftMapY, 
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        right_rect = cv2.remap(
            img_right, self.rightMapX, self.rightMapY, 
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        return left_rect, right_rect

    def _stereo_rectify_vanilla(self, alpha, newImageSize):  
        """
        Compute rectify map in Vanilla approach
        """
        logger.debug("Vanilla rectifying...")
        # calculate rectify matrices using calibration param
        R1, R2, P1, P2, Q, ROI1, ROI2 = \
            cv2.stereoRectify(
                self.camera.cm1, self.camera.cd1, 
                self.camera.cm2, self.camera.cd2, 
                self.camera.image_size, 
                self.camera.R, self.camera.T,
                alpha=alpha,
                newImageSize=newImageSize,
            )
        
        self.Q = Q
        # create map for rectification
        self.leftMapX, self.leftMapY  = cv2.initUndistortRectifyMap(
            self.camera.cm1, self.camera.cd1, R1, P1, newImageSize, cv2.CV_16SC2
        )
        self.rightMapX, self.rightMapY= cv2.initUndistortRectifyMap(
            self.camera.cm2, self.camera.cd2, R2, P2, newImageSize, cv2.CV_16SC2
        )
        logger.debug("Calculate map done.")
        logger.debug(f"rectification map shape: {self.rightMapX.shape}")


def rectify_images_pair(camera_model_path, raw_left_path, raw_right_path, output_dir):
    """
    处理单张双目图像的矫正
    
    Args:
        camera_model_path (str or Path): 相机模型文件的路径
        raw_left_path (str or Path): 原始左图像的路径
        raw_right_path (str or Path): 原始右图像的路径
        output_dir (str or Path): 输出路径目录
    Returns:
        left_save_path：矫正后的左图像路径
        right_save_path：矫正后的右图像路径
    """
    # 确保输入路径是Path对象
    camera_model_path = Path(camera_model_path)
    raw_left_path = Path(raw_left_path)
    raw_right_path = Path(raw_right_path)

    logger.info(f"矫正图像: {raw_left_path} - {raw_right_path.name}")

    # 加载相机模型并创建矫正器
    camera = CameraModel.load_model(camera_model_path)
    rectifier = StereoRectify(camera)
    rectifier.rectify_camera()

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 读取图像
    left_img = cv2.imread(str(raw_left_path))
    right_img = cv2.imread(str(raw_right_path))

    # 验证图像
    assert left_img is not None, f"无法读取左图像: {raw_left_path}"
    assert right_img is not None, f"无法读取右图像: {raw_right_path}"
    assert left_img.shape == right_img.shape, f"左右图像尺寸不一致: {raw_left_path} {left_img.shape} != {raw_right_path} {right_img.shape}"

    # 矫正图像
    left_rect, right_rect = rectifier.rectify_image(left_img, right_img)

    # 保存矫正后的图像
    left_save_path = output_dir / raw_left_path.name
    right_save_path = output_dir / raw_right_path.name
    cv2.imwrite(str(left_save_path), left_rect)
    cv2.imwrite(str(right_save_path), right_rect)

    logger.success(f"矫正图像完成: {left_save_path} - {right_save_path.name}")

    return left_save_path, right_save_path


def rectify_images_pairs_dir(camera_model_path, raw_img_dir, output_dir):
    """
    处理双目相机图像的矫正
    
    Args:
        camera_model_path (str or Path): 相机模型文件的路径
        raw_img_dir (str or Path): 原始图像所在的目录
        output_dir (str or Path): 输出目录路径
    """
    # 确保输入路径是Path对象
    camera_model_path = Path(camera_model_path)
    raw_img_dir = Path(raw_img_dir)
    
    # 加载相机模型并创建矫正器
    camera = CameraModel.load_model(camera_model_path)
    rectifier = StereoRectify(camera)
    rectifier.rectify_camera()

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 获取所有左侧图像路径
    left_images = list(raw_img_dir.glob("A_*.jpg"))
    
    # 处理所有图像对，添加进度条
    for left_path in tqdm(left_images, desc="正在矫正图像", ncols=80):
        right_path = left_path.with_name(left_path.name.replace("A_", "D_"))
        
        # 读取图像
        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))

        # 验证图像
        assert left_img is not None, f"无法读取左图像: {left_path}"
        assert right_img is not None, f"无法读取右图像: {right_path}"
        assert left_img.shape == right_img.shape, f"左右图像尺寸不一致: {left_path} {left_img.shape} != {right_path} {right_img.shape}"

        # 矫正图像
        left_rect, right_rect = rectifier.rectify_image(left_img, right_img)

        # 保存矫正后的图像
        left_save_path = output_dir / left_path.name
        right_save_path = output_dir / right_path.name
        cv2.imwrite(str(left_save_path), left_rect)
        cv2.imwrite(str(right_save_path), right_rect)


if __name__ == "__main__":
    # model_path = r"/home/andy/DCIM/0327_handheld_nuclear/camera_model.json"
    # raw_dir = r"/home/andy/DCIM/0327_handheld_nuclear/"
    # output_dir = r"/home/andy/DCIM/0327_handheld_nuclear/rectified"
    # rectify_images_pairs_dir(model_path, raw_dir, output_dir)
    
    model_path = r"C:\Users\Andy\DCIM\camera_model.json"
    raw_left_path = r"C:\Users\Andy\DCIM\A_17170356423084390.jpg"
    raw_right_path = r"C:\Users\Andy\DCIM\D_17170356423084390.jpg"
    output_dir = r"C:\Users\Andy\DCIM\test_rectified"
    rectify_images_pair(model_path, raw_left_path, raw_right_path, output_dir)
