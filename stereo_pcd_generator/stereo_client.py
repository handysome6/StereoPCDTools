
import requests
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import argparse
import time
from loguru import logger
import imageio
import json
from .Utils import depth2xyzmap, toOpen3dCloud, vis_disparity



def generate_pcd_from_rect_stereo_pair(left_image_path, right_image_path, intrinsic_file, server_url="http://localhost:8000", output_dir=None, scale=1.0):
    """
    处理矫正后的图像对，调用立体视觉服务API，并在本地生成点云
    
    Args:
        left_image_path: 左图像路径
        right_image_path: 右图像路径
        intrinsic_file: 相机内参文件路径
        server_url: 服务器URL
        output_dir: 输出目录路径，如果为None则使用左图像所在目录下的results文件夹
        scale: 图像缩放比例，默认为1.0
    """
    start_time = time.time()
    
    # 确保输出目录存在
    if output_dir is None:
        output_dir = Path(left_image_path).parent / "results"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    
    logger.info(f"开始处理立体图像对: {left_image_path} 和 {right_image_path}")
    logger.info(f"参数设置 - 服务器URL: {server_url}, 输出目录: {output_dir}, 缩放比例: {scale}")
    
    # 记录参数字典
    params = {
        "left_image_path": str(left_image_path),
        "right_image_path": str(right_image_path),
        "intrinsic_file": str(intrinsic_file),
        "server_url": server_url,
        "output_dir": str(output_dir),
        "scale": scale,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    logger.debug(f"输入参数详情: {json.dumps(params, indent=4)}")
    
    endpoint = f"{server_url}/process_stereo_pair/"
    
    # 第1步：读取并缩放图像
    logger.info("第1步: 读取并缩放图像...")
    img_left = cv2.imread(left_image_path)
    img_right = cv2.imread(right_image_path)
    
    if img_left is None or img_right is None:
        logger.error(f"无法读取图像: {left_image_path if img_left is None else right_image_path}")
        return None
    
    logger.info(f"原始图像尺寸 - 左: {img_left.shape}, 右: {img_right.shape}")
    
    if scale != 1.0:
        width = int(img_left.shape[1] * scale)
        height = int(img_left.shape[0] * scale)
        logger.info(f"缩放图像至: {width}x{height}")
        img_left_scaled = cv2.resize(img_left, (width, height), interpolation=cv2.INTER_LINEAR)
        img_right_scaled = cv2.resize(img_right, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        logger.info("无需缩放图像")
        img_left_scaled = img_left
        img_right_scaled = img_right
    
    # 检查图像分辨率是否过大
    image_mp = (img_left_scaled.shape[1] * img_left_scaled.shape[0]) / 1000000
    if image_mp > 200:
        logger.warning(f"警告: 图像分辨率过大 ({image_mp:.1f}MP)，超过200MP！处理可能会很慢或失败。")
        logger.warning(f"建议使用较小的缩放比例或较小的原始图像。")
    else:
        logger.info(f"图像分辨率: {image_mp:.1f}MP")
    
    # 保存缩放后的图像
    scaled_left_path = output_dir / "img0.png"
    scaled_right_path = output_dir / "img1.png"
    cv2.imwrite(str(scaled_left_path), img_left_scaled)
    cv2.imwrite(str(scaled_right_path), img_right_scaled)
    logger.info(f"已保存缩放后的图像到: {scaled_left_path} 和 {scaled_right_path}")
    
    # 第2步：准备API请求
    logger.info(f"第2步: 准备API请求到 {endpoint}...")
    files = {
        'left_image': ('left.png', open(str(scaled_left_path), 'rb'), 'image/png'),
        'right_image': ('right.png', open(str(scaled_right_path), 'rb'), 'image/png'),
    }
    
    # 第3步：调用API获取视差图
    logger.info("第3步: 发送请求并获取视差图...")
    api_start_time = time.time()
    response = requests.post(endpoint, files=files)
    api_time = time.time() - api_start_time
    
    if response.status_code != 200:
        logger.error(f"API请求失败: {response.status_code}, {response.text}")
        return None
    
    logger.success(f"API请求成功, 耗时: {api_time:.2f}秒")
    
    # 第4步：保存视差图到输出目录
    logger.info("第4步: 保存视差图...")
    disp_file = output_dir / "disp.npy"
    with open(disp_file, 'wb') as f:
        f.write(response.content)
        
    logger.info(f"已保存视差图到 {disp_file}")
    
    # 第5步：加载视差图和原始图像，生成点云
    logger.info("第5步: 加载视差图和原始图像...")
    disp = np.load(disp_file)
    img0 = cv2.imread(str(scaled_left_path))
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    
    logger.info(f"视差图尺寸: {disp.shape}, 视差范围: {np.min(disp[disp > 0]):.2f} 至 {np.max(disp):.2f}")
    
    logger.info("第6步: 读取相机内参...")
    # 第6步：读取相机内参
    try:
        with open(intrinsic_file, 'r') as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
            baseline = float(lines[1])
        logger.info(f"相机内参K:\n{K}")
        logger.info(f"基线长度: {baseline}")
    except Exception as e:
        logger.error(f"读取相机内参文件失败: {e}")
        return None
    
    # 应用缩放到内参矩阵
    K[:2] *= scale
    logger.info(f"应用缩放后的内参K:\n{K}")
    
    # 第7步：处理视差图（过滤不可见点）
    logger.info("第7步: 处理视差图（过滤不可见点）...")
    yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    us_right = xx - disp
    invalid = us_right < 0
    disp[invalid] = np.inf
    logger.info(f"过滤了 {np.sum(invalid)} 个不可见点 ({np.sum(invalid)/disp.size*100:.2f}%)")
    
    # 第8步：计算深度图和点云
    logger.info("第8步: 计算深度图和点云...")
    depth_start_time = time.time()
    depth = K[0,0] * baseline / disp
    
    valid_depth = depth[~np.isinf(depth)]
    if len(valid_depth) > 0:
        logger.info(f"深度范围: {np.min(valid_depth):.3f}m 至 {np.max(valid_depth):.3f}m")
    
    np.save(f'{output_dir}/depth_meter.npy', depth)
    logger.info(f"已保存深度图到 {output_dir}/depth_meter.npy")
    
    logger.info("生成点云...")
    xyz_map = depth2xyzmap(depth, K)
    pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0.reshape(-1,3))
    
    # 第9步：过滤无效点
    logger.info("第9步: 过滤无效点...")
    invalid_mask = ~((np.asarray(pcd.points)[:,2]>0) & 
                    (np.asarray(pcd.points)[:,2]<=10))  # z_far=10米
    
    points = np.asarray(pcd.points)
    points_count_before = len(points)
    points[invalid_mask] = np.array([0,0,0])
    pcd.points = o3d.utility.Vector3dVector(points)
    logger.info(f"过滤了 {np.sum(invalid_mask)} 个无效点 ({np.sum(invalid_mask)/points_count_before*100:.2f}%)")
    
    # 第10步：保存点云
    logger.info("第10步: 保存点云...")
    cloud_path = f'{output_dir}/cloud.ply'
    o3d.io.write_point_cloud(cloud_path, pcd)
    logger.success(f"已保存点云到 {cloud_path}")
    
    # 第11步：保存可视化结果
    logger.info("第11步: 保存可视化结果...")
    vis = vis_disparity(disp)
    vis = np.concatenate([img0, vis], axis=1)
    imageio.imwrite(f'{output_dir}/vis.png', vis)
    logger.info(f"已保存可视化结果到 {output_dir}/vis.png")


    total_time = time.time() - start_time
    logger.success(f"处理完成! 总耗时: {total_time:.2f}秒")
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="立体视觉客户端示例")
    parser.add_argument("--left", required=True, help="左图像路径")
    parser.add_argument("--right", required=True, help="右图像路径")
    parser.add_argument("--intrinsic", required=True, help="相机内参文件路径")
    parser.add_argument("--server", default="http://localhost:8000", help="服务器URL")
    parser.add_argument("--output", help="输出目录路径，如果未指定则使用左图像所在目录下的results文件夹")
    parser.add_argument("--scale", type=float, default=1.0, help="图像缩放比例，默认为1.0")
    
    args = parser.parse_args()
    
    generate_pcd_from_rect_stereo_pair(
        args.left,
        args.right,
        args.intrinsic,
        args.server,
        args.output,
        args.scale
    ) 