# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import time,logging,importlib
import open3d as o3d
import cv2
import numpy as np
from loguru import logger
from pathlib import Path


def setup_logger(output_dir=None):
    """配置loguru logger"""
    logger.remove()  # 移除默认处理器
    
    # 添加控制台输出
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 如果提供了输出目录，添加文件输出
    if output_dir:
        log_file = Path(output_dir) / "stereo_process.log"
        
        # 检查并删除现有的日志文件
        if log_file.exists():
            try:
                log_file.unlink()  # 删除现有日志文件
                logger.info(f"已删除旧的日志文件")
            except Exception as e:
                logger.warning(f"无法删除旧的日志文件: {e}")
        
        logger.add(
            str(log_file),
            rotation="10 MB",
            retention="1 week",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            mode='w'  # 使用写入模式，覆盖任何现有文件
        )
    
    return logger



def toOpen3dCloud(points,colors=None,normals=None):
  cloud = o3d.geometry.PointCloud()
  cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
  if colors is not None:
    if colors.max()>1:
      colors = colors/255.0
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
  if normals is not None:
    cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
  return cloud



def depth2xyzmap(depth:np.ndarray, K, uvs:np.ndarray=None, zmin=0.1):
  invalid_mask = (depth<zmin)
  H,W = depth.shape[:2]
  if uvs is None:
    vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
  else:
    uvs = uvs.round().astype(int)
    us = uvs[:,0]
    vs = uvs[:,1]
  zs = depth[vs,us]
  xs = (us-K[0,2])*zs/K[0,0]
  ys = (vs-K[1,2])*zs/K[1,1]
  pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
  xyz_map = np.zeros((H,W,3), dtype=np.float32)
  xyz_map[vs,us] = pts
  if invalid_mask.any():
    xyz_map[invalid_mask] = 0
  return xyz_map


def vis_disparity(disp, min_val=None, max_val=None, invalid_thres=np.inf, color_map=cv2.COLORMAP_TURBO, cmap=None, other_output={}):
  """
  @disp: np array (H,W)
  @invalid_thres: > thres is invalid
  """
  disp = disp.copy()
  H,W = disp.shape[:2]
  invalid_mask = disp>=invalid_thres
  if (invalid_mask==0).sum()==0:
    other_output['min_val'] = None
    other_output['max_val'] = None
    return np.zeros((H,W,3))
  if min_val is None:
    min_val = disp[invalid_mask==0].min()
  if max_val is None:
    max_val = disp[invalid_mask==0].max()
  other_output['min_val'] = min_val
  other_output['max_val'] = max_val
  vis = ((disp-min_val)/(max_val-min_val)).clip(0,1) * 255
  if cmap is None:
    vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), color_map)[...,::-1]
  else:
    vis = cmap(vis.astype(np.uint8))[...,:3]*255
  if invalid_mask.any():
    vis[invalid_mask] = 0
  return vis.astype(np.uint8)


def timeit(func):
  from loguru import logger
  def wrapper(*args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    logger.info(f"{func.__name__} 执行时间: {end_time - start_time} 秒")
    return result
  return wrapper