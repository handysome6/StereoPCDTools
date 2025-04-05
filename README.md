# StereoPCDTools
用于`双目图像对生成点云`项目的一些实用工具。


## 安装依赖

```bash
pip install -r requirements.txt
```
<br>


## 远程获取点云和视差
`stereo_client.py` 是用于端侧获取双目图像对的点云（PLY）以及视差图的程序，可以远程连接部署到GPU服务器的server。

##### Example usage:
```
python stereo_client.py \
    --left ./example/A_106345528590.jpg \
    --right ./example/D_106345528590.jpg \
    --output ./output \
    --intrinsic ./example/K_477module.txt \
    --scale 0.35 \
    --server https://stereo-cfd.andy6.link/
```

##### 客户端命令行使用方式：
```bash
python stereo_client.py --left 左图像路径 --right 右图像路径 --intrinsic 内参文件路径 [--server 服务器URL] [--output 输出目录] [--scale 缩放比例]
```

参数说明：
- `--left`: 左图像路径（必需）
- `--right`: 右图像路径（必需）
- `--intrinsic`: 相机内参文件路径（必需）
- `--server`: 服务器URL，默认为 http://localhost:8000
- `--output`: 输出目录路径，默认使用左图像所在目录下的 results 文件夹
- `--scale`: 图像缩放比例，默认为 1.0


##### K.txt 内参文件示例：
```
1000 0 960 0 1000 540 0 0 1
0.12
```
这个示例表示：
- 焦距 fx = fy = 1000
- 主点 cx = 960, cy = 540
- 基线长度 = 0.12 米

##### 输出结果说明
处理完成后，将在输出目录中生成以下文件：
- `img0.png`, `img1.png`: 预处理后的左右图像
- `disp.npy`: 视差图
- `depth_meter.npy`: 深度图（单位：米）
- `cloud.ply`: 生成的点云文件（可使用 MeshLab 等软件查看）
- `vis.png`: 可视化结果，包含原图和视差图 
<br>


## 图像对比较器
`image_pair_comparator.py` 是一个使用PySide6开发的图像对比工具，可以同步显示两张图像，并在任一视图中缩放和平移时同步到另一个视图。

##### 功能特点

- 同时显示两张图像进行对比
- 在任意一个视图中缩放时，另一个视图也会同步缩放
- 在任意一个视图中平移时，另一个视图也会同步平移
- 支持鼠标拖拽和滚轮缩放操作
- 按住Ctrl键点击可在两个视图中同时添加水平参考线
- 按Delete键可清除所有添加的参考线


##### 使用方法

1. 确保你有两张需要比较的图像 (默认为img0.png和img1.png)
2. 运行主程序
```bash
python image_pair_comparator.py
```
3. 操作说明:
   - 使用鼠标滚轮缩放图像
   - 按住鼠标左键拖拽图像
   - 所有操作都会同步到另一个视图
   - 按住Ctrl键并点击可在两个视图中同时添加水平参考线，便于对比图像中的特定区域
   - 按Delete键可清除所有添加的参考线

##### 自定义图像

如果需要使用其他图像，可以修改image_pair_comparator.py文件中的路径配置：

```python
# 图像路径
image1_path = "你的第一张图像路径.png"
image2_path = "你的第二张图像路径.png"
```
<br>


## 图像对Y坐标错位统计分析工具

`stats_visualizer.py` 是一个用于分析图像对之间特征点Y坐标错位的工具。它可以：

1. 使用多种特征检测器（SIFT、ORB、AKAZE、BRISK）检测图像特征
2. 匹配两幅图像之间的特征点
3. 计算匹配特征点之间的Y坐标差异
4. 生成Y坐标错位的统计直方图
5. 可视化特征匹配结果

##### 使用方法

```bash
python stats_visualizer.py --img1 <第一张图像路径> --img2 <第二张图像路径> --detector <特征检测器> --matcher <匹配器> --output <输出文件夹>
```

参数说明：
- `--img1`：第一张图像的路径（默认：img0.png）
- `--img2`：第二张图像的路径（默认：img1.png）
- `--detector`：特征检测器类型，可选：SIFT、ORB、AKAZE、BRISK（默认：SIFT）
- `--matcher`：特征匹配器类型，可选：BF（暴力匹配）、FLANN（快速近似最近邻匹配）（默认：BF）
- `--output`：输出文件夹路径（默认：results）

##### 示例

```bash
# 使用SIFT检测器和BF匹配器分析img0.png和img1.png
python stats_visualizer.py

# 使用ORB检测器和FLANN匹配器
python stats_visualizer.py --detector ORB --matcher FLANN

# 指定不同的图像文件
python stats_visualizer.py --img1 path/to/image1.jpg --img2 path/to/image2.jpg
```

结果将保存在指定的输出文件夹中，包括：
1. 特征匹配可视化图像
2. Y坐标错位直方图 