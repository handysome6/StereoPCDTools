import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QListWidget, QLabel, QFileSystemModel,
                             QFileDialog, QMenuBar, QMenu, QScrollArea, QListWidgetItem,
                             QPushButton, QSizePolicy, QGridLayout, QSlider)
from PySide6.QtCore import Qt, QDir, QSize
from PySide6.QtGui import QPixmap, QIcon
import subprocess
from multiprocessing import Process


def _show_point_cloud(ply_path):
    import open3d as o3d
    # 读取点云
    pcd = o3d.io.read_point_cloud(ply_path)
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="点云查看器")
    
    # 添加点云到场景
    vis.add_geometry(pcd)
    
    # 设置默认视角
    vis.get_view_control().set_front([0, 0, -1])
    vis.get_view_control().set_lookat([0, 0, 0])
    vis.get_view_control().set_up([0, -1, 0])
    
    # 渲染点云
    vis.run()
    vis.destroy_window()
        
class ProjectExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("项目文件夹浏览器")
        self.setMinimumSize(1600, 900)
        
        # 保存点云查看进程的引用
        self.cloud_viewer_process = None
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局（水平布局）
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建左侧布局
        left_widget = QWidget()
        left_widget.setMinimumWidth(600)  # 设置最小宽度
        left_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_widget)
        
        # 创建图标数量控制滑动条
        size_control_layout = QHBoxLayout()
        size_label = QLabel("每行图标数:")
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(2)  # 最少每行2个
        self.size_slider.setMaximum(6)  # 最多每行6个
        self.size_slider.setValue(4)  # 默认每行4个
        self.size_slider.valueChanged.connect(self.on_icons_per_row_changed)
        size_control_layout.addWidget(size_label)
        size_control_layout.addWidget(self.size_slider)
        left_layout.addLayout(size_control_layout)
        
        # 创建左侧项目列表视图
        self.project_list = QListWidget()
        self.project_list.setViewMode(QListWidget.IconMode)
        self.project_list.setSpacing(10)
        self.project_list.setResizeMode(QListWidget.Adjust)
        self.project_list.setMovement(QListWidget.Static)
        self.project_list.setUniformItemSizes(True)
        self.project_list.currentItemChanged.connect(self.on_project_selected)
        
        # 设置网格大小
        self.update_grid_size()
        
        left_layout.addWidget(self.project_list)
        
        main_layout.addWidget(left_widget)
        
        # 创建右侧图片预览区域
        self.preview_widget = QWidget()
        self.preview_widget.setVisible(False)
        self.preview_widget.setMinimumWidth(600)  # 设置最小宽度为600
        self.preview_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout = QVBoxLayout(self.preview_widget)
        
        # 创建网格布局用于图片显示
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        
        # 创建三个图片显示区域
        self.img0_label = QLabel("img0")
        self.img1_label = QLabel("img1")
        self.vis_label = QLabel("vis")
        
        # 设置图片标签的属性
        for label in [self.img0_label, self.img1_label, self.vis_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid gray; background-color: white;")
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 将图片标签添加到网格布局
        grid_layout.addWidget(self.img0_label, 0, 0)
        grid_layout.addWidget(self.img1_label, 0, 1)
        grid_layout.addWidget(self.vis_label, 1, 0, 1, 2)  # 跨越两列
        
        preview_layout.addLayout(grid_layout)
        
        # 创建操作按钮区域
        button_layout = QHBoxLayout()
        
        # 打开文件夹按钮
        self.open_folder_button = QPushButton("打开文件夹")
        self.open_folder_button.setFixedSize(120, 30)
        self.open_folder_button.clicked.connect(self.open_current_folder)
        
        # 查看点云按钮
        self.view_cloud_button = QPushButton("查看点云")
        self.view_cloud_button.setFixedSize(120, 30)
        self.view_cloud_button.clicked.connect(self.view_point_cloud)
        
        button_layout.addWidget(self.open_folder_button)
        button_layout.addWidget(self.view_cloud_button)
        button_layout.addStretch()
        
        preview_layout.addLayout(button_layout)
        
        # 创建右侧容器（始终可见，内部的preview_widget控制显示/隐藏）
        right_widget = QWidget()
        right_widget.setMinimumWidth(600)  # 设置最小宽度为600
        right_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(self.preview_widget)
        
        main_layout.addWidget(right_widget)
        
        # 设置左右两侧的比例为2:3
        main_layout.setStretch(0, 2)  # 左侧
        main_layout.setStretch(1, 3)  # 右侧
        
        # 保存当前选中的项目路径
        self.current_project_path = None
        
        # 初始化根目录
        self.root_dir = QDir.currentPath()
        
        # 在显示窗口之前先选择文件夹
        self.select_folder()
        
        # 加载项目
        self.load_projects()
    
    def update_grid_size(self):
        icons_per_row = self.size_slider.value()
        available_width = self.project_list.viewport().width() or 600  # 使用600作为初始宽度
        icon_size = (available_width - 20) // icons_per_row  # 减去边距
        self.project_list.setGridSize(QSize(icon_size, icon_size + 40))  # 高度加上文本空间
        self.project_list.setIconSize(QSize(icon_size - 20, icon_size - 20))  # 图标略小于格子
    
    def on_icons_per_row_changed(self, value):
        self.update_grid_size()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_grid_size()
    
    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        
        # 添加选择文件夹动作
        select_folder_action = file_menu.addAction("选择文件夹")
        select_folder_action.triggered.connect(self.select_folder)
    
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择项目根目录",
            self.root_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder:
            self.root_dir = folder
            self.project_list.clear()
            self.preview_widget.setVisible(False)
            self.current_project_path = None
            self.load_projects()
        else:
            # 如果用户取消选择，则退出程序
            sys.exit()
    
    def load_projects(self):
        # 获取根目录下的所有项目文件夹
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path):
                img0_path = os.path.join(item_path, "img0.png")
                
                # 创建列表项
                list_item = QListWidgetItem(item)
                list_item.setTextAlignment(Qt.AlignCenter)
                
                # 设置图标
                if os.path.exists(img0_path):
                    icon = QIcon(img0_path)
                    list_item.setIcon(icon)
                
                # 添加到列表
                self.project_list.addItem(list_item)
        
        # 更新网格大小
        self.update_grid_size()
    
    def on_project_selected(self, current, previous):
        if current is None:
            return
            
        project_name = current.text()
        self.current_project_path = os.path.join(self.root_dir, project_name)
        
        # 显示预览区域
        self.preview_widget.setVisible(True)
        
        # 加载三张图片
        image_files = {
            self.img0_label: "img0.png",
            self.img1_label: "img1.png",
            self.vis_label: "vis.png"
        }
        
        for label, filename in image_files.items():
            img_path = os.path.join(self.current_project_path, filename)
            if os.path.exists(img_path):
                pixmap = QPixmap(img_path)
                scaled_pixmap = pixmap.scaled(label.size(), 
                                            Qt.KeepAspectRatio, 
                                            Qt.SmoothTransformation)
                label.setPixmap(scaled_pixmap)
            else:
                label.clear()
                label.setText(f"No {filename}")
    
    def open_current_folder(self):
        if self.current_project_path and os.path.exists(self.current_project_path):
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', self.current_project_path])
            elif sys.platform == 'win32':  # Windows
                subprocess.run(['explorer', self.current_project_path])
            else:  # Linux
                subprocess.run(['xdg-open', self.current_project_path])
    
    def view_point_cloud(self):
        if not self.current_project_path:
            return
            
        ply_path = os.path.join(self.current_project_path, "cloud.ply")
        if not os.path.exists(ply_path):
            return
            
        # 如果已经有正在运行的点云查看进程，先终止它
        if self.cloud_viewer_process and self.cloud_viewer_process.is_alive():
            self.cloud_viewer_process.terminate()
            self.cloud_viewer_process.join()

        
        # 创建新的点云查看进程
        self.cloud_viewer_process = Process(target=_show_point_cloud, args=(ply_path,))
        self.cloud_viewer_process.start()
    
    def closeEvent(self, event):
        # 在关闭主窗口时，确保点云查看进程也被终止
        if self.cloud_viewer_process and self.cloud_viewer_process.is_alive():
            self.cloud_viewer_process.terminate()
            self.cloud_viewer_process.join()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProjectExplorer()
    window.show()
    sys.exit(app.exec()) 