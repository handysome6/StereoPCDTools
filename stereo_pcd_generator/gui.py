import sys
from pathlib import Path
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QGroupBox,
    QRadioButton, QDoubleSpinBox, QMessageBox, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal
from .generator import generae_pcd_raw_images, generate_pcd_dir

class PcdGeneratorWorker(QThread):
    progress_update = Signal(str)
    finished = Signal(bool, str)  # 成功/失败, 消息

    def __init__(self, is_single_pair, params):
        super().__init__()
        self.is_single_pair = is_single_pair
        self.params = params

    def run(self):
        try:
            if self.is_single_pair:
                self.progress_update.emit("正在处理单个图像对...")
                generae_pcd_raw_images(**self.params)
                self.finished.emit(True, "单个点云生成完成！")
            else:
                self.progress_update.emit("正在处理多个图像对...")
                generate_pcd_dir(**self.params)
                self.finished.emit(True, "目录中的所有点云生成完成！")
        except Exception as e:
            self.finished.emit(False, f"处理过程中出错: {str(e)}")

class StereoPcdGeneratorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("立体视觉点云生成工具")
        self.setMinimumSize(700, 500)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(main_widget)
        
        # 摄像机模型部分
        camera_model_group = QGroupBox("摄像机模型")
        camera_model_layout = QHBoxLayout()
        
        camera_model_layout.addWidget(QLabel("摄像机模型路径:"))
        self.camera_model_path = QLineEdit()
        self.camera_model_path.setText(str(Path.home() / "DCIM" / "camera_model.json"))
        camera_model_layout.addWidget(self.camera_model_path)
        
        self.browse_camera_model_btn = QPushButton("浏览...")
        self.browse_camera_model_btn.clicked.connect(self.browse_camera_model)
        camera_model_layout.addWidget(self.browse_camera_model_btn)
        
        camera_model_group.setLayout(camera_model_layout)
        main_layout.addWidget(camera_model_group)
        
        # 输入模式选择
        input_mode_group = QGroupBox("输入模式")
        input_mode_layout = QVBoxLayout()
        
        # 单图像对模式
        self.single_pair_radio = QRadioButton("单图像对")
        self.single_pair_radio.setChecked(True)
        self.single_pair_radio.toggled.connect(self.toggle_input_mode)
        input_mode_layout.addWidget(self.single_pair_radio)
        
        self.single_pair_widget = QWidget()
        single_pair_layout = QHBoxLayout(self.single_pair_widget)
        single_pair_layout.setContentsMargins(20, 0, 0, 0)
        
        single_pair_layout.addWidget(QLabel("选择一个图像:"))
        self.single_image_path = QLineEdit()
        single_pair_layout.addWidget(self.single_image_path)
        
        self.browse_single_image_btn = QPushButton("浏览...")
        self.browse_single_image_btn.clicked.connect(self.browse_single_image)
        single_pair_layout.addWidget(self.browse_single_image_btn)
        
        input_mode_layout.addWidget(self.single_pair_widget)
        
        # 目录模式
        self.directory_radio = QRadioButton("图像目录")
        self.directory_radio.toggled.connect(self.toggle_input_mode)
        input_mode_layout.addWidget(self.directory_radio)
        
        self.directory_widget = QWidget()
        directory_layout = QHBoxLayout(self.directory_widget)
        directory_layout.setContentsMargins(20, 0, 0, 0)
        
        directory_layout.addWidget(QLabel("图像目录路径:"))
        self.directory_path = QLineEdit()
        directory_layout.addWidget(self.directory_path)
        
        self.browse_directory_btn = QPushButton("浏览...")
        self.browse_directory_btn.clicked.connect(self.browse_directory)
        directory_layout.addWidget(self.browse_directory_btn)
        
        self.directory_widget.setVisible(False)
        input_mode_layout.addWidget(self.directory_widget)
        
        input_mode_group.setLayout(input_mode_layout)
        main_layout.addWidget(input_mode_group)
        
        # 输出目录
        output_group = QGroupBox("输出设置")
        output_layout = QVBoxLayout()
        
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("输出目录:"))
        self.output_path = QLineEdit()
        output_dir_layout.addWidget(self.output_path)
        
        self.browse_output_btn = QPushButton("浏览...")
        self.browse_output_btn.clicked.connect(self.browse_output)
        output_dir_layout.addWidget(self.browse_output_btn)
        
        output_layout.addLayout(output_dir_layout)
        
        # 尺度设置
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("尺度:"))
        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setRange(0.01, 1.0)
        self.scale_spinbox.setSingleStep(0.05)
        self.scale_spinbox.setValue(0.35)
        scale_layout.addWidget(self.scale_spinbox)
        scale_layout.addStretch()
        
        output_layout.addLayout(scale_layout)
        
        # 服务器URL设置
        server_layout = QHBoxLayout()
        server_layout.addWidget(QLabel("服务器URL:"))
        self.server_url = QLineEdit()
        
        # 从.env文件加载默认值
        import dotenv
        dotenv.load_dotenv()
        default_url = os.getenv("server_url", "")
        self.server_url.setText(default_url)
        self.server_url.setPlaceholderText("留空使用环境变量中的URL")
        
        server_layout.addWidget(self.server_url)
        output_layout.addLayout(server_layout)
        
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)
        
        # 状态进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("就绪")
        main_layout.addWidget(self.progress_bar)
        
        # 生成按钮
        self.generate_btn = QPushButton("生成点云")
        self.generate_btn.clicked.connect(self.generate_pcd)
        self.generate_btn.setMinimumHeight(40)
        main_layout.addWidget(self.generate_btn)
        
    def toggle_input_mode(self):
        self.single_pair_widget.setVisible(self.single_pair_radio.isChecked())
        self.directory_widget.setVisible(self.directory_radio.isChecked())
    
    def browse_camera_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择摄像机模型文件", "", "JSON文件 (*.json)"
        )
        if file_path:
            self.camera_model_path.setText(file_path)
    
    def browse_single_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "", "图像文件 (*.jpg *.jpeg *.png)"
        )
        if file_path:
            self.single_image_path.setText(file_path)
    
    def browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择图像目录")
        if dir_path:
            self.directory_path.setText(dir_path)
    
    def browse_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_path.setText(dir_path)
    
    def generate_pcd(self):
        # 检查摄像机模型路径
        camera_model_path = self.camera_model_path.text()
        if not Path(camera_model_path).exists():
            QMessageBox.warning(self, "错误", "摄像机模型文件不存在!")
            return
        
        # 检查输出路径
        output_path = self.output_path.text()
        if not output_path:
            QMessageBox.warning(self, "错误", "请指定输出目录!")
            return
            
        # 创建输出目录(如果不存在)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # 获取尺度值
        scale = self.scale_spinbox.value()
        
        # 获取服务器URL
        server_url = self.server_url.text()
        if server_url:
            os.environ["server_url"] = server_url
        
        if self.single_pair_radio.isChecked():
            # 单图像对模式
            image_path = self.single_image_path.text()
            if not Path(image_path).exists():
                QMessageBox.warning(self, "错误", "所选图像文件不存在!")
                return
                
            # 推断对应的图像对
            filename = Path(image_path).name
            parent_dir = Path(image_path).parent
            
            if filename.startswith("A_"):
                left_image = image_path
                right_image = str(parent_dir / filename.replace("A_", "D_"))
            elif filename.startswith("D_"):
                right_image = image_path
                left_image = str(parent_dir / filename.replace("D_", "A_"))
            else:
                QMessageBox.warning(self, "错误", "图像文件名必须以A_或D_开头!")
                return
                
            if not Path(left_image).exists() or not Path(right_image).exists():
                QMessageBox.warning(self, "错误", "无法找到对应的图像对!")
                return
                
            # 准备参数
            params = {
                "raw_left_path": left_image,
                "raw_right_path": right_image,
                "camera_model_path": camera_model_path,
                "output_dir": output_path,
                "scale": scale
            }
            
            # 启动工作线程
            self.worker = PcdGeneratorWorker(True, params)
            self.start_worker()
            
        else:
            # 目录模式
            dir_path = self.directory_path.text()
            if not Path(dir_path).exists():
                QMessageBox.warning(self, "错误", "所选目录不存在!")
                return
                
            # 准备参数
            params = {
                "raw_dir": dir_path,
                "camera_model_path": camera_model_path,
                "output_dir": output_path,
                "scale": scale
            }
            
            # 启动工作线程
            self.worker = PcdGeneratorWorker(False, params)
            self.start_worker()
    
    def start_worker(self):
        # 禁用生成按钮
        self.generate_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("处理中...")
        
        # 连接信号
        self.worker.progress_update.connect(self.update_progress)
        self.worker.finished.connect(self.process_finished)
        
        # 启动工作线程
        self.worker.start()
    
    def update_progress(self, message):
        self.progress_bar.setFormat(message)
    
    def process_finished(self, success, message):
        # 重新启用生成按钮
        self.generate_btn.setEnabled(True)
        
        # 更新进度条
        if success:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("完成")
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("成功")
            msg_box.setText(message)
            open_btn = msg_box.addButton("打开输出目录", QMessageBox.ActionRole)
            close_btn = msg_box.addButton("关闭", QMessageBox.AcceptRole)
            msg_box.exec()
            
            if msg_box.clickedButton() == open_btn:
                import platform
                import subprocess
                
                output_path = self.output_path.text()
                system = platform.system()
                
                try:
                    if system == 'Darwin':  # macOS
                        subprocess.run(['open', output_path])
                    elif system == 'Windows':  # Windows
                        os.startfile(output_path)
                    else:  # Linux 或其他系统
                        subprocess.run(['xdg-open', output_path])
                except Exception as e:
                    QMessageBox.warning(self, "警告", f"无法打开输出目录: {str(e)}")
        else:
            self.progress_bar.setFormat("错误")
            QMessageBox.critical(self, "错误", message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StereoPcdGeneratorUI()
    window.show()
    sys.exit(app.exec()) 