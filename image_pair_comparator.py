#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QGraphicsView, 
                             QGraphicsScene, QHBoxLayout, QWidget,
                             QFileDialog)
from PySide6.QtGui import QPixmap, QTransform, QPen
from PySide6.QtCore import Qt, Signal, QPointF, QLineF
from loguru import logger

class SyncedGraphicsView(QGraphicsView):
    """自定义GraphicsView类，支持同步缩放和平移"""
    viewChanged = Signal(QTransform)
    horizontalLineAdded = Signal(float)  # 添加水平线的信号，参数为y坐标
    linesCleared = Signal()  # 清除线条的信号
    
    def __init__(self, scene=None, partner=None):
        super().__init__(scene)
        self.partner = partner
        self.setRenderHint(self.renderHints().Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._isPanning = False
        self._lastMousePos = QPointF(0, 0)
        self.lines = []  # 存储添加的线条对象
        
    def setPartner(self, partner):
        """设置伙伴视图"""
        self.partner = partner
        
    def wheelEvent(self, event):
        """处理鼠标滚轮事件，缩放视图"""
        factor = 1.1
        if event.angleDelta().y() < 0:
            factor = 1.0 / factor
            
        self.scale(factor, factor)
        
        # 同步伙伴视图
        if self.partner:
            # 暂时断开连接以避免循环调用
            old_partner = self.partner
            self.partner = None
            old_partner.setTransform(self.transform())
            old_partner.partner = self
            
    def mousePressEvent(self, event):
        """处理鼠标按下事件，根据按键决定拖拽或添加水平线"""
        if event.button() == Qt.LeftButton:
            # 检查是否按下了Ctrl键（用于添加线条）
            if event.modifiers() & Qt.ControlModifier:
                logger.info("Adding h line")
                # 获取鼠标点击位置的场景坐标
                scene_pos = self.mapToScene(event.position().toPoint())
                self.addHorizontalLine(scene_pos.y())
                event.accept()
            else:
                # 原来的拖拽功能
                self._isPanning = True
                self._lastMousePos = event.position()
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
        else:
            super().mousePressEvent(event)
            
    def mouseMoveEvent(self, event):
        """处理鼠标移动事件，平移视图"""
        if self._isPanning:
            delta = event.position() - self._lastMousePos
            self._lastMousePos = event.position()
            
            # 应用平移
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            
            # 同步伙伴视图
            if self.partner:
                self.partner.horizontalScrollBar().setValue(
                    self.horizontalScrollBar().value())
                self.partner.verticalScrollBar().setValue(
                    self.verticalScrollBar().value())
                
            event.accept()
        else:
            super().mouseMoveEvent(event)
            
    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件，结束拖拽"""
        if event.button() == Qt.LeftButton and self._isPanning:
            self._isPanning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)
    
    def setTransform(self, transform):
        """重写setTransform方法以同步伙伴视图"""
        super().setTransform(transform)
        
        # 同步伙伴视图
        if self.partner:
            # 暂时断开连接以避免循环调用
            old_partner = self.partner
            self.partner = None
            old_partner.setTransform(transform)
            old_partner.partner = self

    def addHorizontalLine(self, y_pos):
        """在指定y坐标位置添加水平线，并通知伙伴视图"""
        if not self.scene():
            return
            
        # 获取场景的宽度范围
        scene_rect = self.scene().sceneRect()
        line = QLineF(scene_rect.left(), y_pos, scene_rect.right(), y_pos)
        
        # 创建红色笔，宽度为2
        pen = QPen(Qt.red)
        pen.setWidth(2)
        
        # 添加线到场景
        line_item = self.scene().addLine(line, pen)
        self.lines.append(line_item)
        
        # 发送信号通知伙伴视图也添加线
        if self.partner:
            self.horizontalLineAdded.emit(y_pos)

    def clearLines(self):
        """清除所有水平线"""
        if not self.scene():
            return
            
        # 从场景中移除所有线条
        for line in self.lines:
            self.scene().removeItem(line)
            
        # 清空线条列表
        self.lines.clear()
        
        # 发送信号通知伙伴视图也清除线条
        if self.partner:
            self.linesCleared.emit()

    def keyPressEvent(self, event):
        """处理键盘按键事件"""
        # 当按下Delete键时，清除所有线条
        if event.key() == Qt.Key_Delete:
            self.clearLines()
            event.accept()
        else:
            super().keyPressEvent(event)


class MainWindow(QMainWindow):
    """主窗口类"""
    def __init__(self, image1_path, image2_path):
        super().__init__()
        
        # 设置窗口标题和大小
        self.setWindowTitle("图像对比较器")
        self.resize(1200, 600)
        
        # 创建中央部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # 创建左侧场景和视图
        self.scene1 = QGraphicsScene()
        self.view1 = SyncedGraphicsView(self.scene1)
        
        # 创建右侧场景和视图
        self.scene2 = QGraphicsScene()
        self.view2 = SyncedGraphicsView(self.scene2)
        
        # 关联两个视图
        self.view1.setPartner(self.view2)
        self.view2.setPartner(self.view1)
        
        # 连接添加水平线的信号
        self.view1.horizontalLineAdded.connect(self.view2.addHorizontalLine)
        self.view2.horizontalLineAdded.connect(self.view1.addHorizontalLine)
        
        # 连接清除线条的信号
        self.view1.linesCleared.connect(self.view2.clearLines)
        self.view2.linesCleared.connect(self.view1.clearLines)
        
        # 加载图像
        self.pixmap1 = QPixmap(image1_path)
        self.pixmap2 = QPixmap(image2_path)
        
        # 将图像添加到场景
        self.scene1.addPixmap(self.pixmap1)
        self.scene2.addPixmap(self.pixmap2)
        
        # 设置场景大小
        self.scene1.setSceneRect(self.pixmap1.rect())
        self.scene2.setSceneRect(self.pixmap2.rect())
        
        # 添加视图到布局
        layout.addWidget(self.view1)
        layout.addWidget(self.view2)
        
        # 设置视图初始显示为适合视图大小
        self.view1.fitInView(self.scene1.sceneRect(), Qt.KeepAspectRatio)
        self.view2.fitInView(self.scene2.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        """处理窗口大小变化事件"""
        super().resizeEvent(event)
        # 重新调整视图以适应窗口大小变化
        self.view1.fitInView(self.scene1.sceneRect(), Qt.KeepAspectRatio)
        self.view2.fitInView(self.scene2.sceneRect(), Qt.KeepAspectRatio)
        
    def showEvent(self, event):
        """处理窗口首次显示事件"""
        super().showEvent(event)
        # 确保图像正确适应视图
        self.view1.fitInView(self.scene1.sceneRect(), Qt.KeepAspectRatio)
        self.view2.fitInView(self.scene2.sceneRect(), Qt.KeepAspectRatio)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 创建文件选择对话框
    file_dialog = QFileDialog()
    file_dialog.setNameFilter("图像文件 (*.png *.jpg *.jpeg)")
    file_dialog.setViewMode(QFileDialog.Detail)
    
    if file_dialog.exec():
        selected_files = file_dialog.selectedFiles()
        if not selected_files:
            sys.exit(0)
            
        selected_path = selected_files[0]
        # 获取文件名和目录
        file_name = os.path.basename(selected_path)
        directory = os.path.dirname(selected_path)
        
        # 推断图像对路径
        if file_name.startswith('A_'):
            # 如果是A图像，寻找对应的D图像
            id_part = file_name[2:]  # 获取ID部分
            image1_path = selected_path
            image2_path = os.path.join(directory, f'D_{id_part}')
        elif file_name.startswith('D_'):
            # 如果是D图像，寻找对应的A图像
            id_part = file_name[2:]  # 获取ID部分
            image1_path = os.path.join(directory, f'A_{id_part}')
            image2_path = selected_path
        else:
            logger.error("选择的图像文件名必须以'A_'或'D_'开头")
            sys.exit(1)
            
        # 检查文件是否存在
        if not os.path.exists(image1_path) or not os.path.exists(image2_path):
            logger.error(f"找不到对应的图像对：\n{image1_path}\n{image2_path}")
            sys.exit(1)
            
        window = MainWindow(image1_path, image2_path)
        window.show()
        
        sys.exit(app.exec())
    else:
        sys.exit(0) 