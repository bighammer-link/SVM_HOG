import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QPlainTextEdit
from PyQt5.QtGui import QPixmap, QIcon
from test import test
from HOG import *
from SVM import *



class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('单控2班_王铭')
        self.setWindowIcon(QIcon("images/uilogo/logo.jpg"))

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(500, 500)

        select_image_button = QPushButton('选择图片', self)
        select_image_button.clicked.connect(self.select_image)

        detect_object_button = QPushButton('进行检测', self)
        detect_object_button.clicked.connect(self.detect_object)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(select_image_button)
        layout.addWidget(detect_object_button)

        self.setLayout(layout)
        self.output_textedit = QPlainTextEdit(self)
        self.output_textedit.setReadOnly(True)  # 设置为只读，不可编辑
        self.output_textedit.setMinimumHeight(100)

        layout.addWidget(self.output_textedit)

    def select_image(self):
        file_dialog = QFileDialog() # 交互式的选择照片
        image_path, _ = file_dialog.getOpenFileName(self, 'Select Image', '', 'Image Files (*.png *.jpg *.bmp *.jpeg)')
        if image_path:
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap)
            self.image_path = image_path

    def detect_object(self):
        if hasattr(self, 'image_path'):
            # 在这里调用物体检测函数，并捕获输出信息
            import sys
            from io import StringIO

            # 保存原始sys.stdout
            original_stdout = sys.stdout
            # 重定向sys.stdout到StringIO
            sys.stdout = StringIO()

            try:
                test(self.image_path)
                # 获取输出信息
                output_info = sys.stdout.getvalue()
                # 将输出信息显示到界面上
                self.output_textedit.appendPlainText(output_info)
            finally:
                # 恢复原始sys.stdout
                sys.stdout = original_stdout
    '''
    def detect_object(self):
        if hasattr(self, 'image_path'):     # 确保已经选择了一个图像，未选择图像则打印信息

            test(self.image_path)
        else:
            print('请先选择照片')
    '''





# TODO 进行训练界面的UI设计

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())
