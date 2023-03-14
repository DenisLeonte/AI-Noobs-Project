from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import sys


# class ImageListView(QWidget):
#     def __init__(self):
#         super().__init__()
#
#         self.view = QListWidget()
#         self.view.setIconSize(QSize(64, 64))
#         bigFont = self.font()
#         bigFont.setPointSize(24)
#         self.view.setFont(bigFont)
#
#         self.addButton = QPushButton('Add image(s)')
#
#         layout = QVBoxLayout(self)
#         layout.addWidget(self.view)
#         layout.addWidget(self.addButton)
#         self.addButton.clicked.connect(self.addImage)
#         self.addButton.clicked(self.addImage)

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Test UI")
        self.view = QListWidget()
        self.setGeometry(100, 100, 600, 400)
        self.UIComponents()
        self.show()

    def UIComponents(self):
        addButton = QPushButton("Add image(s)", self)
        addButton.setGeometry(200, 150, 100, 30)
        addButton.clicked.connect(self.clickToAddPhoto)

    def clickToAddPhoto(self):
        paths, _ = QFileDialog.getOpenFileName(self, 'Select image(s)', '', 'Images (*.png *,jpg *jpeg)')
        size = self.view.iconSize()
        for path in paths:
            source = QPixmap(path)
            if source.isNull():
                continue
            if source.width() > size.width() or source.height() > size.height:
                source = source.scaled(size)
            square = QPixmap(size)
            qp = QPainter(square)
            rect = source.rect()
            rect.moveCenter(square.rect().center())
            qp.drawPixmap(rect, source)
            qp.end()

            name = QFileInfo(path).baseName()
            item = QListWidgetItem(name)
            item.setIcon(QIcon(square))
            item.setToolTip(path)
            item.setSizeHint(size)
            self.view.addItem(item)


App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())
