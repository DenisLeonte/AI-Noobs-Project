from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import sys


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Test UI")
        self.view = QListWidget()
        self.setGeometry(100, 100, 800, 600)
        self.UIComponents(text)
        self.show()

    def UIComponents(self, text):
        addButton = QPushButton("Add image(s)", self)
        addButton.setGeometry(30, 30, 100, 30)
        addButton.clicked.connect(self.clickToAddPhoto)
        label = QLabel(text, self)
        label.move(150, 30)

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


text = "Print test";
App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())
