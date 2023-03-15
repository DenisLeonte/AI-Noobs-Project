from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import sys


class Window(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Artificial Intelligence")
        self.view = QListWidget()
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon("AI.jpg"))

        vbox = QVBoxLayout()
        addButton = QPushButton("Browse image")
        addButton.clicked.connect(self.browseImage)
        vbox.addWidget(addButton)

        self.label = QLabel(text)
        vbox.addWidget(self.label)

        self.setLayout(vbox)

        self.show()

    def browseImage(self):
        fileName = QFileDialog.getOpenFileName(self, 'Select image', 'c\\', 'Images (*.png *.jpg *.jpeg)')

        imagePath = fileName[0]
        pixmap = QPixmap(imagePath)
        self.label.setPixmap(QPixmap(pixmap))
        self.resize(pixmap.width(), pixmap.height())

    def UIComponents(self, printText):
        myFont = QFont()
        myFont.setBold(True)
        label = QLabel(printText, self)
        label.setFont(myFont)
        label.move(150, 25)

        vbox = QVBoxLayout()
        addButton = QPushButton("Browse image", self)
        addButton.setGeometry(30, 20, 100, 30)
        addButton.clicked.connect(self.browseImage)

        vbox.addWidget(addButton)

    # def clickToAddPhoto(self):
    #     paths, _ = QFileDialog.getOpenFileName(self, 'Select image(s)', '', 'Images (*.png *,jpg *jpeg)')
    #     size = self.view.iconSize()
    #     for path in paths:
    #         source = QPixmap(path)
    #         if source.isNull():
    #             continue
    #         if source.width() > size.width() or source.height() > size.height:
    #             source = source.scaled(size)
    #         square = QPixmap(size)
    #         qp = QPainter(square)
    #         rect = source.rect()
    #         rect.moveCenter(square.rect().center())
    #         qp.drawPixmap(rect, source)
    #         qp.end()
    #
    #         name = QFileInfo(path).baseName()
    #         item = QListWidgetItem(name)
    #         item.setIcon(QIcon(square))
    #         item.setToolTip(path)
    #         item.setSizeHint(size)
    #         self.view.addItem(item)
    #         return self.view


text = "Print test"
App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())
