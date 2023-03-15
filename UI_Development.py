from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Artificial Intelligence")
        # self.view = QListWidget()
        self.setGeometry(100, 100, 800, 700)
        self.setWindowIcon(QIcon("AI.jpg"))

        vbox = QVBoxLayout()
        self.setLayout(vbox)

        addButton = QPushButton("Browse image")
        addButton.clicked.connect(self.browseImage)
        vbox.addWidget(addButton)

        self.label = QLabel()
        self.label.adjustSize()
        vbox.addWidget(self.label)

        self.text = QLabel(text)
        myFont = QFont()
        myFont.setBold(True)
        self.text.setFont(myFont)
        vbox.addWidget(self.text)

        self.show()

    def browseImage(self):
        fileName = QFileDialog.getOpenFileName(self, 'Select image', 'c\\', 'Images (*.png *.jpg *.jpeg)')

        imagePath = fileName[0]
        pixmap = QPixmap(imagePath)
        self.label.setPixmap(QPixmap(pixmap))
        self.label.setScaledContents(True)
        self.label.setFixedSize(700, 650)


text = "Print test"
App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())
