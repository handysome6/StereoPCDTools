import sys
from PySide6.QtWidgets import QApplication
from .gui import StereoPcdGeneratorUI

def main():
    app = QApplication(sys.argv)
    window = StereoPcdGeneratorUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()