import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtCore import Qt, pyqtSlot
from prediction import FootballMatchPredictor

class FootballMatchPredictorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Premier League Results Predictions")
        self.setGeometry(100, 100, 800, 600)

        self.title_label = QLabel("Premier League Results Predictions", self)
        self.title_label.setGeometry(20, 20, 760, 50)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold")

        self.initial_model_button = QPushButton("Initial Model", self)
        self.initial_model_button.setGeometry(250, 100, 300, 50)
        self.initial_model_button.clicked.connect(self.run_initial_model)

        self.rolling_model_button = QPushButton("Rolling Model", self)
        self.rolling_model_button.setGeometry(250, 200, 300, 50)
        self.rolling_model_button.clicked.connect(self.run_rolling_model)

        self.predict_win_button = QPushButton("Predict number of wins for each team", self)
        self.predict_win_button.setGeometry(250, 400, 300, 50)
        self.predict_win_button.clicked.connect(self.predict_win_loss)

        self.accuracy_label = QLabel(self)
        self.accuracy_label.setGeometry(20, 400, 760, 150)
        self.accuracy_label.setAlignment(Qt.AlignCenter)

        self.show()

    def update_accuracy(self, accuracy):
        self.accuracy_label.setText(f"Prediction Accuracy: {accuracy:.2f}%")

    @pyqtSlot()
    def run_initial_model(self):
        predictor.load_data()
        predictor.clean_data()
        predictor.apply_rolling_averages()
        predictor.train_initial_model()

        combined_initial, accuracy_initial = predictor.make_predictions("initial")
        self.update_accuracy(accuracy_initial)

    @pyqtSlot()
    def run_rolling_model(self):
        predictor.train_rolling_model()

        combined_rolling, accuracy_rolling = predictor.make_predictions("rolling") 
        self.update_accuracy(accuracy_rolling)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    predictor = FootballMatchPredictor("matches.csv")

    gui = FootballMatchPredictorGUI()

    sys.exit(app.exec())
