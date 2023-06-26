import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QStackedWidget, QHBoxLayout, QTableWidget, QTableWidgetItem, QTextEdit
from PyQt5.QtCore import Qt, pyqtSlot
from prediction import FootballMatchPredictor

# this class creates a widget for the first two options in the menu
class PredictionWindow(QWidget):
    def __init__(self, title, description, accuracy_score):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        title_label = QLabel(title, self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold")
        layout.addWidget(title_label, alignment=Qt.AlignTop)

        description_text_edit = QTextEdit(self)
        description_text_edit.setPlainText(description)
        description_text_edit.setReadOnly(True)
        description_text_edit.setFixedHeight(100)
        layout.addWidget(description_text_edit)

        accuracy_score_label = QLabel(f"Prediction accuracy: {accuracy_score:.2f}%", self)
        accuracy_score_label.setAlignment(Qt.AlignCenter)
        accuracy_score_label.setStyleSheet("font-size: 40px")
        layout.addWidget(accuracy_score_label)

        back_button = QPushButton("Back", self)
        back_button.clicked.connect(self.back_to_menu)
        layout.addWidget(back_button, alignment=Qt.AlignBottom)

    @pyqtSlot()
    def back_to_menu(self):
        gui.stacked_widget.setCurrentIndex(0)

# this class creates a widget for the last option in the menu
class WinsComparisonWindow(QWidget):
    def __init__(self, predicted_wins, actual_wins):
        super().__init__()

        layout = QHBoxLayout()
        self.setLayout(layout)

        predicted_table = QTableWidget(self)
        predicted_table.setColumnCount(2)
        predicted_table.setHorizontalHeaderLabels(['Team', 'Predicted Wins'])
        predicted_table.setRowCount(len(predicted_wins))

        for row, (team, wins) in enumerate(predicted_wins.items()):
            team_item = QTableWidgetItem(team)
            wins_item = QTableWidgetItem(str(wins))
            predicted_table.setItem(row, 0, team_item)
            predicted_table.setItem(row, 1, wins_item)

        layout.addWidget(predicted_table)

        actual_table = QTableWidget(self)
        actual_table.setColumnCount(2)
        actual_table.setHorizontalHeaderLabels(['Team', 'Actual Wins'])
        actual_table.setRowCount(len(actual_wins))

        for row, (team, wins) in enumerate(actual_wins.items()):
            team_item = QTableWidgetItem(team)
            wins_item = QTableWidgetItem(str(wins))
            actual_table.setItem(row, 0, team_item)
            actual_table.setItem(row, 1, wins_item)

        layout.addWidget(actual_table)

        back_button = QPushButton("Back", self)
        back_button.clicked.connect(self.back_to_menu)
        layout.addWidget(back_button)

    @pyqtSlot()
    def back_to_menu(self):
        gui.stacked_widget.setCurrentIndex(0)

# this class is responsible for the overall working of the GUI
class FootballMatchPredictorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Premier League results predictions")
        self.setGeometry(100, 100, 800, 600)

        self.stacked_widget = QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)

        self.create_initial_menu()
        self.create_accuracy_window()
        self.create_wins_comparison_window()

        self.stacked_widget.addWidget(self.initial_menu)
        self.stacked_widget.addWidget(self.accuracy_window)
        self.stacked_widget.addWidget(self.wins_comparison_window)

        self.stacked_widget.setCurrentIndex(0)

        self.show()

    def create_initial_menu(self):
        self.initial_menu = QWidget(self)

        # Outer layout for horizontal centering
        outer_layout = QHBoxLayout()
        self.initial_menu.setLayout(outer_layout)

        # Create a QVBoxLayout inside the QHBoxLayout for vertical centering
        layout = QVBoxLayout()
        outer_layout.addStretch()
        outer_layout.addLayout(layout)
        outer_layout.addStretch()

        title_label = QLabel("Premier League results predictions", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold")
        layout.addWidget(title_label)

        layout.addStretch()  # Add a stretch before the buttons to push them towards the center.

        initial_model_button = QPushButton("Random Forest model", self)
        initial_model_button.setFixedSize(400, 50)  # Set button width to 200 and height to 50.
        initial_model_button.clicked.connect(self.run_initial_model)
        layout.addWidget(initial_model_button)

        rolling_model_button = QPushButton("Model with rolling stats averages", self)
        rolling_model_button.setFixedSize(400, 50) 
        rolling_model_button.clicked.connect(self.run_rolling_model)
        layout.addWidget(rolling_model_button)

        predicted_wins_button = QPushButton("Predicted number of wins for each team", self)
        predicted_wins_button.setFixedSize(400, 50)  
        predicted_wins_button.clicked.connect(self.run_predicted_wins)
        layout.addWidget(predicted_wins_button)

        layout.addStretch()  # Add a stretch after the buttons to push them towards the center.
        layout.addStretch()


    def create_accuracy_window(self):
        self.accuracy_window = PredictionWindow("Prediction accuracy", "", 0)

    def create_wins_comparison_window(self):
        self.wins_comparison_window = WinsComparisonWindow({}, {})

    @pyqtSlot()
    def run_initial_model(self):
        predictor.load_data()
        predictor.clean_data()
        predictor.apply_rolling_averages()
        predictor.train_initial_model()

        _, accuracy_initial = predictor.make_predictions("initial")
        description = "This model of machine learning uses some part of the data to train itself. In our case, we use the matches before January 1st 2023 to train it. After that the model predicts the scores for games after that date and compares it to actual results. The accuracy of the model is shown below."
        self.accuracy_window = PredictionWindow("Random Forest model", description, accuracy_initial)
        self.stacked_widget.addWidget(self.accuracy_window)
        self.stacked_widget.setCurrentWidget(self.accuracy_window)

    @pyqtSlot()
    def run_rolling_model(self):
        predictor.train_rolling_model()

        _, accuracy_rolling = predictor.make_predictions("rolling")
        description = "This model is also based on Random Forest algorithm, but this time its performance is boosted, because we use rolling averages of the teams' stats from the last 4 games before January 1st. Because of that we're able to get higher accuracy, however the influence of current form might still make the predicted results quite unrealistic. The accuracy of the model is shown below."
        self.accuracy_window = PredictionWindow("Random Forest model using rolling averages", description, accuracy_rolling)
        self.stacked_widget.addWidget(self.accuracy_window)
        self.stacked_widget.setCurrentWidget(self.accuracy_window)


    @pyqtSlot()
    def run_predicted_wins(self):
        predicted_wins = predictor.get_predicted_wins()
        actual_wins = predictor.get_actual_wins()
        self.wins_comparison_window = WinsComparisonWindow(predicted_wins, actual_wins)
        self.stacked_widget.addWidget(self.wins_comparison_window)
        self.stacked_widget.setCurrentWidget(self.wins_comparison_window)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    predictor = FootballMatchPredictor("matches.csv")
    gui = FootballMatchPredictorGUI()

    sys.exit(app.exec())

