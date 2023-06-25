import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

class FootballMatchPredictor:
    def __init__(self, data_file):
        self.data_file = data_file
        self.matches = None
        self.rf_initial = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
        self.rf_rolling = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
        self.predictors = ["venue_code", "opp_code", "hour", "day_code"]
        self.cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
        self.new_cols = [f"{c}_rolling" for c in self.cols]
        
    def load_data(self):
        self.matches = pd.read_csv(self.data_file, index_col=0)
        self.matches["date"] = pd.to_datetime(self.matches["date"])
        self.matches["target"] = (self.matches["result"] == "W").astype(int)
        self.matches["venue_code"] = self.matches["venue"].astype("category").cat.codes
        self.matches["opp_code"] = self.matches["opponent"].astype("category").cat.codes
        self.matches["hour"] = self.matches["time"].str.replace(":.+", "", regex=True).astype(int)
        self.matches["day_code"] = self.matches["date"].dt.dayofweek
        
    def clean_data(self):
        del self.matches["comp"]
        del self.matches["notes"]
        
    def rolling_averages(self, group):
        group = group.sort_values("date")
        rolling_stats = group[self.cols].rolling(3, closed='left').mean()
        group[self.new_cols] = rolling_stats
        group = group.dropna(subset=self.new_cols)
        return group
        
    def apply_rolling_averages(self):
        grouped_matches = self.matches.groupby("team")
        self.matches = grouped_matches.apply(self.rolling_averages).droplevel('team')
        self.matches.index = range(self.matches.shape[0])
        
    def train_initial_model(self):
        train = self.matches[self.matches["date"] < '2022-01-01']
        self.rf_initial.fit(train[self.predictors], train["target"])
        
    def train_rolling_model(self):
        train = self.matches[self.matches["date"] < '2022-01-01']
        self.rf_rolling.fit(train[self.predictors + self.new_cols], train["target"])
        
    def make_predictions(self, model):
        test = self.matches[self.matches["date"] > '2022-01-01']
        if model == "initial":
            rf = self.rf_initial
            predictors = self.predictors
        else:
            rf = self.rf_rolling
            predictors = self.predictors + self.new_cols
        preds = rf.predict(test[predictors])
        combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
        error = precision_score(test["target"], preds, average="weighted")
        return combined, error
    
    def get_predicted_wins(self):
        test = self.matches[self.matches["date"] > '2022-01-01']
        predictors = self.predictors + self.new_cols
        preds = self.rf_rolling.predict(test[predictors])
        predicted_wins = test[preds == 1]["team"].value_counts().to_dict()
        return predicted_wins
    
    def get_projected_points(self):
        test = self.matches[self.matches["date"] > '2022-01-01'].copy()
        predictors = self.predictors + self.new_cols
        preds = self.rf_rolling.predict(test[predictors])
        test["predicted"] = preds
        test["points"] = 0

        # Calculate points based on predictions
        test.loc[(test["predicted"] == 1), "points"] = 3
        test.loc[(test["predicted"] == 0) & (test["predicted"].eq(test["actual"])) & (test["predicted"].eq(0)), "points"] = 1

        # Calculate projected points for each team
        projected_points = test.groupby("team")["points"].sum().astype(int)
        projected_points = projected_points.sort_values(ascending=False)
        return projected_points

        
    def run_simulation(self):
        self.load_data()
        self.clean_data()
        self.apply_rolling_averages()
        
        # Initial Model
        print("Initial Model:")
        self.train_initial_model()
        combined_initial, error_initial = self.make_predictions("initial")
        print("Precision Score (Initial Model): {:.2f}%".format(error_initial * 100))
        combined_initial = combined_initial.merge(self.matches[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
        print(combined_initial)

        # Rolling Model
        print("\nRolling Model:")
        self.train_rolling_model()
        combined_rolling, error_rolling = self.make_predictions("rolling")
        print("Precision Score (Rolling Model): {:.2f}%".format(error_rolling * 100))
        combined_rolling = combined_rolling.merge(self.matches[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
        print(combined_rolling)

        # Predicting One Side to Win and the Other Side to Lose/Draw
        print("\nPredicting One Side to Win and the Other Side to Lose/Draw:")
        combined_merged = combined_rolling.merge(combined_rolling, left_on=["date", "team"], right_on=["date", "opponent"])
        predictions = combined_merged[(combined_merged["predicted_x"] == 1) & (combined_merged["predicted_y"] == 0)]
        predictions_count = predictions["actual_x"].value_counts()
        print("Predicted Number of Wins:")
        print(predictions_count)
        print("Precision Score (Predicting One Side to Win and the Other Side to Lose/Draw): {:.2f}%".format(predictions_count[1] / predictions_count.sum() * 100))
        
        # Predicted Number of Wins
        print("\nPredicted Number of Wins:")
        predicted_wins = self.get_predicted_wins()
        for team, wins in predicted_wins.items():
            print(f"{team}: {wins}")
        
        # Projected Points
        print("\nProjected Points:")
        projected_points = self.get_projected_points()
        for team, points in projected_points.items():
            print(f"{team}: {int(points)}")


if __name__ == '__main__':
    predictor = FootballMatchPredictor("matches.csv")
    predictor.run_simulation()




