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
        # the average stats are calculated from the previous 4 matches
        rolling_stats = group[self.cols].rolling(4, closed='left').mean()
        group[self.new_cols] = rolling_stats
        group = group.dropna(subset=self.new_cols)
        return group
        
    def apply_rolling_averages(self):
        grouped_matches = self.matches.groupby("team")
        self.matches = grouped_matches.apply(self.rolling_averages).droplevel('team')
        self.matches.index = range(self.matches.shape[0])
        
    def train_initial_model(self):
        train = self.matches[self.matches["date"] < '2023-01-01']
        self.rf_initial.fit(train[self.predictors], train["target"])
        
    def train_rolling_model(self):
        train = self.matches[self.matches["date"] < '2023-01-01']
        self.rf_rolling.fit(train[self.predictors + self.new_cols], train["target"])

    def train_merged_dataframe(self):
        train = self.matches[self.matches["date"] < '2023-01-01']
        self.rf_merged = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
        self.rf_merged.fit(train[self.predictors + self.new_cols], train["target"])
        
    def make_predictions(self, model):
        test = self.matches[self.matches["date"] > '2023-01-01']
        if model == "initial":
            rf = self.rf_initial
            predictors = self.predictors
        else:
            rf = self.rf_rolling
            predictors = self.predictors + self.new_cols
        preds = rf.predict(test[predictors])
        combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
        error = precision_score(test["target"], preds, average="weighted")
        accuracy = error * 100
        return combined, accuracy

    
    def get_predicted_wins(self):
        test = self.matches[self.matches["date"] > '2023-01-01']
        predictors = self.predictors + self.new_cols
        preds = self.rf_rolling.predict(test[predictors])
        predicted_wins = test[preds == 1]["team"].value_counts().to_dict()
        return predicted_wins
    
    # def get_projected_points(self):
    #     test = self.matches[self.matches["date"] > '2023-01-01'].copy()
    #     predictors = self.predictors + self.new_cols
    #     preds = self.rf_rolling.predict(test[predictors])
    #     test["predicted"] = preds
    #     test["points"] = 0

    #     # Calculate points based on predictions
    #     test.loc[(test["predicted"] == 1), "points"] = 3
    #     test.loc[(test["predicted"] == 0) & (test["target"] == 0), "points"] = 1

    #     # Calculate projected points for each team
    #     projected_points = test.groupby("team")["points"].sum().astype(int)
    #     projected_points = projected_points.sort_values(ascending=False)
    #     return projected_points

        
    def run_simulation(self):
        self.load_data()
        self.clean_data()
        self.apply_rolling_averages()

        # Save modified matches dataFrame to CSV
        self.matches.to_csv("modified_matches.csv", index=False)

        # Initial model
        print("Initial Model:")
        self.train_initial_model()
        combined_initial, error_initial = self.make_predictions("initial")
        print("Precision Score (Initial Model): {:.2f}%".format(error_initial))
        combined_initial = combined_initial.merge(self.matches[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
        print(combined_initial)

        # Rolling model
        print("\nRolling model:")
        self.train_rolling_model()
        combined_rolling, error_rolling = self.make_predictions("rolling")
        print("Precision score (Rolling model): {:.2f}%".format(error_rolling))
        combined_rolling = combined_rolling.merge(self.matches[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
        print(combined_rolling)

        # Predicting one side to win and the other side to lose/draw
        print("\nPredicting one side to win and the other side to lose/draw:")
        combined_merged = combined_rolling.merge(combined_rolling, left_on=["date", "team"], right_on=["date", "opponent"])
        predictions = combined_merged[(combined_merged["predicted_x"] == 1) & (combined_merged["predicted_y"] == 0)]
        predictions_count = predictions["actual_x"].value_counts()
        print("Predicted number of wins:")
        print(predictions_count)
        print("Precision Score (Predicting one side to win and the other side to lose/draw): {:.2f}%".format(predictions_count[1] / predictions_count.sum() * 100))

        # Predicted number of wins
        print("\nPredicted number of wins:")
        predicted_wins = self.get_predicted_wins()
        for team, wins in predicted_wins.items():
            print(f"{team}: {wins}")



if __name__ == '__main__':
    predictor = FootballMatchPredictor("matches.csv")
    predictor.run_simulation()
