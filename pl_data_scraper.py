import requests
from bs4 import BeautifulSoup
import pandas as pd
import time


class PLDataScraping:
    def __init__(self):
        self.standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
        self.years = list(range(2022, 2020, -1))
        self.all_matches = []
    
    # this method scrapes the Premier League data from the FBRef website for seasons 2021-22 and 2022-23
    def scrape_data(self):
        for year in self.years:
            data = requests.get(self.standings_url)
            soup = BeautifulSoup(data.text, features="html.parser")
            standings_table = soup.select('table.stats_table')[0]

            links = [l.get("href") for l in standings_table.find_all('a')]
            links = [l for l in links if '/squads/' in l]
            team_urls = [f"https://fbref.com{l}" for l in links]
    
            previous_season = soup.select("a.prev")[0].get("href")
            self.standings_url = f"https://fbref.com{previous_season}"
    
            for team_url in team_urls:
                team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
                data = requests.get(team_url)
                matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
                soup = BeautifulSoup(data.text, features="html.parser")
                links = [l.get("href") for l in soup.find_all('a')]
                links = [l for l in links if l and 'all_comps/shooting/' in l]
                data = requests.get(f"https://fbref.com{links[0]}")
                shooting = pd.read_html(data.text, match="Shooting")[0]
                shooting.columns = shooting.columns.droplevel()
                try:
                    team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
                except ValueError:
                    continue
                team_data = team_data[team_data["Comp"] == "Premier League"]
        
                team_data["Season"] = year
                team_data["Team"] = team_name
                self.all_matches.append(team_data)
                time.sleep(1)
    
    def export_to_csv(self, filename):
        match_df = pd.concat(self.all_matches)
        match_df.columns = [c.lower() for c in match_df.columns]
        match_df.to_csv(filename, index=False)


# Usage:
scraper = PLDataScraping()
scraper.scrape_data()
scraper.export_to_csv("matches.csv")

