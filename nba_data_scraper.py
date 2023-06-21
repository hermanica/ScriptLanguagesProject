import os
import time
import asyncio
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

SEASONS = list(range(2016, 2023))
DATA_DIR = "data"
STANDINGS_DIR = os.path.join(DATA_DIR, "standings")
SCORES_DIR = os.path.join(DATA_DIR, "scores")


class NBADataScraper:
    def __init__(self):
        pass

    def get_html(self, url, selector, sleep=5, retries=3):
        html = None
        for i in range(1, retries + 1):
            time.sleep(sleep * i)
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch()
                    page = browser.new_page()
                    page.goto(url)
                    print(page.title())
                    html = page.inner_html(selector)
            except PlaywrightTimeout:
                print(f"Timeout error on {url}")
                continue
            else:
                break
        return html

    def scrape_season(self, season):
        url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
        html = self.get_html(url, "#content .filter")

        soup = BeautifulSoup(html, 'html.parser')
        links = soup.find_all("a")
        standings_pages = [f"https://www.basketball-reference.com{l['href']}" for l in links]

        for url in standings_pages:
            save_path = os.path.join(STANDINGS_DIR, url.split("/")[-1])
            if os.path.exists(save_path):
                continue

            html = self.get_html(url, "#all_schedule")
            with open(save_path, "w+") as f:
                f.write(html)

    def scrape_game(self, standings_file):
        with open(standings_file, 'r') as f:
            html = f.read()

        soup = BeautifulSoup(html, 'html.parser')
        links = soup.find_all("a")
        hrefs = [l.get('href') for l in links]
        box_scores = [f"https://www.basketball-reference.com{l}" for l in hrefs if l and "boxscore" in l and '.html' in l]

        for url in box_scores:
            save_path = os.path.join(SCORES_DIR, url.split("/")[-1])
            if os.path.exists(save_path):
                continue

            html = self.get_html(url, "#content")
            if not html:
                continue
            with open(save_path, "w+") as f:
                f.write(html)

    def scrape_data(self):
        for season in SEASONS:
            self.scrape_season(season)

        standings_files = os.listdir(STANDINGS_DIR)
        for season in SEASONS:
            files = [s for s in standings_files if str(season) in s]

            for f in files:
                filepath = os.path.join(STANDINGS_DIR, f)
                self.scrape_game(filepath)


# testing datascraper class
scraper = NBADataScraper()
scraper.scrape_data()