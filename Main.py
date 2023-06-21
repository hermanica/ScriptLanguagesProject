from nba_data_scraper import NBADataScraper

def main():
    url = 'https://www.basketball-reference.com/leagues/NBA_2016_games.html'
    scraper = NBADataScraper(url)
    games_data = scraper.scrape_games_data()

    # 
    for game_data in games_data:
        print(game_data)

if __name__ == '__main__':
    main()
