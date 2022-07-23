import json
import numpy as np
import pandas as pd
import requests


class ClashScraper:
    
    def __init__(self, cards_filename: str):
        self.cards = self.get_card_names(cards_filename)
        self.web_host = "https://statsroyale.com/card/"
        
    @staticmethod    
    def get_card_names(json_file: str) -> list[str]:
        all_cards = []
        with open(json_file, 'r') as j:
            contents = json.loads(j.read())

        for card in contents['items']:
            all_cards.append(card['name'])

        return all_cards
    
    def set_index(self, card: str, df: pd.DataFrame) -> pd.DataFrame:
        card_col= pd.Series([card]* df.shape[0])
        df["name"] = card_col
        df.set_index("name", inplace=True)
        return df
        
        
    
    def get_card_stats(self, card: str) -> None:
        response = requests.get(self.web_host+card.replace(" ", "+"))
        data = pd.read_html(response.text)[0]
        self.set_index(card, data)
        data.to_csv(f'./images/{card}/{card}.csv')
        
    def run(self):
        found = 0
        error = 0
        
        for card in self.cards:
            try:
                self.get_card_stats(card)
                print('finished', card)
                found += 1
            except Exception as e:
                print('could not find table for spell', card)
                error += 1
            
        print('done')
        print(f'found {found} cards and could not find tables for {error} cards')
    
    
    
    
    
class ClashDataLoader:
    
    def __init__(self):
        self.cards = ClashScraper.get_card_names('all_cards.json')
        
        
    def load_data(self, as_array = False) -> pd.DataFrame:
        main_df = pd.DataFrame()
        try:
            for card in self.cards:
                df = pd.read_csv((f'./images/{card}/{card}.csv'))
                main_df = pd.concat([main_df, df])
                
        except FileNotFoundError: #spell cards that could not be scraped
            pass
        
        main_df = self._clean_data(main_df)
        
        if as_array:
            return main_df.values
        
        return main_df
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        
        data.fillna(0, inplace=True)
        arr = data[["name", "Card Level"]].values.T
        tuples = list(zip(*arr))
        index = pd.MultiIndex.from_tuples(tuples)
        data.set_index(index, inplace=True)
        data.drop(['Card Level', 'name'], axis = 1, inplace=True)
        data.sort_index(inplace=True)
        
        return data
    

    
    
def main():
    scraper = ClashScraper('all_cards.json')
    scraper.run()
    
if __name__ == '__main__':
    main()