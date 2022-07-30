import mss
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.config import INPUT_DIM, READER, NORMALIZE, GAME_MON
from card_info.card_stats_scraper import ClashDataLoader


class Recorder:
    
    def __init__(self):
        self.__past_params = {'enemy_positions':[], 'elixir':5, 'time': 180, 'tower_count': (3, 3)}
        self.__past_action = (0, (0, 0))
        
    def __call__(self, params):
        self.__past_params = params
        
    def save_action(self, action):
        self.__past_action = action
    
    @property
    def past(self): return self.__past_params
    
    @property
    def past_action(self): return self.__past_action  
    
    
class Decks:
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.card_id = {"giant": cv2.imread("assets/giant_id.png",0),
                        "fireball": cv2.imread("assets/fireball_id.png",0),
                        "arrows": cv2.imread("assets/arrows_id.png",0),
                        "goblins": cv2.imread("assets/goblins_id.png",0),
                        "knight": cv2.imread("assets/knight_id.png",0),
                        "minions": cv2.imread("assets/minions_id.png",0),
                        "prince": cv2.imread("assets/prince_id.png",0),
                        "archers": cv2.imread("assets/archers_id.png",0),}

        deck_stats = pd.read_csv("card_info/cards.txt", index_col="name")
        self.deck_stats = pd.get_dummies(deck_stats)
        self.card_stats = ClashDataLoader().load_data()
        self.tower_stats = pd.read_csv("card_info/towers.txt", index_col="type")
        self.grayscale = lambda x: cv2.cvtColor(x, cv2.COLOR_BGRA2GRAY)
        
    def get_curr_hand(self, img):
        curr_hand = []
        img = self.grayscale(img)
        for card, template in self.card_id.items():
            res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where( res >= threshold)
            if len(loc[0]) == 0: ...
            else: 
                curr_hand.append(card)
                
        return curr_hand
    
    def get_card_features(self, img, params):
        
        time = params['time']
        elixir = params['elixir']
        enemy_positions = params['enemy_positions']
        #Filling up max_enemies array
        max_enemies = np.zeros((30, 2)) #able to recognize a maximum of 30 enemies troops
        if len(enemy_positions) > 0:
            for i in range(len(enemy_positions)):
                if i == 30:
                    break
                max_enemies[i] = enemy_positions[i]
        max_enemies = self.scaler.fit_transform(max_enemies).flatten()
        #
        tower_feats = self.tower_stats.values
        tower_feats = self.scaler.fit_transform(tower_feats).flatten()
        #
        card_feats = np.zeros((4, 24))
        cards = self.get_curr_hand(img)
        print(cards)
        for i,card in enumerate(cards):
            card_feats[i] = self.deck_stats.loc[card].values

        card_feats = self.scaler.fit_transform(card_feats).flatten()
        #
        all_feats = np.concatenate((card_feats, tower_feats, max_enemies))
        return all_feats
    
class GameHelper:


    def __init__(self):
        self.time = 180
        self.curr_elixir = 5
        self.scaler = StandardScaler()
        self.to_rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        self.tower_id = {"princess_t": (cv2.imread("assets/princesst_id.png"), .70),
                        "eprincesst": (cv2.imread("assets/eprincesst_id.png"), .75)
                        }
        self.enemy_level_id = {1: (cv2.imread("assets/red-level01.png"), .75),
                               2: (cv2.imread("assets/red-level02.png"), .75),
                               3: (cv2.imread("assets/red-level03.png"), .78),
                               4: (cv2.imread("assets/red-level04.png"), .75),
                               5: (cv2.imread("assets/red-level05.png"), .78),
                               6: (cv2.imread("assets/red-level06.png"), .75),
                               7: (cv2.imread("assets/red-level07.png"), .78),
                               8: (cv2.imread("assets/red-level08.png"), .75),}
       
    def screenshot(self, unprocessed: bool =False):
        with mss.mss() as sct:
            img = np.array(sct.grab(GAME_MON))

        if unprocessed:
            return img
        img = GameHelper.process_image(img)
        #img = GameHelper.add_imageProc(img)
        return img
    
    def process_ocr(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        #img = cv2.Canny(img, 50, 100)
        return img

    @staticmethod 
    def process_image(image):

        proc_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #proc_image = cv2.GaussianBlur(proc_image, (7, 7), 0)
        #proc_image = cv2.Canny(proc_image, 50, 100)
        proc_image = cv2.resize(proc_image, (INPUT_DIM[1], INPUT_DIM[0]))
        #proc_image = proc_image.reshape(INPUT_DIM[0], INPUT_DIM[1], 1)
        return proc_image
    
    @staticmethod
    def add_imageProc(image):
        image= image[:172, :]
        image  = NORMALIZE(image)
        image = np.expand_dims(cv2.resize(image, (160, 208)), -1)
        return image


    def read_time(self, img) -> int:
        time_img = img[20:70, 370:500] #cropping for time portion (gotten through experiments)
        time = READER.readtext(time_img)
        try:
            time: str = time[0][1]

            if ':' in time:
                time = time.split(':')
            else:
                time = time.split('.')

                time = int(time[0])*60 + int(time[1])
                self.time = time
        except:
            time = self.time
        
        return time
    
    def tower_count(self, img):
        friendly_count = 1
        enemy_count = 1
        for tower,(template, threshold) in self.tower_id.items():
            res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
            loc = np.where( res >= threshold)
            if len(loc[0]) == 0: ...
            else: 
                new_loc = self.check_distance(loc)
                if tower.startswith("e"):
                    enemy_count += len(new_loc)
                else: 
                    friendly_count += len(new_loc)
                    
        return friendly_count, enemy_count

    def read_elixir(self, img) -> int:
        elixir_img = img[780: 812, 115:150]
        elixir = READER.readtext(elixir_img)
        try:
            elixir = int(elixir[0][1])
            self.curr_elixir = elixir
        except:
            elixir = self.curr_elixir
        return elixir
    
    def get_state_params(self, img):
        img = img[:,:,:3] # removing alpha channel
        img = self.to_rgb(img)
        params = {}
        enemy_positions = self.find_enemies(img)
        elixir = self.read_elixir(img)
        time = self.read_time(img)
        tower_count = self.tower_count(img)
        
        params['enemy_positions'] = enemy_positions
        params['tower_count'] = tower_count
        params['elixir'] = elixir
        params['time'] = time
        
        return params
        
    @staticmethod
    def on_home_screen(img) -> bool:
        template = cv2.imread('assets/home_id.png', 0)
        res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.9
        loc = np.where( res >= threshold)
        if len(loc[0]) == 0: return False
        else: return True
        
    @staticmethod
    def on_party_screen(img):
        template = cv2.imread('assets/party_id.png', 0)
        res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.9
        loc = np.where( res >= threshold)
        if len(loc[0]) == 0: return False
        else: return True
        
    @staticmethod
    def on_load_screen(img):
        template = cv2.imread('assets/load_id.png', 0)
        res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.9
        loc = np.where( res >= threshold)
        if len(loc[0]) == 0: return False
        else: return True
        
    @staticmethod
    def on_end_game_screen(img):
        template = cv2.imread('assets/end_game_id.png', 0)
        res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.6
        loc = np.where( res >= threshold)
        if len(loc[0]) == 0: return False
        else: return True
        
    def find_enemies(self, img):
        enemies = []
        for _, (template, threshold) in self.enemy_level_id.items():
            template = self.to_rgb(template)
            res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
            loc = np.where( res >= threshold)
            if len(loc[0]) <= 1: ...
            else:
                lvl_enemies = self.check_distance(loc)
                enemies += lvl_enemies
            
        return enemies
    
    def check_distance(self, loc: tuple):
        positions = []
        loc = list(loc)
        loc[0] = np.sort(loc[0])
        loc[1] = np.sort(loc[1])
        
        for i in range(len(loc[0])-1):
            dist = np.sqrt((loc[0][i+1] - loc[0][i])**2 + (loc[1][i+1] - loc[1][i])**2)
            if dist <= 2:
                ...
            else:
                positions.append((loc[0][i], loc[1][i]))
            if i == len(loc[0])-2:
                positions.append((loc[0][i+1], loc[1][i+1]))
        return positions
