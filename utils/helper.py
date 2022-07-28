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
        self.card_id = {"giant": cv2.imread("assets/giant_id.png", 0),
                        "fireball": cv2.imread("assets/fireball_id.png", 0),
                        "arrows": cv2.imread("assets/arrows_id.png", 0),
                        "goblins": cv2.imread("assets/goblins_id.png", 0),
                        "knight": cv2.imread("assets/knight_id.png", 0),
                        "minions": cv2.imread("assets/minions_id.png", 0),
                        "prince": cv2.imread("assets/prince_id.png", 0),
                        "archers": cv2.imread("assets/archers_id.png", 0),}

        deck_stats = pd.read_csv("card_info/cards.txt", index_col="name")
        self.deck_stats = pd.get_dummies(deck_stats)
        self.card_stats = ClashDataLoader().load_data()
        self.tower_stats = pd.read_csv("card_info/towers.txt", index_col="type")
        self.grayscale = lambda x: np.expand_dims(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), -1)
        
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
        max_enemies = np.zeros((30, 2)) #able to recognize a maximum of 20 enemies troops
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
        for i,card in enumerate(cards):
            card_feats[i] = self.deck_stats.loc[card].values
        card_feats = self.scaler.fit_transform(card_feats).flatten()
        #
        add_feats = np.array([[time], [elixir]])
        add_feats = self.scaler.fit_transform(add_feats).flatten()
        all_feats = np.concatenate((card_feats, tower_feats, max_enemies, add_feats))
        return all_feats
    
class GameHelper:


    def __init__(self):
        self.time = 180
        self.curr_elixir = 5
        self.scaler = StandardScaler()
        self.tower_id = {"king_t": cv2.imread("assets/kingt_id.png", 0),
                            "princess_t": cv2.imread("assets/princesst_id.png", 0),
                            "eking_t": cv2.imread("assets/ekingt_id.png", 0),
                            "eprincesst": cv2.imread("assets/eprincesst_id.png", 0)
                            }
        self.enemy_level_id = {1: (cv2.imread("assets/red-level01.png"), .75),
                               2: (cv2.imread("assets/red-level02.png"), .72),
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

    @staticmethod 
    def read_time(img) -> int:
        time_img = img[100:200, 350:600] #cropping for time portion (gotten through experiments)
        #time_img = np.expand_dims(time_img, -1)
        time = READER.readtext(img)
        print(time[2])
        time: str = time[2][1]
        if ':' in time:
            time = time.split(':')
        else:
            time = time.split('.')
        time = int(time[0])*60 + int(time[1])
        return time
    
    def tower_count(self, img):
        friendly_count = 0
        enemy_count = 0
        for tower,template in self.tower_id.items():
            res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
            threshold = 0.7
            if tower == "eking_t":
                threshold = 0.85
            loc = np.where( res >= threshold)
            if len(loc[0]) == 0: ...
            else: 
                if tower.startswith("e"):
                    enemy_count += len(loc[0])
                else: 
                    friendly_count += len(loc[0])
                    
        return friendly_count, enemy_count

    def read_elixir(self, img) -> int:
        elixir_img = img[787: 810, 110:150]
        elixir = READER.readtext(elixir_img)
        try:
            elixir = int(elixir[0][1])
            self.curr_elixir = elixir
        except:
            elixir = self.curr_elixir
        return elixir
    
    def get_state_params(self, img):
        img = img[:,:,:3] # removing alpha channel
        params = {}
        enemy_positions = self.find_enemies(img)
        #elixir = self.read_elixir(img)
        #time = self.read_time(img)
        elixir = 10
        time = 179
        tower_count = (3, 2)#self.tower_count(img)
        
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
        to_bgr = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        enemies = []
        for level, (template, threshold) in self.enemy_level_id.items():
            #template = to_bgr(template)
            res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
            loc = np.where( res >= threshold)
            if len(loc[0]) == 0: ...
            else:
                lvl_enemies = self.check_distance(loc)
                enemies += lvl_enemies
            
        return enemies
    
    def check_distance(self, loc):
        enemy_positions = []
        for i in range(len(loc[0])-1):
            dist = np.sqrt((loc[0][i+1] - loc[0][i])**2 + (loc[1][i+1] - loc[1][i])**2)
            if dist <= 1:
                ...
            else:
                enemy_positions.append((loc[0][i], loc[1][i+1]))
                if i == len(loc[0])-2:
                    enemy_positions.append((loc[0][i+1], loc[1][i+1]))
        return enemy_positions
