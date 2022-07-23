import mss
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.config import INPUT_DIM, READER, NORMALIZE, GAME_MON


class Recorder:
    
    def __init__(self):
        self.__past_action: tuple = None
        self.__past_friendly_towers: int = 3
        self.__past_enemy_towers: int = 3
        self.__past_elixir: int = 5
        self.__past_time = 180
        
    def __call__(self, event, event_name):
        if event_name == 'action':
            self.record_action(event)
        elif event_name == 'towers':
            self.record_friendly_towers(event[0])
            self.record_enemy_towers(event[1])
        elif event_name == 'elixir':
            self.record_elixir(event)
        elif event_name == 'time':
            self.record_time(event)
    
    def record_action(self, action):
        self.__past_action = action
        
    def record_friendly_towers(self, friendly_towers: int):
        self.__past_friendly_towers = friendly_towers
        
    def record_enemy_towers(self, enemy_towers: int):
        self.__past_enemy_towers = enemy_towers
        
    def record_elixir(self, elixir: int):
        self.__past_elixir = elixir
        
    def record_time(self, time: int):
        self.__past_time = time
        
    def get_past_towers(self):
        return self.past_friendly_towers, self.past_enemy_towers
    
    @property
    def past_time(self): return self.__past_time
    
    @property
    def past_elixir(self): return self.__past_elixir
    
    @property
    def past_action(self): return self.__past_action
    
    @property
    def past_friendly_towers(self): return self.__past_friendly_towers
    
    @property
    def past_enemy_towers(self): return self.__past_enemy_towers
    
    @property
    def past_towers(self): return self.__past_friendly_towers, self.__past_enemy_towers
    
    
class Decks:
    
    def __init__(self):
        self.card_id = {"giant": cv2.imread(r"E:\RL\Clash\assets\giant_id.png", 0),
                        "fireball": cv2.imread(r"E:\RL\Clash\assets\fireball_id.png", 0),
                        "arrows": cv2.imread(r"E:\RL\Clash\assets\arrows_id.png", 0),
                        "goblins": cv2.imread(r"E:\RL\Clash\assets\goblins_id.png", 0),
                        "knight": cv2.imread(r"E:\RL\Clash\assets\knight_id.png", 0),
                        "minions": cv2.imread(r"E:\RL\Clash\assets\minions_id.png", 0),
                        "prince": cv2.imread(r"E:\RL\Clash\assets\prince_id.png", 0),
                        "archers": cv2.imread(r"E:\RL\Clash\assets\archers_id.png", 0),}

        card_stats = pd.read_csv(r"E:\RL\Clash Royale\workspace\card_stats\cards.txt", index_col="name")
        self.card_stats = pd.get_dummies(card_stats)

        
        self.tower_stats = pd.read_csv(r"E:\RL\Clash Royale\workspace\card_stats\towers.txt", index_col="type")
        
    def get_curr_hand(self, img):
        curr_hand = []
        for card, template in self.card_id.items():
            res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
            threshold = 0.7
            loc = np.where( res >= threshold)
            if len(loc[0]) == 0: ...
            else: 
                curr_hand.append(card)
                
        return curr_hand
    
    
class GameHelper:


    def __init__(self):
        self.time = 180
        self.curr_elixir = 5
        self.scaler = StandardScaler()
        self.tower_id = {"king_t": cv2.imread(r"E:\RL\Clash\assets\kingt_id.png", 0),
                            "princess_t": cv2.imread(r"E:\RL\Clash\assets\princesst_id.png", 0),
                            "eking_t": cv2.imread(r"E:\RL\Clash\assets\ekingt_id.png", 0),
                            "eprincesst": cv2.imread(r"E:\RL\Clash\assets\eprincesst_id.png", 0)
                            }
       
    def screenshot(self, unprocessed: bool =False):
        with mss.mss() as sct:
            img = np.array(sct.grab(GAME_MON))

        if unprocessed:
            return self.process_ocr(img)
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
    

    def get_card_features(self,img, deck: Decks):
        card_feats = np.zeros((4, 24))
        tower_feats = deck.tower_stats.values
        tower_feats = self.scaler.fit_transform(tower_feats).flatten()
        cards = deck.get_curr_hand(img)
        for i,card in enumerate(cards):
            card_feats[i] = deck.card_stats.loc[card].values
        card_feats = self.scaler.fit_transform(card_feats)
        card_feats = card_feats.flatten()
        all_feats = np.concatenate((card_feats, tower_feats))
        return all_feats
        
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