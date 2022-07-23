import numpy as np
import tensorflow as tf
import time
import keyboard
import pyautogui
from gym import Env
from gym.spaces import Discrete, Box, Dict

from utils.helper import Decks, GameHelper, Recorder




class ClashRoyaleEnv(Env):
    
    def __init__(self):
        self.action_space = Dict({"card_choice": Discrete(5),
                                 "position" : Box(low= np.array([35, 95]), high=np.array([405, 605]), shape =(2,), dtype = int)}
                                 )
        self.observation_space= Dict({"image": Box(low=0, high=1, shape=(208, 160, 1), dtype=float),
                                      'features': Box(low=0, high=1, shape=(104,), dtype=float)})
        self.helper = GameHelper()
        self.deck = Decks()
        self.recorder = Recorder()
        self.battle_pos = (350, 600)
        self.dm_menu_pos = (430, 115)
        self.tc_pos = (280, 300)
        self.ok_pos = (330, 505)
        
    def step(self, action):
        #time.sleep(0.5)
        card_choice = action[0][0]
        if card_choice == 1:
            keyboard.press_and_release('1')
            
        if card_choice == 2:
            keyboard.press_and_release('2')
            
        if card_choice == 3:
            keyboard.press_and_release('3')
            
        if card_choice == 4:
            keyboard.press_and_release('4')
            
        time.sleep(0.15)
        
        if card_choice != 0:
            position = action[1].reshape(4, 2)[card_choice-1]
            print(position)
            pyautogui.click(tuple(np.abs(position)))
            
        pos = tuple(position) if card_choice != 0 else (0, 0)   
        self.recorder((card_choice, pos), 'action')
        
        img_state = self.helper.screenshot()
        reward, done = self.reward_function(img_state)
        card_state = self.helper.get_card_features(img_state, self.deck)
        img_state = self.helper.add_imageProc(img_state)
        states = [img_state, card_state]
        
        states = self.adj_observation(states)
        return states, reward, done
        
        
    def reset(self):
        ''' Takes us back to the home screen
        '''
        time.sleep(0.5)
        pyautogui.click(232, 740)      
        
    def reward_function(self, img):
        reward = 0
        done = False
        
        return reward, done
        
    def start(self):
        '''Takes us from the home screen into the game
        '''
        time.sleep(1)
        screen = self.helper.screenshot()
        
        pyautogui.click(self.dm_menu_pos)
        time.sleep(0.2)
        pyautogui.click(self.tc_pos)
        time.sleep(0.2)
        pyautogui.click(self.ok_pos)
        time.sleep(0.3)
        screen = self.helper.screenshot()
        time.sleep(0.3)
        while True:
            screen = self.helper.screenshot()
            if self.helper.on_load_screen(screen): ...
            else:
                time.sleep(5)
                screen = self.helper.screenshot()
                break
            
        return self.helper.add_imageProc(screen), self.helper.get_card_features(screen, self.deck)
            
        
    def adj_observation(self, states):
        converted_states = []
        for state in states:
            converted_states.append(tf.convert_to_tensor(state, dtype=tf.float32))
            
        return converted_states