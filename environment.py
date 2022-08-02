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
                                 "position" : Box(low= np.array([45, 150]), high=np.array([405, 600]), shape =(2,), dtype = int)}
                                 )
        self.observation_space= Dict({"image": Box(low=0, high=1, shape=(208, 160, 1), dtype=float),
                                      'features': Box(low=0, high=1, shape=(164,), dtype=float)})
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
            diff = self.action_space['position'].high - self.action_space['position'].low
            offset = diff * position
            position = self.action_space['position'].low + offset
            pyautogui.click(tuple(position))
            
        pos = tuple(position) if card_choice != 0 else (0, 0)   
        #self.recorder.save_action((card_choice, pos))
        
        img_state = self.helper.screenshot(unprocessed=True)
        
        params = self.helper.get_state_params(img_state)
        params["action"] = (card_choice, pos)
        card_state = self.deck.get_card_features(img_state, params)
        reward = self.reward_function(params)
        
        img_state = self.helper.process_image(img_state)
        done = self.helper.on_end_game_screen(img_state)
        img_state = self.helper.add_imageProc(img_state)
        states = [img_state, card_state]
        
        states = self.adj_observation(states)
        return states, reward, done
        
        
    def reset(self):
        ''' Takes us back to the home screen
        '''
        time.sleep(5.5)
        pyautogui.click(232, 740)      
        
    def reward_function(self, params):
        reward = 0
        past_params = self.recorder.past
        troop_reward = self.troop_reward(params, past_params)
        tower_reward = self.tower_reward(params, past_params)
        card_reward = self.card_reward(params, past_params)
            
        reward = troop_reward + tower_reward + card_reward
        self.recorder(params)
        
        return reward
    
    def troop_reward(self, params, past_params):
        reward = 0
        enemy_positions = params['enemy_positions']
        n_enemies = len(enemy_positions)
        past_n_enemies = len(past_params['enemy_positions'])
        if n_enemies < past_n_enemies:
            reward  = 5
        elif n_enemies == past_n_enemies:
            reward = -10
            
        return reward
        
    
    def tower_reward(self, params, past_params):
        reward = 0
        friend_tower, enemy_tower = params['tower_count']
        past_friend_tower, past_enemy_tower = past_params['tower_count']
        if past_friend_tower > friend_tower:
            reward = -30
            
        if past_enemy_tower > enemy_tower:
            reward = 20
            
        return reward
        
    def card_reward(self, params, past_params):
        reward = 0
        past_action = past_params['action']
        action = params['action']
        if action[0] != 0 and past_action[0] != 0:
            midXY = np.array((230, 400))
            currXY = np.array(action[1])
            distance = np.linalg.norm(currXY - midXY)
            reward = -0.2 * distance
            
        return reward
        
        
    def start(self):
        '''Takes us from the home screen into the game
        '''
        time.sleep(1)
        screen = self.helper.screenshot()
        
        pyautogui.click(self.dm_menu_pos)
        time.sleep(0.4)
        pyautogui.click(self.tc_pos)
        time.sleep(0.6)
        pyautogui.click(self.ok_pos)
        time.sleep(0.6)
        screen = self.helper.screenshot()
        time.sleep(0.3)
        while True:
            screen = self.helper.screenshot()
            if self.helper.on_load_screen(screen): ...
            else:
                time.sleep(5)
                screen = self.helper.screenshot(unprocessed=True)
                break
        params = self.helper.get_state_params(screen)
        card_feats = self.deck.get_card_features(screen, params)
        screen = self.helper.process_image(screen)
        
        return self.helper.add_imageProc(screen), card_feats
            
        
    def adj_observation(self, states):
        converted_states = []
        for state in states:
            converted_states.append(tf.convert_to_tensor(state, dtype=tf.float32))
            
        return converted_states
