import numpy as np
import mss
import uuid
import keyboard
import cv2


MON = {'top': 150, 'left': 0, 'width': 420, 'height': 600} #game screen area

img_path = './images/all/'


def screenshot(filename, as_array = False):
    observations = []
    checkpoint = 10000
    replay_mon= {'top': 150, 'left': 0, 'width': 470, 'height': 600}
    while True:
        with mss.mss() as sct:
            if keyboard.is_pressed('s'):
                print("grabbing screenshots")
                while True:
                    ##Continuously take screenshots
                    img =  np.array(sct.grab(replay_mon))
                        
                    cv2.imwrite(filename +f'\{uuid.uuid1()}'+'.png', img)
                    if keyboard.is_pressed('p'): #pause inbetween games
                        print("paused..")
                        break
                
            if keyboard.is_pressed('b'):
                return observations
            
            if len(observations) == checkpoint:
                if as_array:
                    np.savez_compressed(f"{filename}", obs=observations)
                    print("saving..")
                break
            
            

                
                
def main():
    ...
    

if __name__ == '__main__':
    main()