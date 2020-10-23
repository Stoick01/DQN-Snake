import pygame
import torch
import os
import shutil
import glob
from PIL import Image

from snake import Game

class Play():

    def __init__(self):
        self.window = pygame.display.set_mode((270, 270))

        self.game = Game(270, 270, 9, self.window, 0, 0)
        self.model = self.load_model()
        if not os.path.exists('out'):
            os.makedirs('out')


    def load_model(self):
        try:
            f = torch.load("best.pth")
        except:
            f = None
        return f

    def run(self):
        clock = pygame.time.Clock()
        cicle = 0
        inf = 0
        while True:
            pygame.time.delay(1)
            clock.tick(100000)

            self.window.fill((0, 0, 0))

            self.game.game_loop(train=False, model=self.model)

            pygame.display.update()

            pygame.image.save(self.window, f'out/screenshoot_{cicle:03d}.jpg')
            cicle += 1
            inf += 1
            if inf == 100:
                cicle = 0
                inf = 0
                shutil.rmtree('out')
                os.makedirs('out')
                self.game.restart()

            if self.game.reward == 0:
                print(self.game.points_ls[-1])
                if self.game.points_ls[-1] >= 18:
                    fp_in = 'out/screenshoot_*.jpg'
                    fp_out = 'sample.gif'

                    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
                    img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=(cicle+1)*4, loop=1)
                    exit()
                else:
                    cicle = 0
                    inf = 0
                    shutil.rmtree('out')
                    os.makedirs('out')

                
if __name__ == "__main__":
    p = Play()
    p.run()