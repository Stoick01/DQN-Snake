import pygame
import torch
import os
import shutil

from snake import Game

class Play():

    def __init__(self):
        self.window = pygame.display.set_mode((270, 270))

        self.game = Game(270, 270, 9, self.window, 0, 0)
        self.model = self.load_model()


    def load_model(self):
        try:
            f = torch.load("best.pth")
        except:
            f = None
        return f

    def run(self):
        clock = pygame.time.Clock()
        while True:
            pygame.time.delay(50)
            clock.tick(10)

            self.window.fill((0, 0, 0))

            self.game.game_loop(train=False, model=self.model)

            pygame.display.update()

                
if __name__ == "__main__":
    p = Play()
    p.run()