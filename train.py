import os
import shutil
import datetime

import pygame
import torch

from snake import Game

class Play():

    def __init__(self):
        self.window = pygame.display.set_mode((270, 270))

        self.game = Game(270, 270, 9, self.window, 0, 0)

        self.model = self.load_model()

        self.cnt = 0

        if self.model != None:
            self.game.agent.update_model(self.model)
            self.game.agent.update_tgt(self.model)

    def save_model(self):
        print("Saving model")
        torch.save(self.game.agent.tgt, "best.pth")

    def load_model(self):
        try:
            f = torch.load("best.pth")
        except:
            f = None
        return f

    def run(self):
        clock = pygame.time.Clock()
        while True:
            pygame.time.delay(1)
            clock.tick(1000000)

            self.window.fill((0, 0, 0))

            self.game.game_loop(train=True)

            if self.game.agent.tgt_updated:
                self.cnt += 1
                print(self.cnt, " Target model updated:", self.game.get_average_reward())
                self.game.agent.tgt_updated = False
                self.save_model()


            if self.cnt == 500:
                exit()

            pygame.display.update()

                
if __name__ == "__main__":
    p = Play()
    p.run()