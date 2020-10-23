import pygame
import torch

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

    def get_average(self, arr):
        s = 0.0
        for a in arr:
            s += a

        return s / len(arr)

    def run(self):
        clock = pygame.time.Clock()

        cicles = 0
        inf_loops = 0
        prev = 0

        while True:
            pygame.time.delay(1)
            clock.tick(1000000)

            self.window.fill((0, 0, 0))

            self.game.game_loop(train=False, model=self.model)

            if self.game.reward == 0:
                cicles += 1
                inf_loops = 0

            if inf_loops == 300 and prev == self.game.points:
                self.game.restart()
                cicles += 1
                inf_loops = 0

            if prev < self.game.points:
                prev = self.game.points

            inf_loops += 1

            if cicles == 1000:
                print(f'Max: {max(self.game.points_ls)}, Average: {self.get_average(self.game.points_ls)}')
                exit()


            pygame.display.update()

                
if __name__ == "__main__":
    p = Play()
    p.run()