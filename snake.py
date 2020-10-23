import math
import random
import itertools

import pygame
import torch

import numpy as np

from agent import DQNAgent, Model, StepData

class Cube():
    def __init__(self, start, settings, dirnx=1, dirny=0, color=(255, 0, 0)):
        self.SETTINGS = settings
        self.pos = start
        self.dirnx = dirnx
        self.dirny = dirny
        self.color = color

    def move(self, dirnx, dirny):
        self.dirnx = dirnx
        self.dirny = dirny
        self.pos = (self.pos[0] + self.dirnx, self.pos[1] + self.dirny)

    def draw(self, surface, eyes=False):
        offx = self.SETTINGS['ox']
        offy = self.SETTINGS['oy']
        dis = self.SETTINGS['sB']
        i = self.pos[0]
        j = self.pos[1]

        pygame.draw.rect(surface, self.color, (i * dis + 1 + offx, j * dis + 1 + offy, dis-2, dis-2))

        if eyes:
            center = dis // 2
            radius = 3
            circle1 = (i*dis + center - radius + offx, j*dis + center // 2 + offy)
            circle2 = (i*dis + center + radius + offx, j*dis + center // 2 + offy)
            pygame.draw.circle(surface, (0, 0, 0), circle1, radius)
            pygame.draw.circle(surface, (0, 0, 0), circle2, radius)

class Snake():
    def __init__(self, color, pos, settings):
        self.SETTINGS = settings
        self.body = []
        self.turns = {}

        self.color = color
        self.head = Cube(pos, settings)

        self.body.append(self.head)

        self.dirnx = 0
        self.dirny = 1

    def move(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()

            keys = pygame.key.get_pressed()

            if keys[pygame.K_LEFT] and self.head.dirnx != 1:
                self.dirnx = -1
                self.dirny = 0
            elif keys[pygame.K_RIGHT] and self.head.dirnx != -1:
                self.dirnx = 1
                self.dirny = 0
            elif keys[pygame.K_UP] and self.head.dirny != 1:
                self.dirnx = 0
                self.dirny = -1
            elif keys[pygame.K_DOWN] and self.head.dirny != -1:
                self.dirnx = 0
                self.dirny = 1

            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            pos = c.pos[:]

            if pos in self.turns:
                turn = self.turns[pos]
                c.move(turn[0], turn[1])

                if i == len(self.body) -1:
                    self.turns.pop(pos)
            else:
                if c.dirnx == -1 and c.pos[0] <= 0:
                    c.pos = (self.SETTINGS['r']-1, c.pos[1])
                elif c.dirnx == 1 and c.pos[0] >= self.SETTINGS['r']-1:
                    c.pos = (0, c.pos[1])
                elif c.dirny == 1 and c.pos[1] >= self.SETTINGS['r']-1:
                    c.pos = (c.pos[0], 0)
                elif c.dirny == -1 and c.pos[1] <= 0:
                    c.pos = (c.pos[0], self.SETTINGS['r']-1)
                else:
                    c.move(c.dirnx, c.dirny)

    def move_model(self, turn):
        if turn == 0 and self.head.dirnx != 1:
            self.dirnx = -1
            self.dirny = 0
        elif turn == 1 and self.head.dirnx != -1:
            self.dirnx = 1
            self.dirny = 0
        elif turn == 2 and self.head.dirny != 1:
            self.dirnx = 0
            self.dirny = -1
        elif turn == 3 and self.head.dirny != -1:
            self.dirnx = 0
            self.dirny = 1

        self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            pos = c.pos[:]

            if pos in self.turns:
                turn = self.turns[pos]
                c.move(turn[0], turn[1])

                if i == len(self.body) -1:
                    self.turns.pop(pos)
            else:
                if c.dirnx == -1 and c.pos[0] <= 0:
                    c.pos = (self.SETTINGS['r']-1, c.pos[1])
                elif c.dirnx == 1 and c.pos[0] >= self.SETTINGS['r']-1:
                    c.pos = (0, c.pos[1])
                elif c.dirny == 1 and c.pos[1] >= self.SETTINGS['r']-1:
                    c.pos = (c.pos[0], 0)
                elif c.dirny == -1 and c.pos[1] <= 0:
                    c.pos = (c.pos[0], self.SETTINGS['r']-1)
                else:
                    c.move(c.dirnx, c.dirny)

    def add_cube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0]-1, tail.pos[1]), self.SETTINGS))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0]+1, tail.pos[1]), self.SETTINGS))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1]-1), self.SETTINGS))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1]+1), self.SETTINGS))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def get_pos(self, start=0):
        pos = []
        for el in self.body[start:]:
            pos.append(el.pos)
        return pos

class Game():
    def __init__(self, width, height, rows, window, offx, offy, idx=""):
        self.SETTINGS = {}
        self.SETTINGS['w'] = width
        self.SETTINGS['h'] = height
        self.SETTINGS['r'] = rows
        self.SETTINGS['sB'] = width // rows
        self.SETTINGS['ox'] = offx * width
        self.SETTINGS['oy'] = offy * height
        self.idx = idx

        self.window = window

        self.snake = Snake((255, 0, 0), (self.SETTINGS['r']//2, self.SETTINGS['r']//2), self.SETTINGS)
        self.snack = Cube(self.randomSnack(), self.SETTINGS, color=(0, 255, 0))

        self.dist = self.get_snack_distance()

        self.walls = self.get_wall_pos()

        self.model = Model(len(self.get_observation()), 4)
        self.tgt = Model(len(self.get_observation()), 4)
        self.agent = DQNAgent(self.model, self.tgt)
        self.reward = 0.0
        self.setp_reward = 0.0
        self.rewards = []
        self.finished = False

        self.points = 0
        self.points_ls = []

    def get_wall_pos(self):
        pos = []

        for l in range(self.SETTINGS['r']):
            pos.append((l, 0))
            pos.append((l, self.SETTINGS['r']-1))
            pos.append((0, l))
            pos.append((self.SETTINGS['r']-1, l))
        
        return pos

    def get_snack_distance(self):
        x = self.snake.head.pos[0] - self.snack.pos[0]
        y = self.snake.head.pos[1] - self.snack.pos[1]
        dist = math.sqrt(x*x + y*y)
        return dist

    def draw_wall(self):
        for l in self.walls:
            offx = self.SETTINGS['ox']
            offy = self.SETTINGS['oy']
            dis = self.SETTINGS['sB']
            i = l[0]
            j = l[1]

            pygame.draw.rect(self.window,(0, 0, 255), (i * dis + 1 + offx, j * dis + 1 + offy, dis-2, dis-2))

    def draw_grid(self):
        x = 0
        y = 0

        for l in range(self.SETTINGS['r']):
            offx = self.SETTINGS['ox']
            offy = self.SETTINGS['oy']
            x += self.SETTINGS['sB']
            y += self.SETTINGS['sB']

            pygame.draw.line(self.window, (30, 30, 30), (x + offx, 0 + offy), (x + offx, self.SETTINGS['w'] + offy))
            pygame.draw.line(self.window, (30, 30, 30), (0 + offx, y + offy), (self.SETTINGS['h'] + offx, y + offy))

    def redrawWindow(self):
        self.draw_grid()
        self.draw_wall()

        self.snake.draw(self.window)
        self.snack.draw(self.window)

    def randomSnack(self):

        while True:
            x = random.randrange(self.SETTINGS['r'])
            y = random.randrange(self.SETTINGS['r'])

            new_pos = (x, y)

            if new_pos not in self.snake.get_pos():
                if new_pos not in self.get_wall_pos():
                    return new_pos

    def restart(self):
        self.snake = Snake((255, 0, 0), (self.SETTINGS['r']//2, self.SETTINGS['r']//2), self.SETTINGS)

        self.snack = Cube(self.randomSnack(), self.SETTINGS, color=(0, 255, 0))
        while self.snake.head.pos == self.snack.pos:
            self.snack = Cube(self.randomSnack(), self.SETTINGS, color=(0, 255, 0))

        self.rewards.append(self.reward)
        self.reward = 0.0
        self.points_ls.append(self.points)
        self.points = 0
        self.finished = False

    def get_average_reward(self):
        avg = sum(self.rewards) / len(self.rewards)
        self.rewards = []
        return avg

    
    def new_generation(self, model, tgt):
        self.agent.update_model(model)
        self.agent.update_tgt(tgt)
        self.restart()


    def game_loop(self, train = False, model = None):
        observation = self.get_observation()

        self.reward -= 0.003
        self.setp_reward = -0.003

        if train:
            action = self.agent.get_action(observation)

            self.snake.move_model(action)
        else:
            assert model != None, "Error, no model"

            action = model(torch.Tensor(observation)).max(-1)[-1].item()
            self.snake.move_model(action)

        if self.snake.body[0].pos == self.snack.pos:
            self.snake.add_cube()
            self.snack = Cube(self.randomSnack(), self.SETTINGS, color=(0, 255, 0))

            self.reward += 3.0
            self.setp_reward = 3.0
            self.points += 1

        for el in self.snake.body[1:]:
            if el.pos == self.snake.head.pos:
                self.snake = Snake((255, 0, 0), (self.SETTINGS['r']//2, self.SETTINGS['r']//2), self.SETTINGS)

                self.reward -= 1.0 
                self.setp_reward = -1.0
                self.finished = True
                break
        
        for el in self.walls:
            if el == self.snake.head.pos:
                self.snake = Snake((255, 0, 0), (self.SETTINGS['r']//2, self.SETTINGS['r']//2), self.SETTINGS)

                self.reward -= 1.0
                self.setp_reward = -1.0
                self.finished = True
                break

        if train:
            new_observation = self.get_observation()
            self.agent.step(self.setp_reward, new_observation, self.finished)

        if self.finished:
            self.restart()

        self.redrawWindow()

    def body_in_pos(self, x, y):
        for bd in self.snake.body[1:]:
            if bd.pos == [x, y]:
                return True

        return False



    def get_observation(self):

        obs = []
        # snake head, and tail pos
        sx = self.snake.head.pos[0]
        sy = self.snake.head.pos[1]
        obs.append(sx)
        obs.append(sy)
        obs.append(self.snake.body[-1].pos[0])
        obs.append(self.snake.body[-1].pos[1])

        # snack pos
        obs.append(self.snack.pos[0])
        obs.append(self.snack.pos[1])

        # snack dist from head
        x = self.snack.pos[0] - sx
        y = self.snack.pos[1] - sy
        dist = math.sqrt(x**2 + y**2)
        obs.append(dist)

        # head dist from closes obsticle (wall || body)
        # -x dir
        if self.body_in_pos(sx-1, sy) or sx-1 == 0:
            obs.append(0)
        else:
            obs.append(1)

        # +x dir
        if self.body_in_pos(sx+1, sy) or sx+2 == self.SETTINGS['r']:
            obs.append(0)
        else:
            obs.append(1)

        # -y dir
        if self.body_in_pos(sx, sy-1) or sy-1 == 0:
            obs.append(0)
        else:
            obs.append(1)

        # +y dir
        if self.body_in_pos(sx, sy+1) or sy+2 == self.SETTINGS['r']:
            obs.append(0)
        else:
            obs.append(1)

        return obs

    def start(self):
        clock = pygame.time.Clock()
        while True:
            pygame.time.delay(50)
            clock.tick(10)

            self.game_loop()

