import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]
        self.score = 0
        self.food = None
        self.portals = []
        self._place_food()
        self._place_portals()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake or any(x == self.food for x in self.portals):
            self._place_food()

    def _place_portals(self):
        self.portals = []
        for _ in range(2):  # Place 2 portals
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            portal = Point(x, y)
            while portal in self.snake or portal == self.food or portal in self.portals:
                x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                portal = Point(x, y)
            self.portals.append(portal)

    # def play_step(self, action):
    #     self.frame_iteration += 1
    #     # 1. collect user input
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             quit()

    #     # 2. move
    #     self._move(action)  # update the head
    #     self.snake.insert(0, self.head)

    #     # 3. check for portals
    #     self._check_portals()

    #     # 4. check if game over
    #     reward = 0
    #     game_over = False
    #     if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
    #         game_over = True
    #         reward = -10
    #         return reward, game_over, self.score

    #     # 5. place new food or just move
    #     if self.head == self.food:
    #         self.score += 1
    #         reward = 10
    #         self._place_food()
    #     else:
    #         self.snake.pop()

    #     # 6. update ui and clock
    #     self._update_ui()
    #     self.clock.tick(SPEED)
    #     # 7. return game over and score
    #     return reward, game_over, self.score
    def play_step(self, action):
        self.frame_iteration += 1

        # 1. Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. Check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Place new food or just move
        distance_to_food_before = abs(self.food.x - self.snake[1].x) + abs(self.food.y - self.snake[1].y)
        distance_to_food_after = abs(self.food.x - self.head.x) + abs(self.food.y - self.head.y)

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            if distance_to_food_after < distance_to_food_before:
                reward = 1  # Reward for moving closer to the food
            else:
                reward = -1  # Penalty for moving away from food

        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. Return game over and score
        return reward, game_over, self.score


    def _check_portals(self):
        for i in range(0, len(self.portals), 2):
            if self.head == self.portals[i]:
                # Teleport to the paired portal
                self.head = Point(self.portals[i + 1].x, self.portals[i + 1].y)
                break
            elif self.head == self.portals[i + 1]:
                self.head = Point(self.portals[i].x, self.portals[i].y)
                break

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # Draw the snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # Draw the food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw portals
        for i, portal in enumerate(self.portals):
            color = (0, 255, 0) if i % 2 == 0 else (255, 0, 255)
            pygame.draw.rect(self.display, color, pygame.Rect(portal.x, portal.y, BLOCK_SIZE, BLOCK_SIZE))

        # Display score
        score_text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(score_text, [0, 0])

        # Create a smaller font for descriptions
        small_font = pygame.font.Font('arial.ttf', 18)

        # Create description lines
        red_text = small_font.render("Red", True, RED)
        food_text = small_font.render("means food", True, WHITE)
        pink_text = small_font.render("Pink", True, (255, 105, 180))  # Pink color
        and_text = small_font.render("and", True, WHITE)
        green_text = small_font.render("Green", True, (0, 255, 0))  # Green color
        portals_text = small_font.render("represent portals", True, WHITE)

        # Get window dimensions
        window_width, window_height = self.display.get_size()

        # Calculate positions
        right_margin = 10  # Space from the right edge
        line_spacing = 5  # Spacing between lines

        # Line 1: "Red means food"
        red_text_pos = (window_width - red_text.get_width() - food_text.get_width() - right_margin, window_height - 50)
        food_text_pos = (red_text_pos[0] + red_text.get_width() + 5, window_height - 50)

        # Line 2: "Pink and Green represent portals"
        pink_text_pos = (window_width - pink_text.get_width() - and_text.get_width() - green_text.get_width() - portals_text.get_width() - right_margin, window_height - 25)
        and_text_pos = (pink_text_pos[0] + pink_text.get_width() + 5, window_height - 25)
        green_text_pos = (and_text_pos[0] + and_text.get_width() + 5, window_height - 25)
        portals_text_pos = (green_text_pos[0] + green_text.get_width() + 5, window_height - 25)

        # Draw text for line 1
        self.display.blit(red_text, red_text_pos)
        self.display.blit(food_text, food_text_pos)

        # Draw text for line 2
        self.display.blit(pink_text, pink_text_pos)
        self.display.blit(and_text, and_text_pos)
        self.display.blit(green_text, green_text_pos)
        self.display.blit(portals_text, portals_text_pos)

        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)