# Import the pygame module
import pygame
import tron
import math

# Import pygame.locals for easier access to key coordinates
# Updated to conform to flake8 and black standards
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

g = tron.Game()

# Define constants for the screen width and height
SCREEN_WIDTH = g.width * 25
SCREEN_HEIGHT = g.height * 25

class Bike:
    def __init__(self, screen, x, y, color=(255, 0, 255)):
        self.screen = screen
        self.bike_trail = []
        self.head = (x, y)
        self.make_move(x, y)
        self.color = color

    def draw_bike(self):
        grid_width = math.floor(SCREEN_WIDTH / g.width)
        grid_height = math.floor(SCREEN_HEIGHT / g.height)

        for i in range(1, len(self.bike_trail)):
            start = (self.bike_trail[i - 1][0] * grid_width + grid_width / 2, self.bike_trail[i - 1][1] * grid_height + grid_height / 2)
            end = (self.bike_trail[i][0] * grid_width + grid_width / 2, self.bike_trail[i][1] * grid_height + grid_height / 2)

            pygame.draw.line(self.screen, self.color, start_pos=start, end_pos=end, width=5)

            if i + 1 == len(self.bike_trail):
                pygame.draw.circle(self.screen, self.color, center=end, radius=grid_width / 3, width=0)

    def make_move(self, px, py):
        g.remove_from_neighbors(px, py)
        g.current_state[px, py] = tron.Game.get_char(0)
        g.player_turn = tron.Game.get_char(1)
        self.bike_trail.append((px, py))
        self.head = (px, py)

    def left(self):
        ox, oy = self.head
        px, py = ox - 1, oy
        self.make_move(px, py)

    def right(self):
        ox, oy = self.head
        px, py = ox + 1, oy
        self.make_move(px, py)

    def up(self):
        ox, oy = self.head
        px, py = ox, oy - 1
        self.make_move(px, py)

    def down(self):
        ox, oy = self.head
        px, py = ox, oy + 1
        self.make_move(px, py)

    def make_ai_move(self, heads, depth=3):
        m, (qx, qy) = g.get_ai_move(heads, 1, depth=depth)
        self.make_move(qx, qy)
        g.player_turn = tron.Game.get_char(0)

    def draw_board_visual(self):
        """
        Draw the game board

        :return:
        """
        grid_width = math.floor(SCREEN_WIDTH / g.width)
        grid_height = math.floor(SCREEN_HEIGHT / g.height)

        for x in range(0, g.width):
            for y in range(0, g.height):
                char = g.current_state[x, y]

                # Draw the player on the screen
                pygame.draw.rect(self.screen, tron.Game.get_char_color(char), pygame.Rect(grid_width*x, grid_height*y, grid_width - 1, grid_height - 1))

    def draw_rect(self, x, y, color):
        grid_width = math.floor(SCREEN_WIDTH / g.width)
        grid_height = math.floor(SCREEN_HEIGHT / g.height)
        rect = pygame.Rect(grid_width * x, grid_height * y, grid_width - 1, grid_height - 1)

        shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
        self.screen.blit(shape_surf, rect)

    def draw_voronoi_region(self, heads):
        def color_drawer(x, y, char):
            alpha = 80

            # Articulation points
            if char == "A":
                self.draw_rect(x, y, (0, 0, 255, alpha))

            # Battleground points
            elif char == "B":
                self.draw_rect(x, y, (255, 255, 0, alpha))

            # Player voronoi region
            else:
                self.draw_rect(x, y, tron.Game.get_char_color(char, alpha))

        g.voronoi_heuristic_hypermax(heads, color_drawer)

# Initialize pygame
pygame.init()

# Setup the clock for a decent framerate
clock = pygame.time.Clock()

# Create the screen object
# The size is determined by the constant SCREEN_WIDTH and SCREEN_HEIGHT
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

bikes = [Bike(screen, 0, 0, (255, 255, 0)), Bike(screen, g.width-1, g.height-1, (255, 0, 0))]

# Variable to keep the main loop running
running = True

# Current direction
direction = K_RIGHT

# Main loop
while running:
    # Process one time per second
    clock.tick(1)

    # Look at every event in the queue
    for event in pygame.event.get():

        # Did the user hit a key?
        if event.type == KEYDOWN:

            # Was it the Escape key? If so, stop the loop.
            if event.key == K_ESCAPE:
                running = False

            # The user has changed direction
            elif event.key in [K_UP, K_DOWN, K_LEFT, K_RIGHT]:
                direction = event.key

        # Did the user click the window close button? If so, stop the loop.
        elif event.type == QUIT:
            running = False

    # Fill the screen with white
    screen.fill((0, 0, 0))

    # Convert keypresses into direction
    try:
        if direction == K_UP:
            bikes[0].up()

        elif direction == K_DOWN:
            bikes[0].down()

        elif direction == K_LEFT:
            bikes[0].left()

        elif direction == K_RIGHT:
            bikes[0].right()

    # The player tried making an invalid move
    except (ValueError, KeyError):
        print("Player lost")
        running = False

    try:
        # Let the ai make a move
        bikes[1].make_ai_move([bike.head for bike in bikes], depth=2)

    # The ai tried to make a valid move
    except ValueError:
        print("Ai lost")
        running = False

    # Draw the player on the screen
    # draw_board_visual(screen=screen)
    for bike in bikes:
        bike.draw_bike()
    bikes[0].draw_voronoi_region([bike.head for bike in bikes])

    # Update the display
    pygame.display.flip()
