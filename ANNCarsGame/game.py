from random import choice
import pygame
from pygame.constants import HWSURFACE, DOUBLEBUF
from pathlib import Path

from car import Car
from ann import ArtificialNeuralNetwork as ANN

INPUT_NEURONS = 2
HIDDEN_LAYERS = 2
HIDDEN_NEURONS_PER_LAYER = 2
OUTPUT_NEURONS = 2

ANN_SIZES = [INPUT_NEURONS, HIDDEN_LAYERS, HIDDEN_NEURONS_PER_LAYER, OUTPUT_NEURONS]

class GameState:
    def __init__(self, qtyCars, sprites):
        self.cars: list[Car] = []
        for i in range(qtyCars):
            self.cars.append(Car(choice(sprites), (0,0), ANN(*ANN_SIZES)))

class Ui:
    def __init__(self):
        pygame.init()
        
        self.cars_dict = {}
        for car_img_file in Path("assets/sprites/cars").glob("*_car.png"):
            self.cars_dict[car_img_file.name] = pygame.image.load(car_img_file)
        
        self.gamestate = GameState(1, list(self.cars_dict.keys()))
        self.window = pygame.display.set_mode((1600, 870), HWSURFACE | DOUBLEBUF)
        
        self.info_area = self.window.subsurface(0, 0, 1600, 280)     
        self.graph_area = self.info_area.subsurface(0, 0, 800, 280)
        self.graph_area.fill((255,255,0))
        
        self.ann_demo_area = self.info_area.subsurface(800, 0, 800, 280)
        self.ann_demo_area.fill((0,255,255))
        
        self.game_area = self.window.subsurface(0, 280, 1600, 870-280)
        self.game_area.fill((0,255,0))
        
        self.clock = pygame.time.Clock()
        self.running = True
        
    def process_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                break
                
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_w]:
            self.gamestate.cars[0].x += 1
        
        if keys[pygame.K_s]:
            self.gamestate.cars[0].x -= 1
    
    def renderWorld(self, surface: pygame.Surface):
        tileRect = pygame.Rect(0, 0, 64, 64)
        for car in self.gamestate.cars:
            surface.blit(self.cars_dict[car.sprite], (car.x, car.y), tileRect)
    
    def render(self):
        self.game_area.fill((0,255,0))
        self.renderWorld(self.game_area)
        pygame.display.update()
        
    def run(self):
        while self.running:
            self.clock.tick(60)
            self.process_input()
            self.render()
                    
        pygame.quit()
    
def main():
    gameui = Ui()
    gameui.run()

if __name__ == "__main__":
    main()