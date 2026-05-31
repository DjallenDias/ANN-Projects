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

class SpritesManager:
    def __init__(self):
        
        self.cars_dict = {}
        for car_img_file in Path("assets/sprites/cars").glob("car*small*.png"):
            self.cars_dict[car_img_file.name] = pygame.transform.smoothscale(pygame.image.load(car_img_file), (20, 35))
            
        
        self.start = [pygame.image.load("assets/sprites/asphalt_road/road_asphalt42.png"),
                      pygame.image.load("assets/sprites/asphalt_road/road_asphalt43.png")]
        
        self.straight = [pygame.image.load("assets/sprites/asphalt_road/road_asphalt01.png"),
                         pygame.image.load("assets/sprites/asphalt_road/road_asphalt02.png")]
        
        self.curve = [pygame.image.load("assets/sprites/asphalt_road/road_asphalt05.png"),
                      pygame.image.load("assets/sprites/asphalt_road/road_asphalt03.png"),
                      pygame.image.load("assets/sprites/asphalt_road/road_asphalt39.png"),
                      pygame.image.load("assets/sprites/asphalt_road/road_asphalt41.png"),]                
        

class GameState:
    def __init__(self, qtyCars, sprites):
        self.cars: list[Car] = []
        for i in range(qtyCars):
            self.cars.append(Car(choice(sprites), (0,0), ANN(*ANN_SIZES)))

class Ui:
    def __init__(self):
        pygame.init()
        
        self.spritesManager = SpritesManager()
        self.gamestate = GameState(1, list(self.spritesManager.cars_dict.keys()))
        
        self.window = pygame.display.set_mode((1600, 870))
        
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
        
        accel = False
        brake = False
        turn_left = False
        turn_right = False
        
        if keys[pygame.K_w]: accel = True
        if keys[pygame.K_s]: brake = True
        if keys[pygame.K_a]: turn_left = True
        if keys[pygame.K_d]: turn_right = True
    
        self.gamestate.cars[0].update_car(accel, brake, turn_left, turn_right)
    
    def renderWorld(self, surface: pygame.Surface):
        tileRect = pygame.Rect(0, 0, 64, 64)
        for car in self.gamestate.cars:
            orig_sprite = self.spritesManager.cars_dict[car.sprite]
            rotated_sprite = pygame.transform.rotate(orig_sprite, car.angle)
            sprite_rect = rotated_sprite.get_rect(center=(car.x, car.y))
            surface.blit(rotated_sprite, sprite_rect.topleft)
    
    def render(self):
        self.game_area.fill((0,255,0))
        self.game_area.blit(self.spritesManager.curve[1], (0,0,0,0))
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