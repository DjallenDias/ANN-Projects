import pygame
from pygame.constants import HWSURFACE, DOUBLEBUF

class Ui:
    def __init__(self):
        pygame.init()
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
            
    def render(self):
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