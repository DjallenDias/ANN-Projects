class Car:
    def __init__(self, sprite, initial_xy: tuple[int, int], brain):
        self.sprite = sprite
        self.x, self.y = initial_xy
        self.angle = 0
        self.alive = True
        
        self.brain = brain
        self.fitness = 0
        
    def update_car(self, x, y, angle):
        self.x += x
        self.y += y
        self.angle += angle
        
        