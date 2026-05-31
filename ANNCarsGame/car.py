from pygame.math import Vector2
from math import radians, cos, sin


class Car:
    def __init__(self, sprite, initial_xy: tuple[int, int], brain):
        self.sprite = sprite
        self.x, self.y = initial_xy
        
        self.angle = 0
        self.velocity = 0
        
        self.max_vel = 8
        self.acceleration = 0.2
        self.deceleration = 0.1
        self.base_turn_speed = 5
        
        self.brain = brain
        self.fitness = 0
        
        self.alive = True
        
    def update_car(self, accel: bool, brake: bool, turn_left: bool, turn_right: bool):
        if not self.alive:
            return
        
        if accel:
            self.velocity = min(self.velocity + self.acceleration, self.max_vel)
            
        elif brake:
            self.velocity = max(self.velocity - self.acceleration, -2)
            
        else:
            if self.velocity > 0:
                self.velocity = max(self.velocity - self.deceleration, 0)
            
            elif self.velocity < 0:
                self.velocity = min(self.velocity + self.deceleration, 0)
                
        if self.velocity != 0:
            vel_factor = 1.0 - (abs(self.velocity) / (self.max_vel * 1.5))
            vel_factor = max(0.2, vel_factor)
            
            turn_rate = self.base_turn_speed * vel_factor
            
            if turn_left:
                self.angle += turn_rate
            
            if turn_right:
                self.angle -= turn_rate
                
        move_fac = Vector2(0, -self.velocity)
        move_fac.rotate_ip(-self.angle)
        
        self.x += move_fac.x
        self.y += move_fac.y
        