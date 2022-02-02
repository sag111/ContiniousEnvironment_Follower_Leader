import pygame
from math import pi, degrees, radians, cos, sin
import numpy as np
from classes import AbstractRobot, GameObject
from scipy.spatial import distance

class Trajectory():
    pass
    #TODO: класс, который хранит траекторию, умеет её отрисовать и вообще молодец

class Game():
    def __init__(self, game_width=1024, 
                 game_height=768,
                 caption="Serious Robot Follower Simulation v.-1",
                 trajectory=None,
                 leader_pos_epsilon=5,
                 show_leader_path=True,
                 show_leader_trajectory=True):
        
        self.colours = {
                            'white':(255,255,255),
                            'black':(0,0,0),
                            'gray':(30,30,30),
                            'blue':(0,0,255),
                            'red':(255,0,0),
                            'green':(0,255,0)
                        }
        
        self.DISPLAY_WIDTH=game_width
        self.DISPLAY_HEIGHT=game_height
        
        self.trajectory = self.generate_trajectory()
        
        pygame.init()
        self.gameDisplay = pygame.display.set_mode((self.DISPLAY_WIDTH,self.DISPLAY_HEIGHT))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        
        if trajectory is None:
            trajectory=self.generate_trajectory()
        
        
        self.leader_pos_epsilon = leader_pos_epsilon
        
        self.show_leader_path = show_leader_path
        self.show_leader_trajectory = show_leader_trajectory
        
        
    
    def show_object(self,object_to_show):
        cur_image = object_to_show.image
        if hasattr(object_to_show, "direction"):
            cur_image = pygame.transform.rotate(cur_image, -object_to_show.direction)
            
        self.gameDisplay.blit(cur_image, object_to_show.position)
        
    
    def generate_trajectory(self, n=3, min_distance = 50, border = 20):
        """Генерирует точки на карте, по которым должен пройти ведущий"""
        
        trajectory = list()
        
        while len(trajectory) < n:
            new_point = np.array((np.random.randint(border,high=self.DISPLAY_WIDTH-border),
                                  np.random.randint(border,high=self.DISPLAY_HEIGHT-border)))
            
            if len(trajectory)==0:
                trajectory.append(new_point)
            else:
                to_add = True
                
                for prev_point in trajectory:
                    if distance.euclidean(prev_point,new_point) < min_distance:
                        to_add=False
                
                if to_add:
                    trajectory.append(new_point)            
            
        return trajectory
    
#     def generate_path(self, robot, trajectory):
#         """На основе списка точек генерирует путь"""
        
#         pass
    
    
    
    
    def main_loop(self):
        leader_img =  pygame.image.load("imgs/car_yellow.png")
        follower_img = pygame.image.load("imgs/car_poice.png")
        
        leader = AbstractRobot("leader",
                            image=leader_img,
                            height = 20,
                            width = 30,
                            start_position=(self.DISPLAY_WIDTH/2, self.DISPLAY_HEIGHT/2))

        follower = AbstractRobot("follower",
                              image=follower_img,
                              height = 20,
                              width = 30,
                              start_position=((self.DISPLAY_WIDTH/2)+50, (self.DISPLAY_HEIGHT/2)+50))


        
        self.trajectory.insert(0, ((self.DISPLAY_WIDTH/2), (self.DISPLAY_HEIGHT/2)))
        
        cur_target_id = 0
        
        done = False
        
        prev_leader_position = np.zeros(2)
#         self.gameDisplay.fill(colours["white"])

        self.game_object_list = list()
        self.game_object_list.append(leader)
        self.game_object_list.append(follower)
        
        self.leader_factual_trajectory = list()
        
        leader_finished = False
        
        cur_target_point = self.trajectory[1]
        
        while not done:
            self.gameDisplay.fill(self.colours["white"])
            # TODO: отдельный обработчик команд, чтобы это не было такой сосиской
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
        #                 follower.rotation_direction=-1
                        if follower.rotation_direction > 0:
                            follower.rotation_speed=0
                            follower.rotation_direction=0
                        else:
                            follower.rotation_direction=-1
                            follower.rotation_speed+=5
                        follower.command_turn(follower.rotation_speed,-1)
                        print("agent rotation speed and rotation direction", follower.rotation_speed, follower.rotation_direction)
                        print("current follower direction: ", follower.direction)
        

                    if (event.key == pygame.K_RIGHT):
        #                 follower.rotation_direction = 1
                        if follower.rotation_direction < 0:
                            follower.rotation_speed=0
                            follower.rotation_direction=0
                        else:
                            follower.rotation_direction=1
                            follower.rotation_speed+=5
                        follower.command_turn(follower.rotation_speed,1)
                        print("agent rotation speed and rotation direction", follower.rotation_speed, follower.rotation_direction)
                        print("current follower direction: ", follower.direction)


                    if event.key == pygame.K_UP:
                        follower.command_forward(follower.speed+1)
                        print("agent speed", follower.speed)

                    if event.key == pygame.K_DOWN:
                        follower.command_forward(follower.speed-1)
                        print("agent speed", follower.speed)
            
            
            
            prev_leader_position = leader.position.copy()
            
            if distance.euclidean(leader.position, cur_target_point) < self.leader_pos_epsilon:
                cur_target_id+=1
                if cur_target_id >= len(self.trajectory):
                    leader_finished = True
                else:
                    cur_target_point = self.trajectory[cur_target_id]
                
            if not leader_finished:
                leader.move_to_the_point(cur_target_point)
            else:
                leader.command_forward(0)
                leader.command_turn(0,0)
            follower.move()
                
#             if np.array_equal(prev_leader_position,leader.position):
            self.leader_factual_trajectory.append(leader.position.copy())
            
            if self.show_leader_path:
                pygame.draw.aalines(self.gameDisplay,self.colours["red"],False,self.trajectory)
            
            if self.show_leader_trajectory:
                for cur_point in self.leader_factual_trajectory[::10]:
                    pygame.draw.circle(self.gameDisplay, self.colours["black"], cur_point, 2)
            
            for cur_object in self.game_object_list:
                self.show_object(cur_object)

            pygame.display.update()
            self.clock.tick(60)
            pygame.time.wait(200)


        
if __name__=="__main__":

    game = Game()
    game.main_loop()
    pygame.quit()
    quit()
