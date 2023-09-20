import json
from dataclasses import dataclass

@dataclass
class Reward:
    name: str = "base_reward"
    reward_in_box: float = 1. # reward for the fact that the agent is on the route and at the required distance
    reward_on_track: float = 0.1  # reward for having the agent on the route (but not in the box)
    reward_in_dev: float = 0.5 # reward for the fact that the agent is within error of the route
    leader_movement_reward: float = 1. # reward for the leader moving
    
    crash_penalty: float = -10. # penalty for hitting the agent into a wall/leader
    not_on_track_penalty: float = -1. # penalty for being outside the “box” and not on the route
    too_close_penalty: float = -5. # penalty for being too close to the leader
    leader_stop_penalty: float = -1. # penalty for leader downtime as a result of the agent's "stop" command
        
    def to_json(self, filepath):
        with open(filepath, "w") as json_out_file:
            json.dump(self.__dict__, json_out_file)
        
    @classmethod
    def from_json(cls, filepath):
        with open(filepath, "r") as json_inp_file:
            inp_dict = json.load(json_inp_file)
            
        return Reward(**inp_dict)
        
    def from_dict(self, inp_dict):
        self.__init__(**inp_dict)