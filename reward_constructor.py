import json
from dataclasses import dataclass

@dataclass
class Reward:
    name: str = "base_reward"
    reward_in_box: float = 1. # награда за то, что агент на маршруте и на нужной дистанции
    reward_on_track: float = 0.1  # награда за то, что агент на маршруте (но не в коробке)
    reward_in_dev: float = 0.5 # награда за то, что агент в пределах погрешности от маршрута
    leader_movement_reward: float = 1. # награда за то, что лидер движется
    
    crash_penalty: float = -10. # штраф за то, что агент врезался в стену/лидера
    not_on_track_penalty: float = -1. # штраф за нахождение вне "коробки" и не на маршруте
    too_close_penalty: float = -5. # штраф за то, что агент слишком близок к лидеру
    leader_stop_penalty: float = -1. # штраф за простой лидера в результате команды агента "остановись"
        
    def to_json(self, filepath):
        with open(filepath, "w") as json_out_file:
            json.dump(self.__dict__, json_out_file)
        
        
    def from_json(self, filepath):
        with open(filepath, "r") as json_inp_file:
            inp_dict = json.load(json_inp_file)
            
        self.__init__(**inp_dict)
        
    def from_dict(self, inp_dict):
        self.__init__(**inp_dict)