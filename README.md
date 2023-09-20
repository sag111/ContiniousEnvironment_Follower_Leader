# continuous-grid-arctic

## 2D environment
[src/continuous_grid_arctic](src/continuous_grid_arctic) contains gym environment for "following the leader" task. 

- [Environment classes description](docs/README.md)
- [Sensors description](docs/Sensors.md)
- [Wrappers description](docs/Wrappers.md)

## 3D environment
[src/arctic_gym](src/arctic_gym) contains gym-gazebo environment "following the leader" task.

- [Gym-gazebo description](docs/Arctic.md)

## Setup
```
git clone https://github.com/sag111/continuous-grid-arctic
```
Use as package
```
pip install git+https://github.com/sag111/continuous-grid-arctic.git
```

### Anaconda env install
1. Installation using conda.yml 
``` 
conda env create -f conda.yml 
```

2. If errors occurred:
``` 
conda create -n rl -c conda-forge python-pdal=3.1.2 python=3.7; 
conda activate rl;
pip install -r requirements.txt
```

3. For 2D environment 
```
pip install pygame==2.1.2 pandas numpy
``` 

### Requirements:
- Python 3.7.12
- setuptools==66.0.0
- wheel==0.38.4
- opencv-python==4.5.4.60
- ray\[rllib\]==1.9.5
- pygame==2.1.2
- pyhocon==0.3.60
- rospkg==1.4.0
- importlib-metadata==4.13.0
- open3d==0.17.0
- torch==1.13.1
- protobuf==3.20

## 2D environment usage:
To demonstrate how the environment works in manual mode run the file main.py:
```
python run_2d.py --mode manual --seed 0 --hardcore --manual_input gamepad --log_results
```
Possible command line arguments are described in the script [run_2d.py](src/run_2d.py). Кроме этого настроить среду 
можно в скрипте 
follow_the_leader_continus_env.py. Для этого надо посмотреть в run_2d.py, какая именно среда запускается и в
follower_the_leader_continuous_env изменить параметры. Например можно ускорить симуляцию увеличив framerate или 
frames_per_step. 

По умолчанию одна симуляция длится не более 5000 шагов (задаётся при создании конкретной среды параметром max_steps) 
или до тех пор, пока агент не попадёт в аварию.

В папке notebooks располагается два демонстрационных Jupyter блокнота. 
1. [Env_demo](src/continuous_grid_arctic/notebooks/Env_demo.ipynb) содержит демонстрационную программу для взаимодействия со средой
2. [Ray_train_demo](src/continuous_grid_arctic/notebooks/Ray_train_demo.ipynb) содержит демонстрационную программу для обучения агента и тестирования полученной модели 
с использованием библиотеки ray[rllib]

Ниже представлено тестовое прохождение одного маршрута с использованием **LeaderPositionsTracker_v2**, 
**LeaderCorridor_Prev_lasers_v2**, **LaserPrevSensor**. Модель обучена в конфигурации среды с 35 статическими 
препятствиями и 1 динамическим. 

<p align="center">
<img src="src/continuous_grid_arctic/figures/demo_video.gif" width="500">
</p>

## Примеры использования [3D среды](src/arctic_gym):
Для демонстрации работы модели в 3D-среде необходимо ознакомиться с инструкцией [Arctic.md](docs%2FArctic.md)

<p align="center">
<img src="src/arctic_gym/figures/demo_gazebo.gif" width="500">
</p>

## Конфигурация собственной среды
Чтобы создать собственную конфигурацию среды, необходимо выполнить следующие шаги: 
1. В файле follow_the_leader_continuous_env.py создать наследующий основную среду класс (как, например, Test-Cont-Env-Auto-v0);
2. В методе init созданного класса задать нужные параметры при инициализации родительского класса 
(полный список параметров смотреть в методе init класса Game);
3. Далее, "зарегистрировать" среду как среду gym с помощью gym_register, по следующему шаблону:

    3.1. id=Test-Cont-Env-<собственное_название>-v0;
    
    3.2. follow_the_leader_continuous_env:<название класса среды, который создан в п.1>;
    
    3.3. reward_threshold по своему желанию.


## Citation

Please use this bibtex if you want to cite this repository in your publications:
```
@article{selivanov2022environment,
  title={An environment emulator for training a neural network model to solve the “Following the leader” task},
  author={Selivanov, Anton and Rybka, Roman and Gryaznov, Artem and Shein, Vyacheslav and Sboev, Alexander},
  journal={Procedia Computer Science},
  volume={213},
  pages={209--216},
  year={2022},
  publisher={Elsevier}
}
```
