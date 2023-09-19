# continuous-grid-arctic

## 2D среда continuous-grid-arctic для обучения 
В репозитории в [папке src/continuous_grid_arctic](src/continuous_grid_arctic) представлена реализация gym-среды решения задачи следования за лидером. В среде реализованы два агента: ведущий и ведомый, реализованы статические и динамические препятствия. 

- [Описание основных классов среды](docs/README.md)
- [Описание сенсоров](docs/Sensors.md)
- [Описание врапперов](docs/Wrappers.md)

## 3D среда arctic_gym для апробации
В репозитории в [папке src/arctic_gym](src/arctic_gym) представлена реализация решения задачи следования за лидером 
в 3D среде Gazebo. В папке представлен код, который можно подключить к собственной среде, реализованной в ROS Gazebo.

- [Описание использования системы управления роботом](docs/Arctic.md)

## Установка
```
git clone https://github.com/sag111/continuous-grid-arctic
cd continuous-grid-arctic
```
Если нет необходимости менять код среды, можно установить через файл setup.py.
```
pip install git+https://github.com/sag111/continuous-grid-arctic.git
```

### Установка окружения
1. Установка с помощью файла conda.yml 
``` 
conda env create -f conda.yml 
```

2. Установка в случае возникновения ошибок:
``` 
conda create -n rl -c conda-forge python-pdal=3.1.2 python=3.7 

pip install requirements.txt
```

3. Для 2д среды достаточно установить pygame~=2.1.2, pandas и numpy 


### Требования:
- Python 3.6+
- gym~=0.21.0
- numpy~=1.19.5
- scipy~=1.7.3
- setuptools~=58.0.4
- matplotlib~=3.5.1
- pygame~=2.1.2
- pandas~=1.3.5


Пожалуйста, используйте этот bibtex, если вы хотите цитировать этот репозиторий в своих публикациях:
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


## Примеры использования [2D среды](src/continuous_grid_arctic):
Для демонстрации работы среды в ручном режиме работы необходимо запускать файл main.py:
```
python run_2d.py --mode manual
или
python run_2d.py --mode manual --seed 0 --hardcore --manual_input gamepad --log_results
```
Возможные аргументы командной строки описаны в самом скрипте run_2d.py. Кроме этого настроить среду можно в скрипте 
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



