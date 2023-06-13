# Модель арктического робота под управлением gym

Требования: Ubuntu 20 LTS, ROS Noetic

Инструкция по установке ROS Noetic: http://wiki.ros.org/noetic/Installation/Ubuntu

## TODO
- [x] Интегрировать пакет (https://github.com/aaselivanov/continuous-grid-arctic)
- [x] Перевод кватерниона в угол рыскания
- [x] Применение сохраненных координат ведущего к радару
- [x] Расчет наград за действия (dataclass Reward)
- [x] Завершение симуляции метка done:
   - максимальное число временных шагов
   - ведомый врезался в ведущего
   - ведомый отстал от ведущего
   - ведомый набрал -200 награды
- [ ] Завершение симуляции, когда ведомый не может продолжать движение (перевернулся, застрял и т.д.)
- [ ] Применить RL по библиотеки Rllib, используя External Env

## Установка

1. Последняя версия арктического робота на момент 22.03.2022. \
   Распаковать архив **separated_arctic_install_2022_03_22.tar.gz** (https://disk.yandex.ru/d/noeV8srqj3lpsA) в папку src. 
  

2. Выполнить установку дополнительных зависимостей

        cd ~/arctic_ws;
        rosdep install --from-paths src --ignore-src -r -y

3. Устанвить дополнительные библиотеки и пакеты

        sudo apt-get install ros-noetic-velodyne* ros-noetic-map-server ros-noetic-move-base ros-noetic-velocity-controllers;
        sudo pip install -U rdflib ply lxml requests pycollada shapely descartes scipy networkx matplotlib

4. Проект с arctic_gym в располагается в отдельной папке, для него требуется установленная anaconda и окружение rl-ros

## Запуск

1. В первом терминале запуск без окружения Anaconda:

        roslaunch arctic_model_gazebo arctic_semiotic_soar.launch rl_follow:=True

2. Во втором терминале из директории ~/robots_HRI:

        sh server.sh

3. В третьем терминале запуск из директории ~/robots_HRI:

        sh inference.sh
