## Врапперы 
Классы обёртки вокруг класса среда для пред/постобработки наблюдений и действий при взаимодействии среды и алгоритма управления. Кастомные врапперы реализованы в модуле [wrappers](../continuous_grid_arctic/utils/wrappers.py). Важно, при добавлении новых сенсоров, надо добавить в используемый враппер их обработку.
- [ContinuousObserveModifier_v0](https://github.com/sag111/continuous-grid-arctic/blob/slava_3/continuous_grid_arctic/utils/wrappers.py#L70)
- [ContinuousObserveModifierPrev](https://github.com/sag111/continuous-grid-arctic/blob/slava_3/continuous_grid_arctic/utils/wrappers.py#L252) - Враппер для накопления предыдущих значений двух модернизированных сенсоров (1) Лучевой сенсор с 12 лучами на коридор
    и препятствия; 2) Лучевой сенсор на препятствия с 30 (вариативно) лучами
- [ContinuousObserveModifier_lidarMap2d](https://github.com/sag111/continuous-grid-arctic/blob/slava_3/continuous_grid_arctic/utils/wrappers.py#L336) - Враппер, который выходы лидара преобразует в 2д картинку, на которой отображаются: препятствия, положение лидера, сейф зона на маршруте.
- [ContinuousObserveModifier_lidarMap2d_v2](https://github.com/sag111/continuous-grid-arctic/blob/slava_3/continuous_grid_arctic/utils/wrappers.py#L500) - тоже, что и предыдущий, но сейф зона отрисовывается по другому

## Прочие
- gym.wrappers.Monitor - враппер для записи mp4 рендеров симуляций.

## Устаревшие
- [MyFrameStack](https://github.com/sag111/continuous-grid-arctic/blob/slava_3/continuous_grid_arctic/utils/wrappers.py#L14) - класс для накапливания наблюдений и использования информации не только с текущего шага, но и с предыдущих. Давно не обновлялся, вместо него сейчас используется [ContinuousObserveModifierPrev](https://github.com/sag111/continuous-grid-arctic/blob/slava_3/continuous_grid_arctic/utils/wrappers.py#L252)
- [LeaderTrajectory_v0](https://github.com/sag111/continuous-grid-arctic/blob/slava_3/continuous_grid_arctic/utils/wrappers.py#L342) - нужен только для проверки обратной совместимости с экспериемнтами, запущенными на коммите 86211bf4a3b0406e23bc561c00e1ea975c20f90b
