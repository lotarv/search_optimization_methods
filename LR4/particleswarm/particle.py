import numpy.random


class Particle(object):
    """
    Класс, описывающий одну частицу
    """
    def __init__(self, swarm):
        """
        swarm - экземпляр класса Swarm, хранящий параметры алгоритма, список частиц и лучшее значение роя в целом
        """
        # Текущее положение частицы
        self.__currentPosition = self.__getInitPosition(swarm)

        # Лучшее положение частицы
        self.__localBestPosition = self.__currentPosition[:]

        # Лучшее значение целевой функции
        self.__localBestFinalFunc = swarm.getFinalFunc(self.__currentPosition)

        self.__velocity = self.__getInitVelocity(swarm)

    @property
    def position(self):
        return self.__currentPosition

    @property
    def velocity(self):
        return self.__velocity

    def __getInitPosition(self, swarm):
        """
        Возвращает список со случайными координатами для заданного интервала изменений
        """
        return numpy.random.rand(swarm.dimension) * (swarm.maxvalues - swarm.minvalues) + swarm.minvalues

    def __getInitVelocity(self, swarm):
        """
        Сгенерировать начальную случайную скорость
        """
        assert len(swarm.minvalues) == len(self.__currentPosition)
        assert len(swarm.maxvalues) == len(self.__currentPosition)

        minval = -(swarm.maxvalues - swarm.minvalues)
        maxval = (swarm.maxvalues - swarm.minvalues)

        return numpy.random.rand(swarm.dimension) * (maxval - minval) + minval

    def nextIteration(self, swarm):
        """
        Выполнить следующую итерацию для частицы: обновить скорость, позицию и личное лучшее
        """
        # Случайные векторы для коррекции скорости
        rnd_currentBestPosition = numpy.random.rand(swarm.dimension)
        rnd_globalBestPosition = numpy.random.rand(swarm.dimension)

        # Коэффициент сжатия ( Clerc-Kennedy )
        veloRatio = swarm.localVelocityRatio + swarm.globalVelocityRatio
        commonRatio = (2.0 * swarm.currentVelocityRatio /
                       (numpy.abs(2.0 - veloRatio - numpy.sqrt(veloRatio ** 2 - 4.0 * veloRatio))))

        # Новая скорость
        newVelocity_part1 = commonRatio * self.__velocity
        newVelocity_part2 = (commonRatio * swarm.localVelocityRatio * rnd_currentBestPosition *
                             (self.__localBestPosition - self.__currentPosition))
        newVelocity_part3 = (commonRatio * swarm.globalVelocityRatio * rnd_globalBestPosition *
                             (swarm.globalBestPosition - self.__currentPosition))
        self.__velocity = newVelocity_part1 + newVelocity_part2 + newVelocity_part3

        # Обновить позицию
        self.__currentPosition += self.__velocity

        # Оценить целевую функцию и обновить личное лучшее
        finalFunc = swarm.getFinalFunc(self.__currentPosition)
        if finalFunc < self.__localBestFinalFunc:
            self.__localBestPosition = self.__currentPosition[:]
            self.__localBestFinalFunc = finalFunc