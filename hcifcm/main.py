import sys
from collections import defaultdict

from hcifcm.fuzzy import *
from hcifcm.fcm import FCM, sigmoid
from hcifcm.opt import hebbian_learning, pso, FcmPso


def create_mfs(names, min_x, max_x, std: float=1):
    n = len(names)
    for i, name in enumerate(names):
        x = i / (n - 1.0)
        center = min_x + x * (max_x - min_x)
        yield GaussianMf(name, center, std)


def plot(mfs, x_from: float=0, x_to: float=1):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(0, 1.2)
    for mf in mfs:
        centroid = mf.center()
        ax.text(centroid, 1.05, mf.name, horizontalalignment='center')
        x = np.linspace(x_from, x_to, 100)
        y = np.zeros(100)
        for i, xx in enumerate(x):
            y[i] = mf.sample(xx)
        plt.plot(x, y, label=mf.name, c='black')
    plt.title('Функция принадлежности')
    plt.ylabel('Степень принадлежности')
    #plt.legend()
    plt.show()


EXTREMELY_LOW = 'Чрезвычайно низкое'
VERY_LOW = 'Очень низкое'
LOW = 'Низкое'
MEDIUM = 'Среднее'
HIGH = 'Высокое'
VERY_HIGH = 'Очень высокое'
EXTREMELY_HIGH = 'Чрезвычайно высокое'

STRONG_DECREASE = 'Сильное снижает'
DECREASE = 'Понижает'
SLIGHT_DECREASE = 'Немного снижает'
NO_CHANGE = 'Не влияет'
SLIGHT_INCREASE = 'Немного повышает'
INCREASE = 'Повышает'
STRONG_INCREASE = 'Сильно повышает'


def main(*args):
    mfs_concept = list(create_mfs([VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH], 0, 1, 0.1))
    mfs_edge = list(create_mfs([STRONG_DECREASE, DECREASE, SLIGHT_DECREASE, NO_CHANGE, SLIGHT_INCREASE, INCREASE, STRONG_INCREASE], -1, 1, 0.2))
    #plot(mfs_concept, 0, 1)
    #plot(mfs_edge, -1, 1)
    #return

    fcm = FCM(mfs_concept, mfs_edge)
    c1 = fcm.add_concept('Нормативно-правовая база', LOW)
    c2 = fcm.add_concept('Теория построения АСДУ', LOW)
    c3 = fcm.add_concept('Качество методологии построения АСДУ', LOW)
    fcm.add_edge(c1, c3, INCREASE)
    fcm.add_edge(c2, c3, INCREASE)

    c4 = fcm.add_concept('Степень вертикальной интеграции', LOW)
    fcm.add_edge(c3, c4, INCREASE)

    c5 = fcm.add_concept('Ориентированность на автоматизацию управления бизнес-процессом', LOW)
    fcm.add_edge(c4, c5, INCREASE)

    c6 = fcm.add_concept('Использования инновационных технологий ТП', LOW)
    fcm.add_edge(c3, c6, INCREASE)

    c7 = fcm.add_concept('Качество подготовки персонала', MEDIUM)
    fcm.add_edge(c3, c7, INCREASE)

    c7 = fcm.add_concept('Использование зарубежного ПО', HIGH)
    fcm.add_edge(c3, c7, DECREASE)

    c8 = fcm.add_concept('Стандартизация и унификация систем', LOW)
    fcm.add_edge(c3, c8, INCREASE)

    c9 = fcm.add_concept('Риск вредоносного вмешательства', MEDIUM)
    fcm.add_edge(c7, c9, INCREASE)

    c10 = fcm.add_concept('Разнородность систем', MEDIUM)
    fcm.add_edge(c7, c10, SLIGHT_INCREASE)
    fcm.add_edge(c8, c10, STRONG_DECREASE)

    c11 = fcm.add_concept('Качество и оснащение систем поддержки принятия решений', LOW)
    fcm.add_edge(c6, c11, INCREASE)

    c12 = fcm.add_concept('Степень автоматизации бизнес процессов', LOW)
    fcm.add_edge(c4, c12, INCREASE)
    fcm.add_edge(c5, c12, INCREASE)
    fcm.add_edge(c6, c12, INCREASE)

    c13 = fcm.add_concept('Квалификация диспетчерского персонала', MEDIUM)
    fcm.add_edge(c7, c13, INCREASE)

    c14 = fcm.add_concept('Квалификация дежурного персонала', MEDIUM)
    fcm.add_edge(c7, c14, INCREASE)

    c15 = fcm.add_concept('Качество и оснащение систем поддержки принятия управленческих решений', MEDIUM)
    fcm.add_edge(c11, c15, INCREASE)

    c16 = fcm.add_concept('Ценность представляемой информации для принятия управленческих решений', MEDIUM)
    fcm.add_edge(c12, c16, INCREASE)

    c17 = fcm.add_concept('Качество принимаемых управленческих решений', MEDIUM)
    fcm.add_edge(c16, c17, INCREASE)
    fcm.add_edge(c15, c17, INCREASE)

    c18 = fcm.add_concept('Количество устаревших систем', HIGH)
    fcm.add_edge(c6, c18, DECREASE)

    c19 = fcm.add_concept('Затраты на поддержание АСУ ТП', MEDIUM)
    fcm.add_edge(c18, c19, INCREASE)

    c20 = fcm.add_concept('Производственные затраты', MEDIUM)
    fcm.add_edge(c19, c20, INCREASE)

    c21 = fcm.add_concept('Эффективность поддержки деятельности производства', MEDIUM)
    fcm.add_edge(c11, c21, INCREASE)
    fcm.add_edge(c12, c21, INCREASE)

    c22 = fcm.add_concept('Производительность производственных бизнес процессов', MEDIUM)
    fcm.add_edge(c21, c22, INCREASE)

    c23 = fcm.add_concept('Риск ошибок по причине ЧФ', MEDIUM)
    fcm.add_edge(c21, c23, DECREASE)
    fcm.add_edge(c13, c23, DECREASE)
    fcm.add_edge(c14, c23, DECREASE)

    c25 = fcm.add_concept('Уровень автоматизации управления и мониторинга физических процессов', MEDIUM)
    fcm.add_edge(c12, c25, INCREASE)

    c26 = fcm.add_concept('Оснащение АСУ ТП системами телемеханики', MEDIUM)
    fcm.add_edge(c12, c26, INCREASE)

    c27 = fcm.add_concept('Степень обработки информации вручную', MEDIUM)
    fcm.add_edge(c25, c27, DECREASE)
    fcm.add_edge(c26, c27, DECREASE)
    fcm.add_edge(c27, c23, DECREASE)

    c28 = fcm.add_concept('Качество и оснащение датчиков и актуаторов', MEDIUM)
    fcm.add_edge(c26, c28, INCREASE)

    c29 = fcm.add_concept('Надежность датчиков и актуаторов', MEDIUM)
    fcm.add_edge(c28, c29, INCREASE)

    c30 = fcm.add_concept('Качество измерений', MEDIUM)
    fcm.add_edge(c29, c30, INCREASE)

    c31 = fcm.add_concept('Надежность систем телемеханики', MEDIUM)
    fcm.add_edge(c29, c31, INCREASE)

    c32 = fcm.add_concept('Надежность АСУ ТП', MEDIUM)
    fcm.add_edge(c29, c32, INCREASE)

    c33 = fcm.add_concept('Качество и своевременность осблуживания элементов технологической цепочки', MEDIUM)
    fcm.add_edge(c12, c33, INCREASE)

    c34 = fcm.add_concept('Надежность ТП', MEDIUM)
    fcm.add_edge(c31, c34, INCREASE)
    fcm.add_edge(c32, c34, INCREASE)

    c35 = fcm.add_concept('Функциональная безопасность', MEDIUM)
    fcm.add_edge(c34, c35, INCREASE)

    c36 = fcm.add_concept('Организационная безопасность', MEDIUM)
    fcm.add_edge(c13, c36, INCREASE)
    fcm.add_edge(c14, c36, INCREASE)

    c37 = fcm.add_concept('Информационная безопасность', MEDIUM)
    fcm.add_edge(c16, c37, INCREASE)

    c38 = fcm.add_concept('Кибербезопасность', MEDIUM)
    fcm.add_edge(c37, c38, INCREASE)
    fcm.add_edge(c35, c38, INCREASE)

    c38 = fcm.add_concept('Риск выхода из строя элементов технологической цепочки', LOW)
    fcm.add_edge(c33, c38, DECREASE)
    fcm.add_edge(c18, c38, INCREASE)

    c39 = fcm.add_concept('Риск связанный с человеческим фактором', LOW)
    fcm.add_edge(c23, c39, INCREASE)

    c40 = fcm.add_concept('Риск чрезвычайных событий', LOW)
    fcm.add_edge(c38, c40, INCREASE)
    fcm.add_edge(c39, c40, INCREASE)

    c41 = fcm.add_concept('Риск эффективного и безопасного функционирования производства', LOW)
    fcm.add_edge(c40, c41, INCREASE)
    fcm.add_edge(c41, c35, DECREASE)

    for idx, (name, concept) in enumerate(fcm.iter_concepts()):
        c_var = concept.get('var')
        c_val = concept.get('value')
        print('c{0}\t{1}\t{2}'.format(idx, name, c_var.fuzzify_max(c_val).name))

    #new_weights = hebbian_learning(fcm, sigmoid)
    #pso = FcmPso(fcm)
    #pso.optimize()

    #fcm.draw_graph()

    log = defaultdict(list)
    dt = .1
    sim_time = 10.0
    iterations = int(max(min(sim_time / dt, 10000), 1))
    concepts = list(fcm.graph.nodes.keys())
    for i in range(iterations):
        fcm.update(dt)
        for key in concepts:
            log[key].append(fcm.graph.get_node_attr(key))

    """
    change_var = Variable('change', (-1, 1), *mfs_edge)
    diff = []
    for i, (concept, history) in enumerate(log.items()):
        delta = history[-1] - history[0]
        diff.append((concept, delta, change_var.fuzzify_max(delta).name, i))

    #diff = sorted(diff, reverse=True, key=lambda d: abs(d[1]))
    for d in diff:
        print('c{3}\t{2}\t{0}\t{1:.1f}\t'.format(*d))
    """

    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = FontProperties()
    font.set_size('small')
    #legend([plot1], "title", prop=font)
    ax = plt.subplot(111)
    for name, history in log.items():
        ax.plot(history, label=name)
    box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    #ax.legend(loc='center left', prop=font, bbox_to_anchor=(0.8, 0.5))
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
