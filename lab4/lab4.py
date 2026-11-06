import argparse
import csv
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy import signal


class ControlSystem:
    """Три апериодических звена, регулятор и единичная отрицательная ОС.

    По рис. 4.1: y = Wo(Wp*e + f), e = g-y = eg-ef.
    Канал ef принят положительным, как Wfe в формуле (4.1) методички.
    """

    def __init__(self, name, kp=1., controller_integrators=0, plant_integrators=0):
        if not np.isfinite(kp) or kp <= 0:
            raise ValueError('Требуется конечное kp > 0')
        if any(n not in (0, 1, 2) for n in (controller_integrators, plant_integrators)):
            raise ValueError('Число интеграторов должно быть 0, 1 или 2')
        self.name, self.kp = name, kp
        self.nc, self.no = controller_integrators, plant_integrators
        self.k = 4.
        self.plant_den = np.polymul(np.polymul([1., 1.], [.8, 1.]), [.04, 1.])
        self.den_open = np.polymul(self.plant_den, [1.] + [0.] * (self.nc + self.no))
        self.den_closed = np.polyadd(self.den_open, [self.k * kp])
        self.num_g = self.den_open.copy()
        self.num_f = np.array([self.k] + [0.] * self.nc)
        self.poles = np.roots(self.den_closed)

    def is_stable(self):
        return bool(np.all(self.poles.real < -1e-10))

    def get_system(self, channel):
        return signal.TransferFunction(self.num_g if channel == 'g' else self.num_f,
                                       self.den_closed)

    def coefficients(self, channel, order=2):
        """C_m = W^(m)(0); рекурсия деления полиномов без SymPy."""
        num = (self.num_g if channel == 'g' else self.num_f)[::-1]
        den = self.den_closed[::-1]
        a = np.zeros(order + 1)
        for m in range(order + 1):
            a[m] = ((num[m] if m < len(num) else 0.)
                    - sum(den[j] * a[m-j] for j in range(1, min(m+1, len(den))))) / den[0]
        return np.array([a[m] * math.factorial(m) for m in range(order + 1)])


class ErrorAnalyzer:
    """Моделирование при нулевых начальных условиях и независимый расчёт."""

    def __init__(self, system):
        self.system = system
        if not system.is_stable():
            raise ValueError('Установившийся режим существует только у устойчивой системы')
        self.duration = max(40., 30. / -max(system.poles.real))

    def polynomial(self, channel, amplitude, ramp):
        t = np.linspace(0, self.duration, int(self.duration / .01) + 1)
        u = amplitude * (t if ramp else np.ones_like(t))
        _, measured, _ = signal.lsim(self.system.get_system(channel), U=u, T=t)
        c = self.system.coefficients(channel)
        calculated = amplitude * (c[0] * t + c[1] if ramp else np.full_like(t, c[0]))
        # Сравнение на всём последнем участке, а не только в одной точке.
        tail = t >= .8 * self.duration
        residual = np.max(np.abs(measured[tail] - calculated[tail]))
        if residual > 1e-6 * max(1., np.max(np.abs(calculated[tail]))):
            raise RuntimeError(f'Не сошёлся установившийся режим: {residual}')
        return t, measured, calculated, residual

    def harmonic(self, channel, amplitude, omega):
        # Последние 10 целых периодов: оценка амплитуды и фазы МНК.
        period = 2 * np.pi / omega
        duration = self.duration + 10 * period
        count = int(np.ceil(duration / min(.01, period / 160))) + 1
        t = np.linspace(0, duration, count)
        _, measured, _ = signal.lsim(self.system.get_system(channel),
                                      U=amplitude * np.sin(omega*t), T=t)
        _, response = signal.freqresp(self.system.get_system(channel), w=[omega])
        exact = amplitude * response[0]
        c = self.system.coefficients(channel)
        mco = amplitude * (c[0] + 1j*omega*c[1] - omega**2*c[2]/2)
        tail = t >= duration - 10 * period
        basis = np.column_stack((np.sin(omega*t[tail]), np.cos(omega*t[tail])))
        fit, *_ = np.linalg.lstsq(basis, measured[tail], rcond=None)
        experimental = complex(*fit)
        if abs(experimental-exact) > 2e-4 * max(1., abs(exact)):
            raise RuntimeError('Амплитуда/фаза не согласуется с частотным расчётом')
        return t, measured, exact, experimental, mco


class SystemPlotter:
    """Моделирование - сплошная линия, аналитический расчёт - пунктир."""

    def __init__(self, output_dir, show=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.show = show

    def save(self, fig, filename):
        fig.tight_layout(rect=(0, .05, 1, .96) if len(fig.texts) > 1 else (0, 0, 1, 1))
        fig.savefig(self.output_dir / filename, dpi=160)
        if not self.show:
            plt.close(fig)

    @staticmethod
    def system_label(system):
        return {'P': 'П-регулятор', 'I_controller': 'Интегратор в регуляторе',
                'I_plant': 'Интегратор в объекте'}[system.name]

    def polynomial(self, system, cases):
        """Все шесть опытов на одном рисунке; одинаковые каналы не дублируются."""
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for row, channel in enumerate(('g', 'f')):
            for col, ramp in enumerate((False, True)):
                ax = axes[row, col]
                amplitudes = sorted({case[row][0] for case in cases if case[row][1] == ramp})
                for amplitude in amplitudes:
                    t, measured, calculated, _ = ErrorAnalyzer(system).polynomial(channel, amplitude, ramp)
                    label = f'{channel}(t)={amplitude:g}' + ('t' if ramp else '')
                    line, = ax.plot(t, measured, label=label, linewidth=1.8)
                    ax.plot(t, calculated, '--', color=line.get_color(), linewidth=1.3)
                ax.set(xlabel='Время t, с', ylabel=f'e{channel}(t)',
                       title=('По заданию' if channel == 'g' else 'По помехе') +
                       (' · линейный вход' if ramp else ' · постоянный вход'))
                ax.grid(True, linestyle=':')
                ax.legend(fontsize=9)
        fig.suptitle(self.system_label(system) + ' · Негармонические воздействия', fontsize=14)
        fig.text(.5, .01, 'Сплошная линия — моделирование; пунктир — МКО. Все опыты таблицы 4.2.',
                 ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .035, 1, .95))
        self.save(fig, f'{system.name}_polynomial.png')

    def harmonic(self, system, cases):
        """Все пять опытов: каналы по строкам, частоты по столбцам."""
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for row, channel in enumerate(('g', 'f')):
            for col, omega in enumerate((1, 50)):
                ax = axes[row, col]
                amplitudes = sorted({case[row][0] for case in cases if case[row][1] == omega})
                for amplitude in amplitudes:
                    t, measured, exact, _, _ = ErrorAnalyzer(system).harmonic(channel, amplitude, omega)
                    # Общая шкала трёх периодов с началом в нулевой фазе входа.
                    period = 2*np.pi/omega
                    start = np.floor(t[-1]/period)*period - 3*period
                    mask = (t >= start) & (t <= start+3*period)
                    local_t = t[mask]-start
                    line, = ax.plot(local_t, measured[mask], linewidth=1.8,
                                    label=f'{channel}(t)={amplitude:g} sin({omega:g}t)')
                    ax.plot(local_t, exact.real*np.sin(omega*t[mask]) + exact.imag*np.cos(omega*t[mask]),
                            '--', color=line.get_color(), linewidth=1.3)
                ax.set(xlabel='Время от начала выбранного периода, с', ylabel=f'e{channel}(t)',
                       title=('По заданию' if channel == 'g' else 'По помехе') + f' · ω={omega:g} рад/с')
                ax.grid(True, linestyle=':')
                ax.legend(fontsize=9)
        fig.suptitle(self.system_label(system) + ' · Гармонические воздействия', fontsize=14)
        fig.text(.5, .01, 'Сплошная линия — моделирование; пунктир — точный расчёт. Все опыты таблицы 4.3.',
                 ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .035, 1, .95))
        self.save(fig, f'{system.name}_harmonic.png')

def save_table(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(fields)
        writer.writerows(rows)


def main(show=True):
    output_dir = Path(__file__).resolve().parent / 'results'
    plotter = SystemPlotter(output_dir, show=show)
    # kp и A не заданы методичкой: приняты kp=1, A=1 для основных опытов.
    systems = [ControlSystem('P', kp=1.),
               ControlSystem('I_controller', kp=.05, controller_integrators=1),
               ControlSystem('I_plant', kp=.05, plant_integrators=1),
               ControlSystem('II_controller', kp=.05, controller_integrators=2),
               ControlSystem('II_plant', kp=.05, plant_integrators=2)]
    polynomial_cases = [((1,False),(5,False)), ((1,False),(1,False)),
                        ((5,False),(1,False)), ((1,True),(5,True)),
                        ((1,True),(1,True)), ((5,True),(1,True))]
    harmonic_cases = [((1,1),(1,50)), ((1,1),(1,1)), ((1,50),(1,1)),
                      ((10,1),(1,1)), ((1,1),(10,1))]
    log = ['Лабораторная № 4. Вариант 10. Метод коэффициентов ошибки.',
           'Wo=4/D(s), D(s)=0.032s^3+0.872s^2+1.84s+1; kp=1; A=1.',
           'По схеме e=eg-ef; положительный канал ef соответствует (4.1).',
           'C_m=W^(m)(0), eуст=C0*u+C1*u\'+C2*u\'\'/2.',
           'Гармоника: eуст=B*sin(wt+phi); МКО-2 - только приближение.']
    stability, coefficients = [], []
    for system in systems:
        stable = system.is_stable()
        stability.append([system.name, system.kp, system.nc, system.no, stable,
                          max(system.poles.real), ' | '.join(map(str, system.poles))])
        log.append(f'\n{system.name}: kp={system.kp}; устойчивость={stable}; корни={system.poles}')
        if not stable:
            log.append('Установившиеся ошибки не определяются: система неустойчива.')
            continue
        analyzer = ErrorAnalyzer(system)
        for channel in ('g','f'):
            c = system.coefficients(channel)
            coefficients.append([system.name,channel,*c])
            log.append(f'C0,C1,C2 ({channel}): {c}')
        rows = []
        for i, cases in enumerate(polynomial_cases, 1):
            for channel, (amplitude, ramp) in zip(('g','f'), cases):
                t, measured, calculated, residual = analyzer.polynomial(channel, amplitude, ramp)
                c = system.coefficients(channel)
                slope, intercept = (amplitude*c[0], amplitude*c[1]) if ramp else (0., amplitude*c[0])
                limit = f'{intercept:.8g}' if abs(slope)<1e-12 else ('+inf' if slope>0 else '-inf')
                rows.append([i,channel,f'{amplitude:g}'+('t' if ramp else ''),
                             t[-1],measured[-1],calculated[-1],slope,intercept,limit,residual])
        plotter.polynomial(system,polynomial_cases)
        save_table(output_dir/f'table_4_2_{system.name}.csv',
                   ['Опыт','Канал','Воздействие','tкон, с','Ошибка экспер. при tкон',
                    'Ошибка МКО при tкон','Наклон a (at+b)','Смещение b (at+b)',
                    'Предел ошибки','Макс. расхождение на хвосте'],rows)
        rows = []
        for i,cases in enumerate(harmonic_cases,1):
            for channel,(amplitude,omega) in zip(('g','f'),cases):
                _,_,exact,experimental,mco = analyzer.harmonic(channel,amplitude,omega)
                rows.append([i,channel,amplitude,omega,abs(experimental),np.angle(experimental,deg=True),
                             abs(exact),np.angle(exact,deg=True),abs(mco),np.angle(mco,deg=True),
                             abs(mco-exact)/abs(exact)*100])
        plotter.harmonic(system,harmonic_cases)
        save_table(output_dir/f'table_4_3_{system.name}.csv',
                   ['Опыт','Канал','Амплитуда входа','Частота, рад/с','Амплитуда экспер.',
                    'Фаза экспер., град','Амплитуда точная','Фаза точная, град',
                    'Амплитуда МКО-2','Фаза МКО-2, град','Отн. ошибка комплексной амплитуды МКО-2, %'],rows)
    save_table(output_dir/'coefficients.csv',['Система','Канал','C0','C1','C2'],coefficients)
    save_table(output_dir/'stability.csv',['Система','kp','Интеграторы регулятора','Интеграторы объекта',
                                         'Устойчива','Макс. Re полюса','Полюса'],stability)
    log.extend(['\nВыводы:',
                '1. В статической системе постоянная ошибка пропорциональна амплитуде воздействия.',
                '2. При линейном воздействии в статической системе ошибка растёт: конечного предела нет.',
                '3. Интегратор в регуляторе устраняет постоянные ошибки обоих каналов.',
                '4. Интегратор в объекте устраняет постоянную ошибку по заданию, но ef= f/kp сохраняется.',
                '5. При одном интеграторе ошибка по линейному заданию постоянна и ненулевая.',
                '6. Два чистых интегратора при выбранном kp приводят к неустойчивости; МКО неприменим.',
                '7. Гармонические ошибки периодичны; сравниваются амплитуды и фазы, а не предел e(t).',
                '8. Конечный ряд МКО около s=0 нельзя считать точным на частоте 50 рад/с.'])
    (output_dir/'calculations.txt').write_text('\n'.join(log)+'\n',encoding='utf-8')
    print('\n'.join(log))
    print(f'\nPNG, CSV и расчёты: {output_dir}')
    if show:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show',action='store_true',help='Сохранить графики без открытия окон')
    main(show=not parser.parse_args().no_show)
