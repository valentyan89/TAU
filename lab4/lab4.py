import math
import sys

import matplotlib.pyplot as plt
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

    def polynomial(self, system, results):
        """Все шесть опытов на одном рисунке; одинаковые каналы не дублируются."""
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for row, channel in enumerate(('g', 'f')):
            for col, ramp in enumerate((False, True)):
                ax = axes[row, col]
                amplitudes = sorted({amplitude for key, amplitude, mode in results if key == channel and mode == ramp})
                for amplitude in amplitudes:
                    t, measured, calculated, _ = results[channel, amplitude, ramp]
                    label = f'{channel}(t)={amplitude:g}' + ('t' if ramp else '')
                    line, = ax.plot(t, measured, label=label, linewidth=1.8)
                    ax.plot(t, calculated, '--', color=line.get_color(), linewidth=1.3)
                ax.set(xlabel='Время t, с', ylabel=f'e{channel}(t)',
                       title=('По заданию' if channel == 'g' else 'По помехе') +
                       (' · линейный вход' if ramp else ' · постоянный вход'))
                ax.grid(True, linestyle=':')
                ax.legend(fontsize=9)
        fig.suptitle(system.name + ' · Негармонические воздействия', fontsize=14)
        fig.text(.5, .01, 'Сплошная линия — моделирование; пунктир — МКО. Все опыты таблицы 4.2.',
                 ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .035, 1, .95))

    def harmonic(self, system, results):
        """Все пять опытов: каналы по строкам, частоты по столбцам."""
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for row, channel in enumerate(('g', 'f')):
            for col, omega in enumerate((1, 50)):
                ax = axes[row, col]
                amplitudes = sorted({amplitude for key, amplitude, frequency in results if key == channel and frequency == omega})
                for amplitude in amplitudes:
                    t, measured, exact, _, _ = results[channel, amplitude, omega]
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
        fig.suptitle(system.name + ' · Гармонические воздействия', fontsize=14)
        fig.text(.5, .01, 'Сплошная линия — моделирование; пунктир — точный расчёт. Все опыты таблицы 4.3.',
                 ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .035, 1, .95))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    # Усиление П-регулятора и амплитуда гармоник для варианта 10.
    kp = 1.0
    amplitude = 1.0
    integrator_kp = 0.05
    systems = [
        ControlSystem("П-регулятор", kp=kp),
        ControlSystem("Интегратор в регуляторе", kp=integrator_kp, controller_integrators=1),
        ControlSystem("Интегратор в объекте", kp=integrator_kp, plant_integrators=1),
        ControlSystem("Два интегратора в регуляторе", kp=integrator_kp, controller_integrators=2),
        ControlSystem("Два интегратора в объекте", kp=integrator_kp, plant_integrators=2),
    ]

    # Таблица 4.2: пары амплитуд для постоянных и линейных входов.
    amplitude_pairs = [(1, 5), (1, 1), (5, 1)]
    polynomial_cases = [
        ((g, ramp), (f, ramp))
        for ramp in (False, True)
        for g, f in amplitude_pairs
    ]

    # Таблица 4.3: для каждого входа заданы множитель A и частота.
    harmonic_cases = [
        ((1, 1), (1, 50)),
        ((1, 1), (1, 1)),
        ((1, 50), (1, 1)),
        ((10, 1), (1, 1)),
        ((1, 1), (10, 1)),
    ]

    plotter = SystemPlotter()
    print("Лабораторная № 4. Вариант 10. Метод коэффициентов ошибки.")
    print("Знаковое соглашение: e = eg − ef.")

    for system in systems:
        stable = system.is_stable()
        status = "устойчива" if stable else "неустойчива"
        print(f"\n{system.name}, kп={system.kp:g}. Система {status}.")
        print(f"Корни характеристического уравнения: {system.poles}")
        if not stable:
            print("Установившиеся ошибки не определяются.")
            continue

        analyzer = ErrorAnalyzer(system)
        for channel in ("g", "f"):
            print(f"C0, C1, C2 ({channel}): {system.coefficients(channel)}")

        print("Таблица 4.2. Ошибка в конце моделирования")
        polynomial_results = {}
        for i, inputs in enumerate(polynomial_cases, 1):
            for channel, (level, ramp) in zip(("g", "f"), inputs):
                key = channel, level, ramp
                if key not in polynomial_results:
                    polynomial_results[key] = analyzer.polynomial(*key)
                t, measured, calculated, _ = polynomial_results[key]
                input_label = f"{level:g}t" if ramp else f"{level:g}"
                print(f"  Опыт {i}, {channel}(t)={input_label}, t={t[-1]:.6g} с: "
                      f"эксперимент={measured[-1]:.8g}, МКО={calculated[-1]:.8g}")
        plotter.polynomial(system, polynomial_results)

        print("Таблица 4.3. Амплитуда и фаза ошибки")
        harmonic_results = {}
        for i, inputs in enumerate(harmonic_cases, 1):
            for channel, (factor, omega) in zip(("g", "f"), inputs):
                level = factor * amplitude
                key = channel, level, omega
                if key not in harmonic_results:
                    harmonic_results[key] = analyzer.harmonic(*key)
                _, _, exact, experimental, mco = harmonic_results[key]
                print(f"  Опыт {i}, {channel}(t)={level:g} sin({omega:g}t)")
                for label, response in (("Эксперимент", experimental), ("Точный расчёт", exact),
                                        ("МКО", mco)):
                    print(f"    {label}: амплитуда={abs(response):.8g}, "
                          f"фаза={np.angle(response, deg=True):.6g}°")
        plotter.harmonic(system, harmonic_results)

    plt.show()
