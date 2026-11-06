import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy import signal


S = sp.Symbol('s')


class ControlSystem:
    """Три апериодических звена, регулятор и помеха на входе объекта."""

    def __init__(self, name, kp=1., controller_integrators=0, plant_integrators=0,
                 gains=(4., 1., 1.), times=(1., .8, .04)):
        self.name = name
        self.kp = float(kp)
        self.controller_integrators = controller_integrators
        self.plant_integrators = plant_integrators
        if len(gains) != 3 or len(times) != 3:
            raise ValueError('Требуются три звена объекта')
        if not np.all(np.isfinite([kp, *gains, *times])) or min(kp, *gains, *times) <= 0:
            raise ValueError('Коэффициенты должны быть конечными и положительными')
        if any(not isinstance(v, int) or v < 0
               for v in (controller_integrators, plant_integrators)):
            raise ValueError('Число интеграторов должно быть целым неотрицательным')
        plant_denominator = sp.expand(sp.prod(sp.Rational(str(t)) * S + 1 for t in times))
        gain = sp.prod(sp.Rational(str(k)) for k in gains)
        regulator_gain = sp.Rational(str(kp))
        self.plant = gain / (plant_denominator * S**plant_integrators)
        self.controller = regulator_gain / S**controller_integrators
        # Несокращенный полином сохраняет внутренние моды замкнутой системы.
        self.characteristic = sp.expand(
            plant_denominator * S**(controller_integrators + plant_integrators)
            + regulator_gain * gain
        )
        self.poles = np.roots([float(v) for v in sp.Poly(self.characteristic, S).all_coeffs()])
        loop = self.controller * self.plant
        self.channels = {
            'eg': sp.cancel(1 / (1 + loop)),
            'ef': sp.cancel(-self.plant / (1 + loop)),
            'yg': sp.cancel(loop / (1 + loop)),
            'yf': sp.cancel(self.plant / (1 + loop)),
        }
        self.systems = {}
        for key, expression in self.channels.items():
            num, den = sp.fraction(expression)
            self.systems[key] = signal.TransferFunction(
                [float(value) for value in sp.Poly(num, S).all_coeffs()],
                [float(value) for value in sp.Poly(den, S).all_coeffs()],
            )

    def is_stable(self):
        return bool(np.all(self.poles.real < -1e-9))

    def coefficients(self, channel):
        return [sp.simplify(sp.diff(self.channels[channel], S, m).subs(S, 0))
                for m in range(3)]

    def __str__(self):
        lines = [f'\n{self.name}: kп={self.kp:g}', f'Wо(s) = {self.plant}',
                 f'Wр(s) = {self.controller}', f'Характеристический полином: {self.characteristic}',
                 'Корни характеристического уравнения:']
        lines += [f'  {p.real:+.9f} {p.imag:+.9f}j' for p in self.poles]
        lines.append('УСТОЙЧИВА' if self.is_stable() else 'НЕУСТОЙЧИВА / граница устойчивости')
        if self.is_stable():
            for key in ('eg', 'ef'):
                lines += [f'{key}(s) = {self.channels[key]}',
                          f'C0, C1, C2 = {self.coefficients(key)}']
        return '\n'.join(lines)


@dataclass(frozen=True)
class Excitation:
    kind: str
    amplitude: float
    omega: float = 0.

    def values(self, t):
        if self.kind == 'constant':
            return np.full_like(t, self.amplitude)
        if self.kind == 'ramp':
            return self.amplitude * t
        return self.amplitude * np.sin(self.omega * t)

    def label(self):
        if self.kind == 'constant':
            return f'{self.amplitude:g}'
        if self.kind == 'ramp':
            return f'{self.amplitude:g}t'
        return f'{self.amplitude:g} sin({self.omega:g}t)'

    def mko(self, t, coefficients):
        c0, c1, c2 = map(float, coefficients)
        if self.kind == 'constant':
            return np.full_like(t, self.amplitude * c0)
        if self.kind == 'ramp':
            return self.amplitude * (c0 * t + c1)
        return self.amplitude * ((c0 - self.omega**2 * c2 / 2) * np.sin(self.omega * t)
                                 + self.omega * c1 * np.cos(self.omega * t))


@dataclass
class ChannelResult:
    simulated: np.ndarray
    mko: np.ndarray
    measured: float
    estimated: float
    exact: np.ndarray | None = None


@dataclass
class ErrorResult:
    system: ControlSystem
    t: np.ndarray
    tail: np.ndarray
    g: np.ndarray
    f: np.ndarray
    y: np.ndarray
    e: np.ndarray
    eg: ChannelResult
    ef: ChannelResult


class ErrorAnalyzer:
    """Нулевые начальные условия, МКО и независимый частотный расчет."""

    def analyze(self, system, g, f):
        if not system.is_stable():
            raise ValueError('Установившийся режим требует устойчивости')
        decay = -max(system.poles.real)
        # Не менее 20 постоянных затухания; гармоники оцениваем за 5 полных периодов.
        start = max(20., 20. / decay)
        if g.kind == "harmonic":
            min_omega = min(excitation.omega for excitation in (g, f) if excitation.omega > 0)
            duration = start + 5 * (2 * np.pi / min_omega)
        else:
            duration = start + max(10., 5. / decay)
        max_omega = max(g.omega, f.omega)
        dt = min(.01, 1 / (20 * max(abs(system.poles))))
        if max_omega:
            dt = min(dt, 2 * np.pi / (120 * max_omega))
        count = int(np.ceil(duration / dt)) + 1
        if count > 2_000_000:
            raise RuntimeError('Слишком длинная сетка; требуется пересмотреть параметры')
        t = np.linspace(0., duration, count)
        g_values, f_values = g.values(t), f.values(t)
        inputs = {"eg": g_values, "ef": f_values, "yg": g_values, "yf": f_values}
        responses = {key: signal.lsim(tf, U=inputs[key], T=t)[1]
                     for key, tf in system.systems.items()}
        error = responses['eg'] + responses['ef']
        output = responses['yg'] + responses['yf']
        expected_error = g_values - output
        residual = float(np.max(np.abs(error - expected_error)))
        if not np.all(np.isfinite(error)) or not np.allclose(error, expected_error, atol=1e-7, rtol=1e-7):
            raise RuntimeError(f'Нарушено e=g-y: невязка {residual:g}')
        tail = t >= start
        channels = {}
        for key, excitation in (("eg", g), ("ef", f)):
            response = responses[key]
            coefficients = system.coefficients(key)
            predicted = excitation.mko(t, coefficients)
            exact_response = None
            if excitation.kind == 'harmonic':
                w, a = excitation.omega, excitation.amplitude
                basis = np.column_stack((np.sin(w * t[tail]), np.cos(w * t[tail]),
                                         np.ones(np.count_nonzero(tail))))
                fitted = np.linalg.lstsq(basis, response[tail], rcond=None)[0]
                empirical = float(np.hypot(*fitted[:2]))
                exact = complex(system.channels[key].subs(S, sp.I * w)) * a
                c0, c1, c2 = map(float, coefficients)
                approximation = a * complex(c0 - w*w*c2/2, w*c1)
                exact_response = abs(exact) * np.sin(w * t + np.angle(exact))
                estimated = abs(approximation)
                if not np.isclose(empirical, abs(exact), rtol=.003, atol=1e-7):
                    raise RuntimeError('Амплитуда моделирования не совпала с частотным расчетом')
            else:
                slope, intercept = np.polyfit(t[tail], response[tail], 1)
                c0, c1, _ = map(float, coefficients)
                expected_slope = excitation.amplitude * c0 if excitation.kind == 'ramp' else 0.
                expected_intercept = excitation.amplitude * (c1 if excitation.kind == 'ramp' else c0)
                if not np.allclose([slope, intercept], [expected_slope, expected_intercept],
                                   rtol=1e-5, atol=2e-6):
                    raise RuntimeError('Асимптота моделирования не совпала с МКО')
                empirical = response[-1]
                estimated = predicted[-1]
            channels[key] = ChannelResult(response, predicted, empirical, estimated, exact_response)

        return ErrorResult(system, t, tail, g_values, f_values, output, error,
                           channels["eg"], channels["ef"])


class SystemPlotter:
    """Графики с русскими подписями, сплошное моделирование и пунктирный расчет."""

    def plot_case(self, result, table, index):
        system = result.system
        t, tail = result.t, result.tail
        fig, axes = plt.subplots(3, 2, figsize=(14, 11))
        axes[0, 0].plot(t, result.g, label="Задание g(t)")
        axes[0, 0].plot(t, result.f, '--', label="Возмущение f(t)")
        axes[0, 1].plot(t, result.y, label="Выход y(t)")
        axes[0, 1].plot(t, result.g, '--', label="Задание g(t)")
        axes[1, 0].plot(t, result.eg.simulated, label="По заданию")
        axes[1, 0].plot(t, result.ef.simulated, label="По возмущению")
        axes[1, 0].plot(t, result.e, label="Общая")

        # Для гармоник показываем последние 2π секунд, чтобы различить колебания.
        mask = tail & (t >= t[-1] - 2 * np.pi) if table == "4.3" else tail
        for ax, label, channel in ((axes[1, 1], "eg", result.eg),
                                   (axes[2, 0], "ef", result.ef)):
            ax.plot(t[mask], channel.simulated[mask], label=f"{label}: моделирование")
            ax.plot(t[mask], channel.mko[mask], '--', label="МКО (3 коэффициента)")
            if channel.exact is not None:
                ax.plot(t[mask], channel.exact[mask], ':', label="Точный частотный расчет")

        eg_delta = result.eg.simulated - result.eg.mko
        ef_delta = result.ef.simulated - result.ef.mko
        axes[2, 1].plot(t[tail], eg_delta[tail], label="eg − МКО")
        axes[2, 1].plot(t[tail], ef_delta[tail], '--', label="ef − МКО")
        for ax in axes.flat:
            ax.set(xlabel='Время t, с', ylabel='Амплитуда')
            ax.grid(True, linestyle=':')
            ax.legend(fontsize=8)
        fig.suptitle(f'{system.name}, kп={system.kp:g}. Таблица {table}, опыт {index}')
        fig.tight_layout()

    def plot_scheme(self):
        fig, ax = plt.subplots(figsize=(13, 3))
        ax.set(xlim=(0, 13), ylim=(0, 3))
        ax.axis('off')
        for x, text in ((4, 'Регулятор Wр(s)'), (9, 'Объект Wо(s)')):
            ax.text(x, 1.5, text, ha='center', va='center', bbox=dict(boxstyle='square,pad=.6', fc='white'))
        for x in (2, 6.5):
            ax.text(x, 1.5, '+', ha='center', va='center', bbox=dict(boxstyle='circle', fc='white'))
        for a, b in (((.2, 1.5), (1.8, 1.5)), ((2.2, 1.5), (3, 1.5)),
                     ((5, 1.5), (6.3, 1.5)), ((6.7, 1.5), (8, 1.5)),
                     ((10, 1.5), (12.5, 1.5)), ((6.5, 2.7), (6.5, 1.7)),
                     ((2, .4), (2, 1.3))):
            ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->'))
        ax.plot([11.5, 11.5, 2], [1.5, .4, .4], color='black')
        for x, y, text in ((.5, 1.8, 'g(t)'), (2.5, 1.8, 'e=g−y'), (5.6, 1.8, 'u(t)'),
                           (6.7, 2.5, 'f(t)'), (11.7, 1.8, 'y(t)'), (2.15, .9, '−')):
            ax.text(x, y, text)
        ax.set_title('Единичная отрицательная ОС; возмущение прибавляется на входе объекта')
        fig.tight_layout()


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    # Исходные настройки варианта 10.
    kp = 1.0
    amplitude = 1.0

    # Исходный контур и варианты размещения интеграторов при том же kп.
    systems = [
        ControlSystem('Исходная система', kp=kp),
        ControlSystem('Интегратор в регуляторе', kp=kp, controller_integrators=1),
        ControlSystem('Интегратор в объекте', kp=kp, plant_integrators=1),
        ControlSystem('Два интегратора в регуляторе', kp=kp, controller_integrators=2),
        ControlSystem('Два интегратора в объекте', kp=kp, plant_integrators=2),
        ControlSystem('По интегратору в регуляторе и объекте', kp=kp, controller_integrators=1,
                      plant_integrators=1),
    ]
    # Уменьшенное усиление для опытов с одним интегратором.
    low_kp = 0.1
    systems += [
        ControlSystem('Интегратор в регуляторе, уменьшенное усиление', kp=low_kp,
                      controller_integrators=1),
        ControlSystem('Интегратор в объекте, уменьшенное усиление', kp=low_kp,
                      plant_integrators=1),
    ]
    print(f'Лабораторная № 4. Вариант 10. Допущения: kп={kp:g}, A={amplitude:g}')
    print('Знаковое соглашение: e=g-y=eg+ef, eg=Sg, ef=-Qf.')

    # Таблица 4.2: сначала постоянные воздействия, затем линейные.
    amplitude_pairs = [(1, 5), (1, 1), (5, 1)]
    polynomial_cases = [
        (Excitation(kind, g), Excitation(kind, f))
        for kind in ("constant", "ramp")
        for g, f in amplitude_pairs
    ]

    # Таблица 4.3: для каждого воздействия указаны множитель A и частота.
    harmonic_settings = [
        ((1, 1), (1, 50)),
        ((1, 1), (1, 1)),
        ((1, 50), (1, 1)),
        ((10, 1), (1, 1)),
        ((1, 1), (10, 1)),
    ]
    harmonic_cases = [
        (Excitation("harmonic", g_amp * amplitude, g_omega),
         Excitation("harmonic", f_amp * amplitude, f_omega))
        for (g_amp, g_omega), (f_amp, f_omega) in harmonic_settings
    ]
    cases_by_table = {
        "4.2": polynomial_cases,
        "4.3": harmonic_cases,
    }

    analyzer = ErrorAnalyzer()
    plotter = SystemPlotter()
    # Все графики опытов открываются вместе в конце расчёта.
    plt.rcParams["figure.max_open_warning"] = 0
    plotter.plot_scheme()

    for system in systems:
        print(system)
        if not system.is_stable():
            print("Установившиеся ошибки для этой системы не определяются.")
            continue

        for table, cases in cases_by_table.items():
            print(f"\nТаблица {table}")
            for i, (g, f) in enumerate(cases, 1):
                result = analyzer.analyze(system, g, f)
                plotter.plot_case(result, table, i)
                print(f"  Опыт {i}: g={g.label()}, f={f.label()}")
                for label, channel in (("eg", result.eg), ("ef", result.ef)):
                    print(f"    {label}: эксперимент={channel.measured:.8g}, МКО={channel.estimated:.8g}")

    plt.show()
