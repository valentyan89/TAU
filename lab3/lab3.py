"""Лабораторная № 3, вариант 10: качество непрерывных линейных САУ.

Запуск: python lab3/lab3.py
Первое значение параметра — основной вариант, остальные — сравнения.
Результаты: таблица 3.1 (CSV), 3 общих и отдельные графиков (PNG).
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, signal


def _split_param(value):
    """Совместимый с lab1/lab2 формат параметров."""
    values = np.atleast_1d(value).astype(float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("Нужен конечный параметр или непустой список параметров")
    return float(values[0]), list(values[1:])


class ControlSystem:
    """Объект третьего порядка, регулятор и единичная отрицательная ОС."""

    def __init__(self, name, controller_type, a0=3., a1=3., a2=3.,
                 kp=.2, kd=0., ki=0.):
        self.name = name
        self.controller_type = controller_type.upper()
        self.a0, self.a1, self.a2 = map(float, (a0, a1, a2))
        if not np.all(np.isfinite([a0, a1, a2])) or min(a0, a1, a2) <= 0:
            raise ValueError("Коэффициенты объекта должны быть конечными и положительными")
        self.kp, ep = _split_param(kp)
        self.kd, ed = _split_param(kd)
        self.ki, ei = _split_param(ki)
        active = {"P": {"kp"}, "PD": {"kp", "kd"}, "PI": {"kp", "ki"}}
        if self.controller_type not in active:
            raise ValueError("Допустимые регуляторы: P, PD, PI")
        self.comparisons = [(p, v) for p, values in (("kp", ep), ("kd", ed), ("ki", ei))
                            for v in values]
        if any(p not in active[self.controller_type] for p, _ in self.comparisons):
            raise ValueError("Сравниваемый параметр не используется регулятором")
        if self.kp <= 0 or self.kd < 0 or self.ki < 0:
            raise ValueError("Требуется kp > 0, kd >= 0, ki >= 0")
        plant = [self.a0, self.a1, self.a2, 1.]
        if self.controller_type == "P":
            self.num_open, self.den_open = [self.kp], plant
        elif self.controller_type == "PD":
            self.num_open, self.den_open = [self.kd, self.kp], plant
        else:
            if self.ki <= 0:
                raise ValueError("Для ПИ требуется ki > 0")
            self.num_open = [self.kp, self.ki]
            self.den_open = np.polymul(plant, [1., 0.])
        self.num_closed = np.trim_zeros(self.num_open, 'f')
        self.den_closed = np.polyadd(self.den_open, self.num_closed)
        self.closed_system = signal.TransferFunction(self.num_closed, self.den_closed)

    def with_params(self, **overrides):
        params = {p: getattr(self, p) for p in ("a0", "a1", "a2", "kp", "kd", "ki")}
        params.update(overrides)
        return ControlSystem(self.name, self.controller_type, **params)

    def comparison_systems(self):
        return [(p, v, self.with_params(**{p: v})) for p, v in self.comparisons]

    def param_label(self, changed=None):
        label = f"kп={self.kp:g}"
        if self.controller_type == "PD":
            label += f", kд={self.kd:g}"
        elif self.controller_type == "PI":
            label += f", kи={self.ki:g}"
        return label

    @property
    def kind(self):
        return {"P": "П", "PD": "ПД", "PI": "ПИ"}[self.controller_type]

    def is_stable(self):
        return bool(np.all(np.roots(self.den_closed).real < 0))


@dataclass
class QualityResult:
    system: ControlSystem
    t: np.ndarray
    h: np.ndarray
    h_ss: float
    error: float
    h_max: float
    t_max: float
    overshoot: float
    t_reg: float


class QualityAnalyzer:
    """Аналитический статический коэффициент и уточнение экстремумов/границ.

    Отклик строится SciPy при нулевых начальных условиях. Разложение H(s)/s
    на простые дроби даёт непрерывную функцию для уточнения показателей
    и оценку хвоста: |h(t)-hуст| <= sum |r_i| exp(Re(p_i)*t).
    Эта оценка гарантирует отсутствие позднего выхода за коридор.
    При кратных полюсах расчёт явно останавливается (нужна иная оценка хвоста).
    """

    def __init__(self, corridor=.05, max_points=2_000_000):
        if not 0 < corridor < 1:
            raise ValueError("Коридор должен лежать между 0 и 1")
        self.corridor = corridor
        self.max_points = max_points

    def analyze(self, system):
        poles = np.roots(system.den_closed)
        print(f"\n{system.kind}: {system.param_label()}\nКорни характеристического уравнения:")
        for pole in poles:
            print(f"  {pole.real:+.9f} {pole.imag:+.9f}j")
        stable = system.is_stable()
        print("Система " + ("УСТОЙЧИВА" if stable else "НЕУСТОЙЧИВА"))
        if not stable:
            raise ValueError("Прямые показатели требуют асимптотической устойчивости")
        h_ss = float(system.num_closed[-1] / system.den_closed[-1])
        band = self.corridor * abs(h_ss)
        residues, response_poles, _ = signal.residue(
            system.num_closed, np.polymul(system.den_closed, [1., 0.]), tol=1e-9)
        if any(abs(p - q) < 1e-8 for i, p in enumerate(response_poles)
               for q in response_poles[i + 1:]):
            raise ValueError("Кратные полюса: требуется полиномиальная оценка хвоста")
        mask = np.abs(response_poles) > 1e-10
        r, p = residues[mask], response_poles[mask]

        def response(t):
            return float(h_ss + np.real(np.sum(r * np.exp(p * t))))

        def derivative(t):
            return float(np.real(np.sum(r * p * np.exp(p * t))))

        def envelope(t):
            return float(np.sum(np.abs(r) * np.exp(p.real * t)))

        decay = -max(poles.real)
        duration = max(10., 8. / decay)
        while envelope(duration) > band * .01:
            duration *= 2
            if not np.isfinite(duration):
                raise RuntimeError("Не удалось подобрать интервал моделирования")

        # Удвоение плотности проверяет сходимость времени и максимума.
        previous = None
        for refinement in range(7):
            dt = min(.02, 1. / (80 * max(abs(poles)))) / 2**refinement
            count = max(5001, int(np.ceil(duration / dt)) + 1)
            if count > self.max_points:
                raise RuntimeError("Недостаточно разрешения: увеличьте max_points")
            t = np.linspace(0, duration, count)
            _, h = signal.step(system.closed_system, T=t)
            if not np.all(np.isfinite(h)):
                raise RuntimeError("Неконечные значения отклика")
            outside = np.flatnonzero(np.abs(h - h_ss) > band)
            if outside.size and outside[-1] == len(t) - 1:
                raise RuntimeError("Отклик не установился; время регулирования не определено")
            if outside.size:
                i = outside[-1]
                boundary = h_ss + np.sign(h[i] - h_ss) * band
                t_reg = optimize.brentq(lambda x: response(x) - boundary, t[i], t[i + 1])
            else:
                t_reg = 0.
            peaks = np.flatnonzero((h[1:-1] >= h[:-2]) & (h[1:-1] > h[2:])) + 1
            candidates = [(0., response(0.)), (duration, response(duration))]
            for i in peaks:
                if derivative(t[i - 1]) * derivative(t[i + 1]) < 0:
                    tm = optimize.brentq(derivative, t[i - 1], t[i + 1])
                    candidates.append((tm, response(tm)))
            t_max, h_max = max(candidates, key=lambda pair: pair[1])
            # Для монотонного отклика максимум понимается как супремум hуст.
            if h_max <= h_ss:
                t_max, h_max = np.inf, h_ss
            current = np.array([t_reg, h_max])
            if previous is not None and np.allclose(current, previous, rtol=1e-6, atol=1e-8):
                break
            previous = current
        else:
            raise RuntimeError("Показатели не сошлись при уточнении сетки")
        if not np.isclose(h[-1], response(duration), atol=1e-8, rtol=1e-7):
            raise RuntimeError("Независимая проверка отклика не пройдена")
        result = QualityResult(system, t, h, h_ss, 1 - h_ss, h_max, t_max,
                               max(0., (h_max - h_ss) / h_ss * 100), t_reg)
        print(f"hуст={h_ss:.8f}; e∞={result.error:.8f}; hmax={h_max:.8f}; "
              f"σ={result.overshoot:.5f}%; tр={t_reg:.6f} с")
        return result


class SystemPlotter:
    """Основной вариант — сплошная линия, сравнительные — пунктир."""

    def __init__(self, output_dir, corridor=.05):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.corridor = corridor

    @staticmethod
    def decorate(ax, title):
        ax.set(title=title, xlabel="Время t, с", ylabel="Переходная характеристика h(t)")
        ax.grid(True, linestyle=":")
        ax.legend(fontsize=9)

    def plot_all(self, systems, show=True):
        """Рассчитать все варианты, сохранить таблицу и построить графики."""
        analyzer = QualityAnalyzer(corridor=self.corridor)
        all_results = []
        for system in systems:
            variants = [system]
            for _, _, alternative in system.comparison_systems():
                variants.append(alternative)

            group = []
            for variant in variants:
                group.append(analyzer.analyze(variant))
            all_results.extend(group)
            self.plot_group(group)

        save_table(all_results, self.output_dir)
        print_best_settings(all_results)
        print(f"\nPNG и CSV: {self.output_dir.resolve()}")
        if show:
            plt.show()
        else:
            plt.close('all')
        return all_results

    def plot_group(self, results):
        fig, ax = plt.subplots(figsize=(11, 6))
        for i, result in enumerate(results):
            ax.plot(result.t, result.h, '-' if i == 0 else '--',
                    linewidth=2 if i == 0 else 1.5, label=result.system.param_label())
        self.decorate(ax, f"Сравнение настроек {results[0].system.kind}-регулятора")
        fig.tight_layout()
        fig.savefig(self.output_dir / f"comparison_{results[0].system.controller_type}.png", dpi=160)
        for i, result in enumerate(results, 1):
            self.plot_case(result, i)

    def plot_case(self, result, index):
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(result.t, result.h, label=result.system.param_label(), linewidth=1.8)
        band = self.corridor * abs(result.h_ss)
        ax.axhline(result.h_ss, color='black', linestyle='--', label=f"hуст={result.h_ss:.4f}")
        ax.axhspan(result.h_ss - band, result.h_ss + band, color='green', alpha=.12,
                   label=f"Коридор ±{self.corridor:.0%}")
        for level in (result.h_ss - band, result.h_ss + band):
            ax.axhline(level, color='green', linestyle=':', linewidth=1)
        if np.isfinite(result.t_max):
            ax.plot(result.t_max, result.h_max, 'ro', label=f"hmax={result.h_max:.4f}")
        else:
            ax.plot([], [], 'ro', label=f"sup h(t)=hуст={result.h_max:.4f}, t→∞")
        ax.axvline(result.t_reg, color='purple', linestyle='-.', label=f"tр={result.t_reg:.4f} с")
        self.decorate(ax, f"{result.system.kind}-регулятор: σ={result.overshoot:.3f}%, e∞={result.error:.4f}")
        fig.tight_layout()
        fig.savefig(self.output_dir / f"{result.system.controller_type}_{index:02d}.png", dpi=160)


def save_table(results, output_dir):
    fields = ["Регулятор", "kп", "kд", "kи", "tр, с", "σ, %", "e∞", "hуст", "hmax"]
    rows = []
    for r in results:
        s = r.system
        rows.append([s.kind, s.kp, s.kd if s.controller_type == 'PD' else '',
                     s.ki if s.controller_type == 'PI' else '', r.t_reg,
                     r.overshoot, r.error, r.h_ss, r.h_max])
    with (Path(output_dir) / 'table_3_1.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(fields)
        writer.writerows(rows)
    print("\nТаблица 3.1 (коридор 5%)")
    formatted = [[f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
                 for row in rows]
    widths = [max(len(row[i]) for row in [fields] + formatted) for i in range(len(fields))]
    for row in [fields] + formatted:
        print(' | '.join(value.ljust(width) for value, width in zip(row, widths)))


def print_best_settings(results):
    """Вывести настройки с минимальными показателями, включая равные результаты."""
    print("\nЛучшие исследованные настройки:")
    indicators = [
        ('t_reg', 'Минимальное tр'),
        ('overshoot', 'Минимальное σ'),
        ('error', 'Минимальная |e∞|'),
    ]
    for attr, label in indicators:
        best = min(abs(getattr(result, attr)) for result in results)
        for result in results:
            if np.isclose(abs(getattr(result, attr)), best, rtol=1e-7, atol=1e-9):
                system = result.system
                print(f"{label}: {system.kind}, {system.param_label()} → {best:.6f}")


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    # Исходные данные варианта 10.
    a0 = 3.0
    a1 = 3.0
    a2 = 3.0
    kg = a1 * a2 / a0 - 1

    # Коэффициенты регуляторов: первое значение — основное, остальные — сравнения.
    kp_p = [0.1 * kg, 0.3 * kg, 0.5 * kg, 0.7 * kg]
    kp_pd = 0.9 * kg
    kd = [0.5, 1.0, 2.0, 4.0, 8.0]
    kp_pi = 0.1 * kg
    ki = [0.1, 0.2, 0.3, 0.5]

    print("Лабораторная № 3. Вариант 10")
    print(f"kг = a1·a2/a0 − 1 = {a1:g}·{a2:g}/{a0:g} − 1 = {kg:g}")
    print("Коридор 5%: |h(t) − hуст| ≤ 0.05|hуст|; учитывается последний выход.")

    # 1. П-регулятор: меняем пропорциональный коэффициент.
    p_sys = ControlSystem(
        name="Исследование П-регулятора",
        controller_type="P",
        a0=a0, a1=a1, a2=a2,
        kp=kp_p,
    )

    # 2. ПД-регулятор: kp постоянный, меняем kd.
    pd_sys = ControlSystem(
        name="Исследование ПД-регулятора",
        controller_type="PD",
        a0=a0, a1=a1, a2=a2,
        kp=kp_pd,
        kd=kd,
    )

    # 3. ПИ-регулятор: kp постоянный, сравниваем устойчивые значения ki.
    pi_sys = ControlSystem(
        name="Исследование ПИ-регулятора",
        controller_type="PI",
        a0=a0, a1=a1, a2=a2,
        kp=kp_pi,
        ki=ki,
    )

    ki_max = (1 + pi_sys.kp) * (a1 * a2 - a0 * (1 + pi_sys.kp)) / a1**2
    print(f"ПИ, критерий Гурвица: 0 < kи < (1+kп)[a1·a2−a0(1+kп)]/a1² = {ki_max:g}")

    systems = [p_sys, pd_sys, pi_sys]
    output_dir = Path(__file__).resolve().parent / 'results'
    plotter = SystemPlotter(output_dir)
    plotter.plot_all(systems)
