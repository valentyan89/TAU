import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def _split_param(value):
    '''Первое значение - основной вариант, остальные - для сравнения/исследования.'''
    if isinstance(value, (list, tuple, np.ndarray)):
        values = [float(v) for v in value]
        if not values:
            raise ValueError("Список параметров не должен быть пустым")
        return values[0], values[1:]
    return float(value), []


class ControlSystem:
    '''
    Класс для исследования устойчивости замкнутой САУ с объектом 3-го порядка
    и различными типами регуляторов (P, PD, PI).
    '''

    def __init__(self, name, controller_type: str, a0=1.0, a1=2.0, a2=2.0, kp=1.0, kd=1.0, ki=0.1):
        self.name = name
        self.controller_type = controller_type

        # Коэффициенты объекта управления: W_o(s) = 1 / (a0*s^3 + a1*s^2 + a2*s + 1)
        self.a0, _ = _split_param(a0)
        self.a1, _ = _split_param(a1)
        self.a2, _ = _split_param(a2)

        # Коэффициенты регулятора (первый - основной, остальные - дополнительные исследуемые)
        self.kp, extra_kp = _split_param(kp)
        self.kd, extra_kd = _split_param(kd)
        self.ki, extra_ki = _split_param(ki)

        # Пары параметров для дополнительного сравнения
        self.comparisons = (
            [("kp", v) for v in extra_kp]
            + [("kd", v) for v in extra_kd]
            + [("ki", v) for v in extra_ki]
        )

        self.num_open, self.den_open = self._get_open_loop_tf()
        self.num_closed, self.den_closed = self._get_closed_loop_tf()
        self.open_system = signal.TransferFunction(self.num_open, self.den_open)
        self.closed_system = signal.TransferFunction(self.num_closed, self.den_closed)

    def with_params(self, **overrides):
        '''Создает копию системы с измененными коэффициентами регулятора/объекта.'''
        return ControlSystem(
            name=self.name,
            controller_type=self.controller_type,
            a0=overrides.get("a0", self.a0),
            a1=overrides.get("a1", self.a1),
            a2=overrides.get("a2", self.a2),
            kp=overrides.get("kp", self.kp),
            kd=overrides.get("kd", self.kd),
            ki=overrides.get("ki", self.ki),
        )

    def comparison_systems(self):
        '''Возвращает список дополнительных систем для сравнения на графике.'''
        systems = []
        for param, value in self.comparisons:
            systems.append((param, value, self.with_params(**{param: value})))
        return systems

    def _get_controller_tf(self):
        '''Возвращает числитель и знаменатель ПФ регулятора W_p(s).'''
        if self.controller_type == "P":
            # W_p(s) = kp
            return [self.kp], [1.0]

        elif self.controller_type == "PD":
            # W_p(s) = kp + kd*s = (kd*s + kp) / 1
            return [self.kd, self.kp], [1.0]

        elif self.controller_type == "PI":
            # W_p(s) = kp + ki/s = (kp*s + ki) / s
            return [self.kp, self.ki], [1.0, 0.0]

        else:
            raise ValueError(f"Неизвестный тип регулятора: {self.controller_type}")

    def _get_open_loop_tf(self):
        '''Возвращает числитель и знаменатель разомкнутой системы W_open = W_p * W_o.'''
        num_p, den_p = self._get_controller_tf()
        num_o, den_o = [1.0], [self.a0, self.a1, self.a2, 1.0]

        num_open = np.polymul(num_p, num_o)
        den_open = np.polymul(den_p, den_o)
        return num_open, den_open

    def _get_closed_loop_tf(self):
        '''Возвращает числитель и знаменатель замкнутой системы при единичной ОС.'''
        # W_closed = W_open / (1 + W_open) = num_open / (den_open + num_open)
        num_closed = self.num_open
        den_closed = np.polyadd(self.den_open, self.num_open)
        return num_closed, den_closed

    def is_stable(self):
        '''Проверка основного условия устойчивости (все корни в левой полуплоскости).'''
        roots = np.roots(self.den_closed)
        return np.all(np.real(roots) < 0)

    def param_label(self, changed=None):
        if changed == "kp":
            return f"kп = {self.kp:g}"
        if changed == "kd":
            return f"kд = {self.kd:g}"
        if changed == "ki":
            return f"kи = {self.ki:g}"

        if self.controller_type == "P":
            return f"Вариант: kп={self.kp:g}"
        elif self.controller_type == "PD":
            return f"Вариант: kп={self.kp:g}, kд={self.kd:g}"
        elif self.controller_type == "PI":
            return f"Вариант: kп={self.kp:g}, kи={self.ki:g}"

    def get_sistem(self, mode: str):
        if mode == "Open":
            return self.open_system
        elif mode == "Close": 
            return self.closed_system
        else:
            raise ValueError(f"Неизвестный тип системы: {mode}. Допустимые типы: \"Open\", \"Close\"")

    def get_name(self):
        return self.name

    def __str__(self):
        '''Выводит в консоль подробный расчет корней и статус устойчивости.'''
        res = f"\n{self.name} ({self.param_label()}) \nКорни характеристического уравнения:\n"
        for i, r in enumerate(np.roots(self.den_closed), 1):
            res += f"   s_{i} = {r.real:+.4f} {"+" if r.imag >= 0 else "-"} {abs(r.imag):.4f}j\n"
        res += f"Состояние системы: {"УСТОЙЧИВА" if self.is_stable() else "НЕУСТОЙЧИВА"}"
        return res


class SystemPlotter:
    '''
    Класс для визуализации годографа Найквиста и расчета запасов устойчивости.
    '''

    def __init__(self, w_min=0.001, w_max=100.0, points=5000, mirror=True):
        self.w = np.logspace(np.log10(w_min), np.log10(w_max), points)
        self.mirror = mirror

    def _curves(self, system: ControlSystem):
        '''Основная система (сплошная линия) и сравнения (пунктир).'''
        yield system, system.param_label(), {"linestyle": "-", "linewidth": 2.0}

        for param, _, alt in system.comparison_systems():
            yield alt, alt.param_label(changed=param), {
                "linestyle": "--",
                "linewidth": 1.5,
            }

    def _calculate_margins(self, system: ControlSystem):
        '''Вычисление запасов устойчивости по амплитуде и фазе.'''
        _, H = signal.freqresp(system.get_sistem("Open"), w=self.w)
        mag = np.abs(H)
        phase = np.unwrap(np.angle(H)) * 180.0 / np.pi

        # Запас по фазе (при |W(jω)| = 1 -> 0 dB)
        idx_wc = np.argmin(np.abs(mag - 1.0))
        phase_margin = 180.0 + phase[idx_wc]

        # Запас по амплитуде (при фазе = -180 deg)
        idx_wg = np.argmin(np.abs(phase - (-180.0)))
        mag_wg = mag[idx_wg]
        gain_margin_times = 1.0 / mag_wg if mag_wg > 0 else np.inf
        gain_margin_db = -20 * np.log10(mag_wg) if mag_wg > 0 else np.inf

        return gain_margin_db, gain_margin_times, phase_margin

    def plot_nyquist(self, system: ControlSystem):
        '''Строит годограф Найквиста для системы и выводит запасы устойчивости.'''
        fig, ax = plt.subplots(figsize=(8, 7))

        for (sys_item, label, style) in self._curves(system):
            _, H = signal.freqresp(sys_item.get_sistem("Open"), w=self.w)

            # Прямая
            ax.plot(H.real, H.imag, label=f"{label} (ω > 0)", **style)

            # Вычисление и печать запасов для текущего режима
            gm_db, gm_times, pm = self._calculate_margins(sys_item)
            print(
                f"[{sys_item.name} | {label}] "
                f"Запас по амплитуде: {gm_db:.2f} dB (в {gm_times:.2f} раз), "
                f"Запас по фазе: {pm:.2f}°"
            )

        # Критическая точка (-1, j0)
        ax.plot(-1, 0, "ro", markersize=7, label="Критическая точка (-1, j0)")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="-")

        ax.set_title(f"Годограф Найквиста: {system.get_name()}")
        ax.set_xlabel("Re W(jω)")
        ax.set_ylabel("Im W(jω)")
        ax.grid(True, linestyle=":")
        ax.axis("equal")
        ax.legend()

        fig.tight_layout()
        return fig

    def plot_all(self, systems: list[ControlSystem]):
        '''Построение графиков для всех исследуемых систем.'''
        for sys in systems:
            print(sys)
            self.plot_nyquist(sys)
        plt.show()

if __name__ == "__main__":
    # Коэффициенты берутся из таблицы 2.1 согласно варианту
    tabular_variables = {
        "a0": 1.0,
        "a1": 2.0, 
        "a2": 2.0, 
        "kd": 1.0, 
        "ki": 0.1
    }
    # Коэфициенты граничных случаев, подбераются эксперементально
    experimental_variables = {
        "kp": 3.0, # Эксперементальное значение для П-регулятора
        "kd": 0.5, # Эксперементальное значение для ПД-регулятора
        "ki": 1.0, # Эксперементальное значение для ПИ-регулятора
    }

    # 1. Пропорциональный регулятор (П-регулятор)
    # Исследуем:
    # - kp = 1.0 (табличный, система устойчива)
    # - kp = 3.0 (граничное значение)
    # - kp = 4.0 (увеличенное значение, система становится неустойчивой)
    p_sys = ControlSystem(
        name="Исследование П-регулятора",
        controller_type="P",
        a0=tabular_variables["a0"],
        a1=tabular_variables["a1"],
        a2=tabular_variables["a2"],
        kp=[1.0, experimental_variables["kp"], 4.0],
    )

    # 2. Пропорционально-дифференциальный регулятор (ПД-регулятор)
    # При kp = 4:
    # - kd = 1.0 (табличное значение, система устойчива)
    # - kd = 0.5 (граничное значение)
    # - kd = 0.25 (уменьшенное значение, система становится неустойчивой)
    pd_sys = ControlSystem(
        name="Исследование ПД-регулятора",
        controller_type="PD",
        a0=tabular_variables["a0"],
        a1=tabular_variables["a1"],
        a2=tabular_variables["a2"],
        kp=4.0,
        kd=[tabular_variables["kd"], experimental_variables["kd"], 
            tabular_variables["kd"]+2*(experimental_variables["kd"] - tabular_variables["kd"])], 
    )

    # 3. Пропорционально-интегральный регулятор (ПИ-регулятор)
    # При kp = 1:
    # - ki = 0.1 (табличное значение, система устойчива)
    # - ki = 1 (граничное значение)
    # - ki = 1.9 (увеличенное значение, система становится неустойчивой)
    pi_sys = ControlSystem(
        name="Исследование ПИ-регулятора",
        controller_type="PI",
        a0=tabular_variables["a0"],
        a1=tabular_variables["a1"],
        a2=tabular_variables["a2"],
        kp=1.0,
        ki=[tabular_variables["ki"], experimental_variables["ki"], 
            tabular_variables["ki"]+2*(experimental_variables["ki"] - tabular_variables["ki"])], 
    )

    systems = [p_sys, pd_sys, pi_sys]
    plotter = SystemPlotter()
    plotter.plot_all(systems)