import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

VARIANT = 1  # Номер варианта от 1 до 10

# Таблица 5.1. Коэффициенты знаменателя a0·s³ + a1·s² + a2·s + a3.
VARIANTS = {
    1: (0.5, 2, 2, 0.1),
    2: (2, 3, 4, 0),
    3: (5, 10, 2, 5),
    4: (4, 1.5, 2, 10),
    5: (0.2, 2, 1.5, 0),
    6: (5, 4, 2, 1),
    7: (0.4, 0.8, 4, 0.2),
    8: (0.8, 2, 4, 0),
    9: (0.5, 4, 0.2, 2),
    10: (0.25, 5, 1, 0.1),
}

K_VALUES = (0, 1, 2, 3, 5, 7, 10)
SETTLING_BAND = 0.05
SERIES = (
    ("pd", "ПД-регулятор", K_VALUES),
    ("pid_beta0", "ПИД-регулятор, β = 0", K_VALUES),
    ("pid_equal", "ПИД-регулятор, β = ω", K_VALUES[1:]),
)


class ControlSystem:
    """Синтезирует ПД или ПИД и хранит модель замкнутой САУ.

    Атрибуты kp, kd, ki содержат коэффициенты регулятора, num и den —
    коэффициенты передаточной функции по убыванию степеней s, poles —
    комплексные полюса с учётом кратности.
    """

    def __init__(self, plant, controller_type, omega, beta):
        """Принимает (a0, a1, a2, a3), тип PD или PID и частоты в рад/с.

        Нормирует знаменатель объекта по a0, синтезирует регулятор и проверяет
        положение полюсов и соответствие характеристическому полиному.
        Параметр beta используется только для ПИД.
        """
        a0, a1, a2, a3 = plant
        self.b0 = 1 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0
        self.a3 = a3 / a0
        self.omega = omega
        self.beta = beta

        if controller_type == "PD":
            self._synthesize_pd()
        elif controller_type == "PID":
            self._synthesize_pid()
        else:
            raise ValueError(f"Неизвестный тип регулятора: {controller_type}")

        # Аналитические полюса сохраняют кратность. Проверяем их по полиному.
        np.testing.assert_allclose(
            np.poly(self.poles), self.den, rtol=1e-13, atol=1e-13
        )
        assert np.all(self.poles.real < 0)

    def _synthesize_pd(self):
        """Вычисляет kp, kd и полюса по коэффициентам объекта и omega.

        Использует полином (s + η)[(s + η)² + ω²], где η = a1 / 3.
        Записывает ki = 0, num, den и poles в объект и возвращает None.
        """
        eta = self.a1 / 3
        omega = self.omega
        self.kp = (eta**3 + omega**2 * eta - self.a3) / self.b0
        self.kd = (3 * eta**2 + omega**2 - self.a2) / self.b0
        self.ki = 0.0

        # Wзамк = N / (Dобъекта + N), где N = b0(kд·s + kп).
        self.num = self.b0 * np.array([self.kd, self.kp])
        self.den = np.array([
            1, self.a1, self.a2 + self.num[0], self.a3 + self.num[1],
        ])
        self.poles = np.array([-eta, -eta + 1j * omega, -eta - 1j * omega])

    def _synthesize_pid(self):
        """Вычисляет kp, kd, ki и полюса по коэффициентам объекта, omega и beta.

        Использует полином [(s + η)² + ω²][(s + η)² + β²], где η = a1 / 4.
        Записывает num, den и poles в объект и возвращает None.
        """
        eta = self.a1 / 4
        omega, beta = self.omega, self.beta
        frequency_sum = omega**2 + beta**2
        self.kp = (4 * eta**3 + 2 * eta * frequency_sum - self.a3) / self.b0
        self.kd = (6 * eta**2 + frequency_sum - self.a2) / self.b0
        self.ki = (eta**2 + omega**2) * (eta**2 + beta**2) / self.b0

        # Интегратор регулятора добавляет множитель s к Dобъекта.
        self.num = self.b0 * np.array([self.kd, self.kp, self.ki])
        self.den = np.array([
            1, self.a1, self.a2 + self.num[0], self.a3 + self.num[1], self.num[2],
        ])
        self.poles = np.array([
            -eta + 1j * omega, -eta - 1j * omega,
            -eta + 1j * beta, -eta - 1j * beta,
        ])

    @property
    def stability_degree(self):
        """Возвращает −max(Re(s)) по полюсам системы в с⁻¹."""
        return -max(self.poles.real)

    @property
    def oscillation_degree(self):
        """Возвращает безразмерное max(|Im(s) / Re(s)|) по полюсам системы."""
        return max(abs(self.poles.imag / self.poles.real))


class StepResponse:
    """Хранит отклик САУ на единичный скачок при нулевых начальных условиях.

    Массивы time и output содержат время в секундах и значения отклика.
    steady_output — установившийся выход, settling_time — время
    регулирования в секундах, overshoot — перерегулирование в процентах.
    """

    def __init__(self, system):
        """Принимает ControlSystem, задаёт сетку времени и рассчитывает отклик и показатели."""
        self.system = system
        self.steady_output = system.num[-1] / system.den[-1]

        eta = system.stability_degree
        duration = 36 / eta
        rate = max(eta, max(abs(system.poles.imag)))
        sample_count = int(np.ceil(32 * rate * duration)) + 1
        self.time = np.linspace(0, duration, sample_count)
        self.output = self._simulate(duration)
        self.settling_time = self._calculate_settling_time()
        self.overshoot = self._calculate_overshoot()

    def _simulate(self, duration):
        """Принимает длительность моделирования в секундах.

        Строит модель в пространстве состояний, сохраняет решение и возвращает
        массив отклика на self.time. При сбое решения выдаёт RuntimeError.
        """
        a, b, c, _ = signal.tf2ss(self.system.num, self.system.den)
        equilibrium = np.linalg.solve(a, -b).ravel()

        # Интегрируем отклонение состояния от равновесия при единичном скачке.
        self.solution = solve_ivp(
            lambda _, x: a @ x,
            (0, duration),
            -equilibrium,
            method="DOP853",
            rtol=1e-10,
            atol=1e-13,
            dense_output=True,
        )
        if not self.solution.success:
            raise RuntimeError(self.solution.message)

        self.output_matrix = c[0]
        return self._output_at(self.time)

    def _output_at(self, time):
        """Принимает момент или массив моментов в интервале моделирования.

        Возвращает значение отклика или массив значений из сохранённого решения.
        """
        return self.steady_output + self.output_matrix @ self.solution.sol(time)

    def _calculate_settling_time(self):
        """Возвращает время последнего входа отклика в полосу ±5% от |h∞| в секундах."""
        tolerance = SETTLING_BAND * abs(self.steady_output)
        error = np.abs(self.output - self.steady_output)
        last_outside = np.flatnonzero(error > tolerance)[-1]
        assert last_outside < len(self.time) - 1
        assert error[-1] < 1e-7 * max(1, abs(self.steady_output))

        return brentq(
            lambda time: abs(self._output_at(time) - self.steady_output) - tolerance,
            self.time[last_outside],
            self.time[last_outside + 1],
        )

    def _calculate_overshoot(self):
        """Возвращает перерегулирование в процентах по уточнённому пику отклика.

        При отрицательном установившемся выходе используется минимум отклика.
        """
        normalized = self.output / self.steady_output
        peak = np.argmax(normalized)
        left = self.time[max(0, peak - 1)]
        right = self.time[min(len(self.time) - 1, peak + 1)]
        result = minimize_scalar(
            lambda time: -self._output_at(time) / self.steady_output,
            bounds=(left, right),
            method="bounded",
        )
        return 100 * max(0, normalized[peak] - 1, -result.fun - 1)


class SystemPlotter:
    """Строит общий график переходных характеристик для каждой серии."""

    def __init__(self, variant, output_dir):
        """Принимает номер варианта для заголовков и путь Path к папке графиков."""
        self.variant = variant
        self.output_dir = output_dir

    def save_series(self, series, title, experiments):
        """Принимает идентификатор серии, заголовок и список пар (k, StepResponse).

        Сохраняет график в <series>.png, оставляет его для показа в main()
        и возвращает None.
        """
        fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
        colors = ("#516278", "#2466a8", "#147a6d", "#8055a2",
                  "#b36a19", "#b04467", "#243650")
        ax.set_facecolor("#f8fafc")
        ax.set_axisbelow(True)

        if series != "pd":
            ax.axhspan(0.95, 1.05, color="#94a8bf", alpha=0.24,
                       label="Полоса ±5%")
        ax.axhline(1, color="#526176", linewidth=1, linestyle=":", label="Задание")

        for k, response in experiments:
            index = K_VALUES.index(k)
            ax.plot(
                response.time,
                response.output,
                linewidth=1.5,
                solid_capstyle="round",
                color=colors[index],
                label=f"ω = {response.system.omega:g}".replace(".", ","),
            )

        latest_settling_time = max(response.settling_time for _, response in experiments)
        ax.set(
            xlabel="Время t, с",
            ylabel="Выход h(t)",
            xlim=(0, max(1.4, 1.1 * latest_settling_time)),
        )
        ax.set_title(title, loc="left", fontsize=16, fontweight="bold",
                     color="#172638", pad=12)
        ax.set_title(f"Вариант {self.variant}", loc="right", fontsize=10,
                     color="#526176", pad=12)
        ax.grid(color="#d7e1eb", linewidth=0.7)
        ax.tick_params(colors="#445368", labelsize=10)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#a9b7c7")
        handles, labels = ax.get_legend_handles_labels()
        references = len(handles) - len(experiments)
        ax.legend(handles[references:] + handles[:references],
                  labels[references:] + labels[:references],
                  title="Частота ω, рад/с",
                  loc="upper center", bbox_to_anchor=(0.5, -0.17),
                  ncols=5, frameon=False, fontsize=9)
        fig.savefig(self.output_dir / f"{series}.png", dpi=300, facecolor="white")


def run_series(series, k_values, plant):
    """Принимает идентификатор серии, значения k и коэффициенты объекта.

    Для pd, pid_beta0 или pid_equal выбирает регулятор и beta, а omega
    вычисляет как 2.5 * k. Возвращает список пар (k, StepResponse).
    """
    experiments = []
    controller_type = "PD" if series == "pd" else "PID"
    for k in k_values:
        omega = 2.5 * k
        beta = omega if series == "pid_equal" else 0.0
        system = ControlSystem(plant, controller_type, omega, beta)
        experiments.append((k, StepResponse(system)))
    return experiments


def experiment_row(variant, series, k, response):
    """Принимает номер варианта, серию, значение k и объект StepResponse.

    Возвращает словарь для строки CSV с коэффициентами регулятора,
    полюсами, степенями устойчивости и колебательности и показателями отклика.
    """
    system = response.system
    return {
        "variant": variant,
        "series": series,
        "k": k,
        "omega": system.omega,
        "beta": system.beta,
        "kp": system.kp,
        "kd": system.kd,
        "ki": system.ki,
        "eta": system.stability_degree,
        "mu": system.oscillation_degree,
        "poles": "; ".join(f"{pole.real:.9g}{pole.imag:+.9g}j" for pole in system.poles),
        "y_inf": response.steady_output,
        "e_inf": 1 - response.steady_output,
        "ts_s": response.settling_time,
        "overshoot_pct": response.overshoot,
    }


def save_table(path, rows):
    """Принимает путь Path и непустой список словарей с одинаковыми ключами.

    Записывает CSV в UTF-8 с заголовком из ключей первой строки. Возвращает None.
    """
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Запускает все серии для VARIANT, сохраняет CSV и показывает графики.

    Возвращает None. Если VARIANT отсутствует в таблице, выдаёт ValueError.
    """
    if VARIANT not in VARIANTS:
        raise ValueError("Выберите вариант от 1 до 10")

    plant = VARIANTS[VARIANT]
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

    plotter = SystemPlotter(VARIANT, output_dir)
    rows = []
    for series, title, k_values in SERIES:
        experiments = run_series(series, k_values, plant)
        plotter.save_series(series, title, experiments)
        rows.extend(experiment_row(VARIANT, series, k, response)
                    for k, response in experiments)

    save_table(output_dir / "experiments.csv", rows)
    print(f"Вариант {VARIANT}. Выполнено {len(rows)} опытов. Результаты: {output_dir}")
    plt.show()


if __name__ == "__main__":
    main()

