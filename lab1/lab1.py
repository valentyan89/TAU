import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def _split_param(value):
    """Первое значение — основной вариант, остальные — для сравнения."""
    if isinstance(value, (list, tuple, np.ndarray)):
        values = [float(v) for v in value]
        if not values:
            raise ValueError("Список параметров не должен быть пустым")
        return values[0], values[1:]
    return float(value), []


'''
Класс для представления линейного звена.
'''
class Link:
    def __init__(self, name, link_type, k=1.0, T=0.9, zeta=0.4):
        self.name = name
        self.link_type = link_type

        self.k, extra_k = _split_param(k)
        self.T, extra_T = _split_param(T)
        self.zeta, extra_zeta = _split_param(zeta)

        # Пары (имя параметра, значение) для пунктирных кривых.
        # Каждый параметр меняется отдельно при остальных из варианта.
        self.comparisons = (
            [("k", v) for v in extra_k]
            + [("T", v) for v in extra_T]
            + [("zeta", v) for v in extra_zeta]
        )

        self.num, self.den = self._get_transfer_function()
        self.system = self.get_system()

    def with_params(self, **overrides):
        """Копия звена с теми же типом/именем и другими k, T, ζ."""
        return Link(
            name=self.name,
            link_type=self.link_type,
            k=overrides.get("k", self.k),
            T=overrides.get("T", self.T),
            zeta=overrides.get("zeta", self.zeta),
        )

    def comparison_links(self):
        """Звенья для сравнения: одно значение меняется, остальные как в варианте."""
        links = []
        for param, value in self.comparisons:
            links.append((param, value, self.with_params(**{param: value})))
        return links

    def _get_transfer_function(self):
        """Возвращает коэффициенты числителя и знаменателя передаточной функции в виде списков."""

        if self.link_type == "aperiodic":
            # W(s) = k / (Ts + 1)
            num = [self.k]
            den = [self.T, 1]

        elif self.link_type == "forcing":
            # W(s) = k(Ts + 1)
            num = [self.k * self.T, self.k]
            den = [1]

        elif self.link_type == "oscillatory":
            # W(s) = k / (T²s² + 2ζTs + 1)
            num = [self.k]
            den = [
                self.T**2,
                2 * self.zeta * self.T,
                1
            ]

        elif self.link_type == "combined":
            # W(s) = k(2Ts + 1) /
            #        [(T²s² + 2ζTs + 1)(Ts + 1)]

            num = [2 * self.k * self.T, self.k]

            den = np.polymul(
                [self.T**2, 2 * self.zeta * self.T, 1],
                [self.T, 1]
            )

        else:
            raise ValueError(
                f"Неизвестный тип звена: {self.link_type}"
            )

        return num, den

    def get_system(self):
        """Возвращает передаточную функцию в виде объекта scipy.signal.TransferFunction."""

        return signal.TransferFunction(self.num, self.den)

    def param_label(self, changed=None):
        if changed == "k":
            return f"k = {self.k:g}"
        if changed == "T":
            return f"T = {self.T:g}"
        if changed == "zeta":
            return f"ζ = {self.zeta:g}"

        if self.link_type in ("oscillatory", "combined"):
            return f"вариант: k={self.k:g}, T={self.T:g}, ζ={self.zeta:g}"
        return f"вариант: k={self.k:g}, T={self.T:g}"

    def __str__(self):
        extra = ""
        if self.comparisons:
            extra = ", сравнения=" + ", ".join(
                f"{p}={v:g}" for p, v in self.comparisons
            )
        return (
            f"{self.name}: "
            f"type={self.link_type}, "
            f"k={self.k}, T={self.T}, zeta={self.zeta}"
            f"{extra}"
        )



'''
Класс для построения графиков характеристик линейных звеньев.
'''
class LinkPlotter:
    def __init__(self, w_min=0.01, w_max=100, points=2000):
        self.w = np.logspace(
            np.log10(w_min),
            np.log10(w_max),
            points
        )

    def _curves(self, link):
        """Основная кривая (сплошная) и сравнения (пунктир)."""
        yield link, link.param_label(), {"linestyle": "-", "linewidth": 2.2}

        for param, value, alt in link.comparison_links():
            yield alt, alt.param_label(changed=param), {
                "linestyle": "--",
                "linewidth": 1.5,
            }

    def plot_step(self, link, ax):
        """Переходная характеристика."""

        all_links = [link] + [alt for _, _, alt in link.comparison_links()]
        t_end = 12 * max(item.T for item in all_links)
        t = np.linspace(0, t_end, 4000)

        shown_k = set()
        for item, label, style in self._curves(link):
            _, h = signal.step(item.get_system(), T=t)
            ax.plot(t, h, label=label, **style)

            if item.k not in shown_k:
                shown_k.add(item.k)
                ax.axhline(
                    item.k,
                    color="green",
                    linestyle=":" if item is not link else "--",
                    linewidth=1,
                    alpha=0.8,
                    label=f"k = {item.k:g}",
                )

        ax.set_title("Переходная характеристика")
        ax.set_xlabel("t, с")
        ax.set_ylabel("h(t)")
        ax.grid(True, linestyle=":")

        ax.legend()

    def plot_nyquist(self, link, ax):
        """Амплитудно-фазовая характеристика."""

        for i, (item, label, style) in enumerate(self._curves(link)):
            _, H = signal.freqresp(
                item.get_system(),
                w=self.w
            )

            line, = ax.plot(H.real, H.imag, label=label, **style)

            if i == 0:
                ax.plot(
                    H.real, -H.imag, ":",
                    color=line.get_color(),
                    alpha=0.45,
                )
                ax.plot(H.real[0], H.imag[0], "ro")
                ax.plot(H.real[-1], H.imag[-1], "go")

        ax.axhline(0, color="black", linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.6)

        ax.set_title("АФХ годограф")
        ax.set_xlabel("Re W(jω)")
        ax.set_ylabel("Im W(jω)")
        ax.grid(True, linestyle=":")
        ax.axis("equal")
        ax.legend()

    def plot_bode(self, link, ax_mag, ax_phase):
        """ЛАЧХ и ЛФЧХ."""

        for item, label, style in self._curves(link):
            w, mag, phase = signal.bode(
                item.get_system(),
                w=self.w
            )
            ax_mag.semilogx(w, mag, label=label, **style)
            ax_phase.semilogx(w, phase, label=label, **style)

        ax_mag.set_title("ЛАЧХ")
        ax_mag.set_xlabel("ω, рад/с")
        ax_mag.set_ylabel("L(ω), дБ")
        ax_mag.grid(True, which="both", linestyle=":")
        ax_mag.legend()

        ax_phase.set_title("ЛФЧХ")
        ax_phase.set_xlabel("ω, рад/с")
        ax_phase.set_ylabel("φ(ω), г рад")
        ax_phase.grid(True, which="both", linestyle=":")
        ax_phase.legend()

    def plot_link(self, link):
        """Строит все применимые характеристики одного звена."""

        extra = ""
        if link.comparisons:
            extra = " | пунктир — сравнение"

        if link.link_type == "forcing":
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            self.plot_nyquist(link, axes[0])

            self.plot_bode(link, axes[1], axes[2])

            fig.suptitle(
                f"{link.name} | Форсирующее звено\n"
                f"k={link.k:g}, T={link.T:g}{extra}"
            )

        else:
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))

            self.plot_step(link, axes[0, 0])
            self.plot_nyquist(link, axes[0, 1])
            self.plot_bode(link, axes[1, 0], axes[1, 1])

            fig.suptitle(
                f"{link.name}\n"
                f"k={link.k:g}, T={link.T:g}, ζ={link.zeta:g}{extra}"
            )

        fig.tight_layout()
        return fig

    def plot_links(self, links):
        """Строит характеристики для списка звеньев."""

        for link in links:
            self.plot_link(link)

        plt.show()


'''
Построение графиков характеристик линейных звеньев.
'''
if __name__ == "__main__":
    # Вариант 10. Первое число в списке — ваш вариант (сплошная линия),
    # остальные — для сравнения (пунктир). Можно передать и одно число.
    link1 = Link(
        name="Апериодическое звено",
        link_type="aperiodic",
        k=1,
        T=[0.9, 0.45, 1.8],
    )

    link2 = Link(
        name="Форсирующее звено",
        link_type="forcing",
        k=1,
        T=[0.9, 0.45, 1.8],
    )

    link3 = Link(
        name="Колебательное звено",
        link_type="oscillatory",
        k=1,
        T=0.9,
        zeta=[0.4, 0.15, 0.8],
    )

    link4 = Link(
        name="Комбинированное звено",
        link_type="combined",
        k=1,
        T=0.9,
        zeta=[0.4, 0.15, 0.8],
    )

    links = [link1, link2, link3, link4]
    plotter = LinkPlotter()

    plotter.plot_links(links)
