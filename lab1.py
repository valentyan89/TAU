import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


'''
Класс для представления линейного звена.
'''
class Link:
    def __init__(self, name, link_type, k=1.0, T=0.9, zeta=0.4):
        self.name = name
        self.link_type = link_type
        self.k = k
        self.T = T
        self.zeta = zeta

        self.num, self.den = self._get_transfer_function()
        self.system = self.get_system()

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

    def __str__(self):
        return (
            f"{self.name}: "
            f"type={self.link_type}, "
            f"k={self.k}, T={self.T}, zeta={self.zeta}"
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

    def plot_step(self, link, ax):
        """Переходная характеристика."""

        t_end = 12 * link.T
        t = np.linspace(0, t_end, 4000)

        t, h = signal.step(link.get_system(), T=t)

        ax.plot(t, h, label=link.name)
        ax.axhline(link.k, color="green", linestyle="--",
                   linewidth=1, label=f"k = {link.k:g}")

        ax.set_title("Переходная характеристика")
        ax.set_xlabel("t, с")
        ax.set_ylabel("h(t)")
        ax.grid(True, linestyle=":")

        ax.legend()

    def plot_nyquist(self, link, ax):
        """Амплитудно-фазовая характеристика."""

        w, H = signal.freqresp(
            link.get_system(),
            w=self.w
        )

        ax.plot(H.real, H.imag, label=link.name)
        ax.plot(H.real, -H.imag, "--", alpha=0.5)

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

        w, mag, phase = signal.bode(
            link.get_system(),
            w=self.w
        )

        ax_mag.semilogx(w, mag, label=link.name)
        ax_phase.semilogx(w, phase, label=link.name)

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

        if link.link_type == "forcing":
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            self.plot_nyquist(link, axes[0])

            self.plot_bode(link, axes[1], axes[2])

            fig.suptitle(
                f"{link.name} | Форсирующее звено\n"
                f"k={link.k:g}, T={link.T:g}"
            )

        else:
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))

            self.plot_step(link, axes[0, 0])
            self.plot_nyquist(link, axes[0, 1])
            self.plot_bode(link, axes[1, 0], axes[1, 1])

            fig.suptitle(
                f"{link.name}\n"
                f"k={link.k:g}, T={link.T:g}, ζ={link.zeta:g}"
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
    # Вариант 10
    link1 = Link(
        name="Апериодическое звено",
        link_type="aperiodic",
        k=1,
        T=0.9
    )

    link2 = Link(
        name="Форсирующее звено",
        link_type="forcing",
        k=1,
        T=0.9
    )

    link3 = Link(
        name="Колебательное звено",
        link_type="oscillatory",
        k=1,
        T=0.9,
        zeta=0.4
    )

    link4 = Link(
        name="Комбинированное звено",
        link_type="combined",
        k=1,
        T=0.9,
        zeta=0.4
    )

    links = [link1, link2, link3, link4]
    plotter = LinkPlotter()

    plotter.plot_links(links)
