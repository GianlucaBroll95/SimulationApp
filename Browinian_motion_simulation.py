import tkinter
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, ttk, StringVar
from tkinter.messagebox import showerror
from scipy.stats import norm


def arithmetic_brownian_motion(s0, mu, sigma, t=1, n=1000):
    """
        Arithmetic Brownian Motion implementation.
    Args:
        s0 (int | float): Initial value of the stochastic process
        mu (float): Annualized drift rate of the stochastic differential equation
        sigma (float): Annualized variance rate of the stochastic differential equation
        t (int | float): Time span for the process (default = 1)
        n (int): Numer of timestep for the process simulation per unit t (total steps = t*n, dt = t/(t*n) = 1/n)
    Returns:
        [t*n, 1] numpy.ndarray gathering the simulated process.
    """
    dt = 1 / n
    ds = mu * dt + sigma * np.sqrt(dt) * np.random.randn(int(n * t))
    return np.cumsum(np.append([s0], ds))


def geometric_brownian_motion(s0, mu, sigma, t, n):
    """
        Geometric Brownian Motion implementation.
    Args:
        s0 (int | float): Initial value of the stochastic process
        mu (float): Annualized drift rate of the stochastic differential equation
        sigma (float): Annualized variance rate of the stochastic differential equation
        t (int | float): Time span for the process (default = 1)
        n (int): Numer of timestep for the process simulation per unit t (dt = t/n, total steps = n * t)
    Returns:
        [t*n, 1] numpy.ndarray gathering the simulated process.
    """
    mu = mu - 1 / 2 * sigma ** 2
    x = arithmetic_brownian_motion(0, mu, sigma, t, n)
    return s0 * np.exp(x)


def calculate_true_moments(s0, mu, sigma, t, process):
    """
        Calculates the exact first and second moment of the process.
    Args:
        s0 (int | float): initial process value
        mu (float): drift rate for the process
        sigma (float): drift rate for the process
        t (int | float): Time span for the process
        process (str): whether the process is ABM or GBM
    Returns:
        tuple of formatted mean and variance for the final value
    """
    if process == "ABM":
        mean = s0 + mu * t
        variance = sigma ** 2 * t
    else:
        mean = s0 * np.exp(mu * t)
        variance = s0 ** 2 * np.exp(2 * mu * t) * (np.exp(sigma ** 2 * t) - 1)
    return f"{mean:.3f}", f"{variance:.3%}"


def calculate_sim_moments(final_value):
    """
        Calculates the mean and variance of the simulated values
    Args:
        final_value (list): array of each path last value
    Returns:
        tuple of formatted mean and variance for the final value
    """
    final_value = np.array(final_value)
    return f"{final_value.mean():.3f}", f"{final_value.var():.3%}"


def calculate_simulated_probability(final_values, s0, lb, ub):
    """
        Calculates the probability that the simulated path last value falls between a lower bound and an upper bound
    Args:
        final_values (list): array of each path last value
        s0 (int | float): initial process value
        lb (int | float | None): upper bound
        ub (int | float): lower bound
    Returns:
        estimated probability
    """
    final_values = np.array(final_values)
    ub = s0 * (1 + ub)
    if lb is None:
        p = len(final_values[final_values <= ub]) / len(final_values)
    else:
        lb = s0 * (1 + lb)
        p = (len(final_values[final_values <= ub]) - len(final_values[final_values <= lb])) / len(final_values)
    return f"{p:.2%}"


def calculate_true_probability(mu, sigma, t, process, lb, ub):
    """
        Calculates the true probability for the last value of the stochastic process to fall between a lower bound and
        an upper bound.
    Args:
        mu (float): drift rate for the process
        sigma (float): drift rate for the process
        t (int | float): Time span for the process
        process (str): whether the process is ABM or GBM
        lb (int | float | None): upper bound
        ub (int | float): lower bound
    Returns:
        true probability
    """
    if process == "ABM":
        if lb is None:
            p = norm.cdf(ub, mu * t, sigma * np.sqrt(t))
        else:
            p = norm.cdf(ub, mu * t, sigma * np.sqrt(t)) - norm.cdf(lb, mu * t, sigma * np.sqrt(t))
    else:
        mu = (mu - 1 / 2 * sigma ** 2)
        if lb is None:
            p = norm.cdf(ub, mu * t, sigma * np.sqrt(t))
        else:
            p = norm.cdf(ub, mu * t, sigma * np.sqrt(t)) - norm.cdf(lb, mu * t, sigma * np.sqrt(t))
    return f"{p:.2%}"


class SimulationApp(ttk.Frame):

    def __init__(self, master: tkinter.Tk, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        # root container
        master.title("Brownian Motion Simulation")
        master.minsize(660, 660)
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        # main frame
        self.grid(column=0, row=0, sticky="n s w e")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        val_function = (self.register(self._validate_numeric_entry), "%P")
        # input frame
        self.inputs = ttk.LabelFrame(self, borderwidth=2, relief="ridge", text="Inputs:")
        self.inputs.grid(column=0, row=0, sticky="n s w e")

        self.inputs.columnconfigure("0 1 2 3 4", weight=1)
        ttk.Label(self.inputs, text="Initial Value: ").grid(column=0, row=0, sticky="e")
        self.initial_value = StringVar(value="1")
        initial_value_entry = ttk.Entry(self.inputs, textvariable=self.initial_value, width=4,
                                        validate="key", validatecommand=val_function)
        initial_value_entry.grid(column=1, row=0, sticky="w e")
        ttk.Label(self.inputs, text="Drift Rate:").grid(column=0, row=1, sticky="e")
        self.drift = StringVar(value="0.06")
        drift_entry = ttk.Entry(self.inputs, textvariable=self.drift, width=4,
                                validate="all", validatecommand=val_function)
        drift_entry.grid(column=1, row=1, sticky="w e")
        ttk.Label(self.inputs, text="Variance Rate:").grid(column=0, row=2, sticky="e")
        self.variance = StringVar(value="0.25")
        variance_entry = ttk.Entry(self.inputs, textvariable=self.variance, width=4,
                                   validate="all", validatecommand=val_function)
        variance_entry.grid(column=1, row=2, sticky="w e")
        ttk.Label(self.inputs, text="T:").grid(column=2, row=0, sticky="e")
        self.T = StringVar(value="1")
        t_entry = ttk.Entry(self.inputs, textvariable=self.T, width=4,
                            validate="all", validatecommand=val_function)
        t_entry.grid(column=3, row=0, sticky="w e")
        ttk.Label(self.inputs, text="Steps:").grid(column=2, row=1, sticky="e")
        self.steps = StringVar(value="1000")
        steps_entry = ttk.Entry(self.inputs, textvariable=self.steps, width=4,
                                validate="all", validatecommand=val_function)
        steps_entry.grid(column=3, row=1, sticky="w e")
        ttk.Label(self.inputs, text="Number of simulations:").grid(column=2, row=2, sticky="e")
        self.n_sim = StringVar(value="10")
        n_sim_entry = ttk.Entry(self.inputs, textvariable=self.n_sim, width=4,
                                validate="all", validatecommand=val_function)
        n_sim_entry.grid(column=3, row=2, sticky="w e")
        self.process_type = StringVar(value="ABM")
        ttk.Radiobutton(self.inputs, text="Arithmetic Brownian Motion", variable=self.process_type,
                        value="ABM").grid(column=4, row=0)
        ttk.Radiobutton(self.inputs, text="Geometric Brownian Motion", variable=self.process_type,
                        value="GBM").grid(column=4, row=1)
        # probability
        ttk.Separator(self.inputs, orient="horizontal").grid(row=3, columnspan=5, sticky="we", pady=5, padx=5)
        ttk.Label(self.inputs, text="≤   ln S(T)/S(0)   ≤").grid(column=2, row=4)
        ttk.Label(self.inputs, text="Prob:").grid(column=0, row=4, sticky="e")
        self.low_bound = StringVar()
        low_bound_entry = ttk.Entry(self.inputs, textvariable=self.low_bound, width=4)
        low_bound_entry.grid(column=1, row=4, sticky="w e")
        self.upper_bound = StringVar(value="0.05")
        upper_bound_entry = ttk.Entry(self.inputs, textvariable=self.upper_bound, width=4)
        upper_bound_entry.grid(column=3, row=4, sticky="w e")
        ttk.Button(self.inputs, text="Get Simulation", command=self._simulate).grid(column=4, row=2)

        # chart frame
        self.chart = ttk.LabelFrame(self, borderwidth=2, relief="ridge", width=200, height=200, text="Simulated Path:")
        self.chart.grid(column=0, row=1, columnspan=3, rowspan=3, sticky="e w n s")
        self.chart.rowconfigure(0, weight=1)
        self.chart.columnconfigure(1, weight=1)

        for child in self.winfo_children():
            child.grid_configure(padx=4, pady=4)

    def _validate_numeric_entry(self, entry):
        try:
            float(entry)
            test_passed = True
        except ValueError:
            test_passed = False
            self.bell()
        return test_passed

    def _simulate(self):
        s0 = float(self.initial_value.get())
        mu = float(self.drift.get())
        sigma = float(self.variance.get())
        time = float(self.T.get())
        n = int(self.steps.get())
        process_type = self.process_type.get()
        n_sim = int(self.n_sim.get())
        lb = None if self.low_bound.get() == "" else float(self.low_bound.get())
        ub = float(self.upper_bound.get())
        if lb is not None and lb > ub:
            raise ValueError("Upper bound is greater than upper bound.")
        final_value = list()
        try:
            figure = Figure(dpi=100)
            figure_canvas = FigureCanvasTkAgg(figure, self.chart)
            axis = figure.add_subplot()
            axis.set_title("Arithmetic Brownian Motion" if process_type == "ABM" else "Geometric Brownian Motion",
                           fontdict={"weight": "bold", "size": 12})
            axis.set_xlabel("t")
            axis.set_ylabel("X(t)")
            for _ in range(n_sim):
                if process_type == "ABM":
                    xt = arithmetic_brownian_motion(s0, mu, sigma, time, n)
                else:
                    xt = geometric_brownian_motion(s0, mu, sigma, time, n)
                final_value.append(xt[-1])
                # axis.plot(np.linspace(0, time, int(time * n + 1)), xt, color="k", linewidth=0.5, alpha=0.8)
                axis.plot(np.linspace(0, time, int(time * n + 1)), xt, linewidth=0.5, alpha=0.8)
            figure_canvas.get_tk_widget().grid(column=0, row=0, columnspan=3, rowspan=3, sticky="n e w s")
            figure_canvas.get_tk_widget().columnconfigure("0 1 2", weight=1)

            if n_sim > 1:
                sim_prob = calculate_simulated_probability(final_value, s0, lb, ub)
                true_prob = calculate_true_probability(mu, sigma, time, process_type, lb, ub)
                if hasattr(self, "prob"):
                    self.prob.destroy()

                self.prob = ttk.Label(self.inputs, text=f"        = {sim_prob} (True prob: {true_prob})")
                self.prob.grid(column=4, row=4, sticky="w")
                self.result = ttk.LabelFrame(self, borderwidth=2, width=200, height=50, relief="ridge",
                                             text="Final Distribution Moments:")
                self.result.grid(column=0, row=4, columnspan=3, rowspan=3, sticky="e w n s")
                self.result.rowconfigure(0, weight=1)
                self.result.columnconfigure(1, weight=1)

                sim_mean, sim_variance = calculate_sim_moments(final_value)
                true_mean, true_variance = calculate_true_moments(s0, mu, sigma, time, process_type)

                ttk.Label(self.result, text="Simulated Mean:", width=15).grid(column=0, row=0, sticky="w")
                ttk.Label(self.result, text="True Mean:", width=15).grid(column=0, row=1, sticky="w")

                self.sim_mean = ttk.Label(self.result, text=sim_mean)
                self.sim_mean.grid(column=1, row=0, sticky="w")
                self.true_mean = ttk.Label(self.result, text=true_mean)
                self.true_mean.grid(column=1, row=1, sticky="w")

                ttk.Label(self.result, text="Simulated Variance:", width=15).grid(column=2, row=0, sticky="w")
                ttk.Label(self.result, text="True Variance:", width=15).grid(column=2, row=1, sticky="w")

                self.sim_var = ttk.Label(self.result, text=sim_variance)
                self.sim_var.grid(column=3, row=0, sticky="w")
                self.true_var = ttk.Label(self.result, text=true_variance)
                self.true_var.grid(column=3, row=1, sticky="w")

                for child in self.winfo_children():
                    child.grid_configure(padx=4, pady=4)
        except ValueError as err:
            showerror(title="Error", message=err)


root = Tk()
SimulationApp(root)
root.mainloop()
