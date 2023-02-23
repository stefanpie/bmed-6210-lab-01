import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing


def gen_fake_data(T1, M0):
    t = np.array([80, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000])* 1e-3

    s_t = sp.symbols("t", real=True, positive=True)
    s_T1 = sp.symbols("T1", real=True, positive=True)
    s_M0 = sp.symbols("M0", real=True, positive=True)

    s_MZ = s_M0 * (1 - 2 * sp.exp(-1*s_t / s_T1))

    s_signal = sp.Abs(s_MZ)
    signal_func = sp.lambdify((s_t, s_T1, s_M0), s_signal, "numpy")
    signal = signal_func(t, T1, M0)

    # jac = sp.Matrix([s_signal]).jacobian([s_T1, s_M0])
    # jac_func = sp.lambdify((s_t, s_T1, s_M0), jac, "numpy")

    noise = np.random.normal(0, 0.02*M0, len(t))
    signal_noise = signal + noise

    return t, signal, signal_noise


def IR_func(t, T1, M0):
    return np.abs(M0 * (1 - 2 * np.exp(-t / T1)))

def IR_func_jac(t, T1, M0):
    return np.array([
        -2*M0*t*np.exp(-t/T1)*np.sign(1 - 2*np.exp(-t/T1))/T1**2,
        np.abs(1 - 2*np.exp(-t/T1))
    ]).T

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def rmse_loss(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def curve_fit_FLAIR(t, signal):

    def loss_func(params):
        return rmse_loss(signal, IR_func(t, *params))
    
    res = dual_annealing(loss_func, bounds=[(1e-9, 10), (1e-9, 10000)])

    return res.x
 

if __name__ == "__main__":
    t, signal, signal_noise = gen_fake_data(2.300, 1000)

    res_x = curve_fit_FLAIR(t, signal_noise)

    print("T1 = ", res_x[0])
    print("M0 = ", res_x[1])

    plt.plot(t, signal, label="signal")
    plt.plot(t, signal_noise, label="signal_noise")
    plt.plot(t, IR_func(t, *res_x), label="fit")
    plt.legend()
    plt.show()
