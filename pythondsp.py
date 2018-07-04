import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

signal_param = namedtuple('signal_param', ['A', 'f'])


def generate_sin(A, freq, t):
    return A*np.sin(2*np.pi*freq*t)


def generate_frequencies(signal_params, generator):
    fs = 1000.0
    T = 3
    t = np.arange(0, T*fs) / fs
    ret = np.zeros(len(t))

    for a, f in signal_params:
        y = generator(a, f, t)
        ret += y

    return t, ret


def main():
    signal_params = [signal_param(1, 1), signal_param(1, 5)]
    x, y = generate_frequencies(signal_params, generate_sin)
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    main()
