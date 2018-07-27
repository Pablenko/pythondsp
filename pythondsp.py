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


def dft(input):
    signal_length = len(input)
    dft_length = signal_length / 2

    re_signal = np.zeros(dft_length)
    im_signal = np.zeros(dft_length)

    for k in xrange(dft_length):
        for n in xrange(signal_length):
            re_signal[k] += input[n] * np.cos(2*np.pi*k*n / signal_length)
            im_signal[k] += -input[n] * np.sin(2*np.pi*k*n / signal_length)

    return re_signal, im_signal


def polar_coords(re, im):
    def magnitude():
        for k in xrange(len(re)):
            yield np.sqrt(re[k]*re[k] + im[k]*im[k])

    def phase():
        for k in xrange(len(im)):
            yield np.arctan(im[k] / re[k])

    return [m for m in magnitude()], [p for p in phase()]


def main():
    signal_params = [signal_param(1, 1), signal_param(1, 5)]
    x, y = generate_frequencies(signal_params, generate_sin)

    re, im = dft(y)
    x = np.arange(0, len(y)/2)
    mag, phase = polar_coords(re, im)

    plt.subplot(211)
    plt.plot(x, mag)
    plt.subplot(212)
    plt.plot(x, phase)
    plt.show()


if __name__ == "__main__":
    main()
