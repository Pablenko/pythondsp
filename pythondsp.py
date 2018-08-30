import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

signal_param = namedtuple('signal_param', ['A', 'f'])


def generate_sin(A, freq, t):
    return A*np.sin(2*np.pi*freq*t)


def generate_frequencies(sampling_freq, signal_params, generator):
    t = np.arange(0, sampling_freq) / sampling_freq
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
            im_signal[k] -= input[n] * np.sin(2*np.pi*k*n / signal_length)

    return re_signal, im_signal


def idft(re, im):
    dft_length = len(re)
    output_signal_length = dft_length * 2

    re_amplitude = np.zeros(len(re))
    im_amplitude = np.zeros(len(im))
    signal = np.zeros(output_signal_length)

    for k, (r, i) in enumerate(zip(re, im)):
        re_amplitude[k] = re[k] / dft_length
        im_amplitude[k] = -im[k] / dft_length

    re_amplitude[0] /= 2
    re_amplitude[dft_length-1] /= 2

    for n in xrange(output_signal_length):
        for k in xrange(dft_length):
            signal[n] += re_amplitude[k] * np.cos(2*np.pi*k*n/output_signal_length)
            signal[n] += im_amplitude[k] * np.sin(2*np.pi*k*n/output_signal_length)

    return signal



def polar_coords(re, im):
    def magnitude():
        for k in xrange(len(re)):
            yield np.sqrt(re[k]*re[k] + im[k]*im[k])

    def phase():
        for k in xrange(len(im)):
            yield np.arctan(im[k] / re[k])

    return [m for m in magnitude()], [p for p in phase()]


def convolution(input, conv_kernel):
    input_length = len(input)
    kernel_length = len(conv_kernel)
    output_length = input_length + kernel_length - 1

    output = np.zeros(output_length)

    for i in xrange(input_length):
        for j in xrange(kernel_length):
            output[i+j] += input[i] * conv_kernel[j]

    return output


def main():
    signal_params = [signal_param(2, 2)]
    x, y = generate_frequencies(80.0, signal_params, generate_sin)

    conv_kernel = np.zeros(9)
    conv_kernel[2] = 1
    conv_kernel[3] = -1

    output = convolution(y, conv_kernel)

    plt.subplot(211)
    plt.plot(x, y)
    plt.subplot(212)
    plt.plot(np.arange(len(output)), output)
    plt.show()


if __name__ == "__main__":
    main()
