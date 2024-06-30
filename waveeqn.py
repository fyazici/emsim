import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    N = 100
    dt = 1e-3
    y = np.zeros(N)
    yx = np.zeros(N)
    yt = np.zeros(N)
    yxx = np.zeros(N)
    ytt = np.zeros(N)
    
    w = 1
    k = 0.5
    c = w * w / k / k

    x = np.linspace(0, 2 * np.pi * N, N)
    y = np.sin(x * k / 10)
    #yt = 0.01 *  np.cos(t * 2 * np.pi * 10)
    #ytt = 0.0001 *  np.sin(t * 2 * np.pi)

    plt.ion()
    fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_subplot()
    line1, = ax.plot(y)
    ax.set_ylim((-1.1, 1.1))

    for i in range(1000000):
        yx[1:-2] = y[2:-1] - y[0:-3]
        yx[0] = y[1] - y[-1]
        yx[-1] = y[0] - y[-2]
        yx /= 2

        yxx[1:-2] = yx[2:-1] - yx[0:-3]
        yxx[0] = yx[1] - yx[-1]
        yxx[-1] = yx[0] - yx[-2]
        yxx /= 2

        ytt = c * yxx
        yt += ytt * dt
        y += yt * dt
        #y *= 0.99999
        #y[49] = 0.1 * np.sin(i / 6000)

        if i % 100 == 0:
            line1.set_ydata(y)
            fig.canvas.draw()
            fig.canvas.flush_events()
            #if np.abs(y[0]) > 1e-2:
            #    input()