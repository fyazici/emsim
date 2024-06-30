import numpy as np
import matplotlib.pyplot as plt

# maxwell
# div E = rho / eps_0
# div B = 0
# curl E = -dB/dt
# curl B = mu_0 * J + 1/c^2 dE/dt


def curl(f: np.array, r: np.array, dr: np.floating, n: int):
    d = 2 * dr
    dzdy = (f[r[0], (r[1] + 1) % n, r[2], 2] - f[r[0], (r[1] - 1) % n, r[2], 2]) / d
    dydz = (f[r[0], r[1], (r[2] + 1) % n, 1] - f[r[0], r[1], (r[2] - 1) % n, 1]) / d
    dzdx = (f[(r[0] + 1) % n, r[1], r[2], 2] - f[(r[0] - 1) % n, r[1], r[2], 2]) / d
    dxdz = (f[r[0], r[1], (r[2] + 1) % n, 0] - f[r[0], r[1], (r[2] - 1) % n, 0]) / d
    dydx = (f[(r[0] + 1) % n, r[1], r[2], 1] - f[(r[0] - 1) % n, r[1], r[2], 1]) / d
    dxdy = (f[r[0], (r[1] + 1) % n, r[2], 0] - f[r[0], (r[1] - 1) % n, r[2], 0]) / d


    return np.array((dzdy - dydz, dxdz - dzdx, dydx - dxdy), dtype=f.dtype)


if __name__ == "__main__":
    N = 21
    c_0 = 299792458
    c2 = c_0 * c_0
    dr = 2 / N
    dt = 0.5 * dr / c_0
    T = dt * 100

    E_field = np.zeros((N, N, N, 3))
    B_field = np.zeros((N, N, N, 3))
    J_field = np.zeros((N, N, N, 3))
    curl_E = np.zeros((N, N, N, 3))
    curl_B = np.zeros((N, N, N, 3))

    mesh = np.meshgrid(np.arange(0, N), np.arange(0, N), np.arange(0, N))

    co_b = 1
    co_e = c_0

    # init
    for i in range(N):
        for j in range(N):
            for k in range(N):
                x = i - (N - 1) / 2
                y = j - (N - 1) / 2
                z = k - (N - 1) / 2

                E_field[i, j, k, 0] = 0
                E_field[i, j, k, 1] = (i == 0) * 1 * co_e
                E_field[i, j, k, 2] = 0

                B_field[i, j, k, 0] = 0
                B_field[i, j, k, 1] = 0
                B_field[i, j, k, 2] = -(i == 0) * 1 * co_b

                J_field[i, j, k, 0] = 0
                J_field[i, j, k, 1] = 0
                J_field[i, j, k, 2] = 0

    # sim
    for i in range(1000):
        e_x = E_field[mesh[0], mesh[1], mesh[2], 0] / co_e
        e_y = E_field[mesh[0], mesh[1], mesh[2], 1] / co_e
        e_z = E_field[mesh[0], mesh[1], mesh[2], 2] / co_e
        b_x = B_field[mesh[0], mesh[1], mesh[2], 0] / co_b
        b_y = B_field[mesh[0], mesh[1], mesh[2], 1] / co_b
        b_z = B_field[mesh[0], mesh[1], mesh[2], 2] / co_b

        ce_x = curl_E[mesh[0], mesh[1], mesh[2], 0] / co_e
        ce_y = curl_E[mesh[0], mesh[1], mesh[2], 1] / co_e
        ce_z = curl_E[mesh[0], mesh[1], mesh[2], 2] / co_e

        if i % 10 == 0:
            fig = plt.figure()
            ax1 = fig.add_subplot(121, projection="3d")
            ax2 = fig.add_subplot(122, projection="3d")
            ax1.quiver(mesh[0], mesh[1], mesh[2], e_x, e_y, e_z)
            ax2.quiver(mesh[0], mesh[1], mesh[2], b_x, b_y, b_z)
            ax1.set(xlim=(0.0, N - 1), ylim=(0.0, N - 1), zlim=(0.0, N - 1))
            ax2.set(xlim=(0.0, N - 1), ylim=(0.0, N - 1), zlim=(0.0, N - 1))
            fig.tight_layout()
            plt.show()

        # iter
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    curl_E[i, j, k] = curl(E_field, (i, j, k), dr, N)
                    curl_B[i, j, k] = curl(B_field, (i, j, k), dr, N)

                    E_field[i, j, k, 0] += c2 * curl_B[i, j, k, 0] * dt
                    E_field[i, j, k, 1] += c2 * curl_B[i, j, k, 1] * dt
                    E_field[i, j, k, 2] += c2 * curl_B[i, j, k, 2] * dt

                    B_field[i, j, k, 0] -= curl_E[i, j, k, 0] * dt
                    B_field[i, j, k, 1] -= curl_E[i, j, k, 1] * dt
                    B_field[i, j, k, 2] -= curl_E[i, j, k, 2] * dt
