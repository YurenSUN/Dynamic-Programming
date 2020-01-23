
# coding: utf-8
"""
The script to calculate the what k to choose for next period given k of this period
Author: Yuren Sun
Last modified: Dec 4, 2019
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def value_function_iteration():
    """
    loop over period to find the k' correspond to k
    return: k_next, list of chosen k' to max v(k) given k
    """
    period = 1  # count the period
    cur_diff = 10  # max difference of v(k) - v(k')
    # list of chosen k' to max v(k) given k
    k_next = np.zeros(len(k_grid)).tolist()
    # value of k' for the last period is all 0
    v_k_next = np.zeros(len(k_grid)).tolist()

    while cur_diff > threshold:
        # list of max v(k) given different k
        max_v_k = np.zeros(len(k_grid)).tolist()
        # find the k max to max cur
        # TODO improve the algorithm with vectorization
        for i in range(len(k_grid)):  # cur k
            # v_k_next_substitute = np.zeros(n + 1).tolist()
            for j in range(len(k_grid)):  # next k (k')
                # find the k' max v(k)
                # current value of k with different k'
                cur_v_k = np.log(max((k_grid[i] ** alpha) * (l ** (1 - alpha)) - (
                    k_grid[j] - (1 - delta) * k_grid[i]), 0)) + beta * v_k_next[j]

                # ignore the case where two equivalent v(k)
                if cur_v_k > max_v_k[i]:
                    max_v_k[i] = cur_v_k
                    k_next[i] = k_grid[j]

        # compare |v(k) - v(k')| and threshold
        cur_diff = np.amax(np.absolute(
            list(map(lambda x: x[0] - x[1], zip(max_v_k, v_k_next)))))
        print("The difference between v(k) and v(k') in %ith period is %f." %
              (period, cur_diff))

        v_k_next = max_v_k
        period += 1

    print("The difference between v(k) and v(k') is less then the threshold in %ith period.\n" % (period - 1))

    return k_next, v_k_next


if __name__ == "__main__":
    # params
    alpha = 0.333333333
    delta = 0.06
    beta = 0.96
    psi = 0.25
    l = 1
    epsilon = 1E-10
    n = 50  # n needs to be even to have steady state k in k_grid
    threshold = 1E-3  # threshold to exit the loop

    # set up
    k_ss = ((1 / beta - 1 + delta) /
            (alpha * l ** (1 - alpha))) ** (1 / (alpha - 1))
    k_l = k_ss * (1 - psi)
    k_u = k_ss * (1 + psi)
    k_grid = np.arange(k_l, k_u + (k_u - k_l) / n, (k_u - k_l) / n)

    print("Steady state k: %f" % (k_ss))
    print("k_grid: {}\n".format(k_grid))

    k_next,  v_k = value_function_iteration()
    # print("k_next: {}\n".format(k_next))

    # k_to_next = dict(zip(k_grid, k_next))
    # print("Current given k: chosen k' (for next period) {}\n".format(k_to_next))

    # investment
    investment = np.subtract(k_next, np.multiply(1-delta, k_grid))
    # plot the investment
    plt.plot(k_grid, investment)
    plt.xlabel("Given k")
    plt.ylabel("Investment")
    # plt.show()
    plt.savefig('k_investment_output.png')
    plt.clf()

    # plot the k with k next
    plt.plot(k_grid, k_next)
    plt.xlabel("Given k")
    plt.ylabel("Chosen k")
    # plt.show()
    plt.savefig('k_kNext_output.png')
    plt.clf()

    # plot the v(k) of k
    plt.plot(k_grid, v_k)
    plt.xlabel("Given k")
    plt.ylabel("v(k)")
    # plt.show()
    plt.savefig('k_v(k)_output.png')

    # output k, k', v(k), investment to csv
    names = ["k", "k'", "investment", "v(k)"]
    metadata_df = pd.DataFrame(np.array((k_grid, k_next, investment,v_k)).transpose(), columns=names)
    metadata_df.to_csv("neoclassical_k_kNext_vk.csv", index=False)
