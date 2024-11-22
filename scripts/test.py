import math
import os


def fit(n, k):
    if n < (k + 2):
        return 1
    r, t = divmod(n, (k + 1))
    extra = t != 0
    print(n / (k + 1))

    return math.ceil(n / (k + 1))


def main():
    n = 7
    k = 3
    print(fit(n, k))


if __name__ == "__main__":
    main()
