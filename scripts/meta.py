import math
from typing import List


def getUniformIntegerCountInInterval(A: int, B: int) -> int:
    # Write your code here
    """
    0-9,11-99,111-999,1111-9999,...

    """

    na: int = len(str(A))
    nb: int = len(str(B))

    ta: int = int(str(A)[0])
    for digit in str(A):
        if int(digit) > int(str(A)[0]):
            ta += 1
            break
    tb: int = int(str(B)[0])
    for digit in str(B):
        if int(digit) < int(str(B)[0]):
            tb -= 1
            break

    # closest_a: int = int("".join("1" for _ in range(na))) * ta
    # closest_b: int = int("".join("1" for _ in range(nb))) * tb
    #
    # print(closest_a, closest_b)
    lower: int = (na * 10) + ta
    upper: int = (nb * 10) + tb
    cross: int = nb - na
    result: int = upper - lower + 1 - cross
    # print(result)
    return result


def getMaxAdditionalDinersCount(N: int, K: int, M: int, S: List[int]) -> int:
    # Write your code here

    def fit_in_empty(n: int) -> int:
        return math.ceil(n / (K + 1))

    S.sort()
    result: int = 0
    result += fit_in_empty(S[0] - K - 1)
    # print(f"fit({S[0]-K-1}) -> {result}")
    for i in range(1, M):
        upper: int = S[i] - K
        lower: int = S[i - 1] + K
        result += fit_in_empty(upper - lower - 1)
        # print(f"fit({upper-lower-1}) -> {result}")

    result += fit_in_empty(N - S[-1] - 1)
    # print(f"fit({N - S[-1] - 1}) -> {result}")
    return result


def solution(test):
    return getMaxAdditionalDinersCount(*test)


def main() -> None:
    testcases: list = [[10, 1, 2, [2, 6]], [15, 2, 3, [11, 6, 14]]]
    for test in testcases:
        result = solution(test)
        print(f"{test} -> {result}")

    print("completed")


if __name__ == "__main__":
    main()
