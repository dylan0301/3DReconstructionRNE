from time import perf_counter
from numba import jit

# 일반적인 loop
def pure_sum(n):
    result = 0
    for i in range(0, n+1):
        result += i
    return result

# Numba 모듈 사용
@jit(nopython=True, cache=True)
def numba_sum(n):
    result = 0
    for i in range(0, n+1):
        result += i
    return result

# 시간 재기: 일반
start = perf_counter()
pure_sum(100000000)
print(perf_counter() - start)

# 시간 재기에 앞서 먼저 Compile을 해준다.
numba_sum(1)

# 시간 재기: Numba
start = perf_counter()
numba_sum(100000000)
print(perf_counter() - start)