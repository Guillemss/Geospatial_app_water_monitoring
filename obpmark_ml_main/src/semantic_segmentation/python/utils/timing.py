from functools import wraps
from time import time, sleep


def measure_time(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        return result, te-ts
    return wrap

# Example how to use the timing decorator
@measure_time
def hello():
    print('hello')
    sleep(1)
    print('world')
    return 0

if __name__ == "__main__":
    print("(Return value, Execution time)\n", hello())
