def powerOf2(x: int):
    return (x & (x - 1) == 0) and x != 0
