def stein_gcd(a, b):
    if a == 0:
        return b
    if b == 0:
        return a

    k = 0

    # Remove common factors of 2
    while a % 2 == 0 and b % 2 == 0:
        a //= 2
        b //= 2
        k += 1

    while a != b:
        if a % 2 == 0:
            a //= 2
        elif b % 2 == 0:
            b //= 2
        elif a > b:
            a = (a - b) // 2
        else:
            b = (b - a) // 2

    return a * (2 ** k)

print(stein_gcd(12, 18))  # 6