import random

def experiment():
    u = ["R"] * 3 + ["A"] * 4 + ["N"] * 2

    z = random.randint(1, 6)

    if z in [2, 3, 5]:
        u.append("N")
    elif z == 6:
        u.append("R")
    else:  
        u.append("A")

    return random.choice(u)

N = 1_000_000
r = 0

for _ in range(N):
    if experiment() == "R":
        r += 1

p = r / N
print("Probabilitatea simulata de a extrage o bila rosie este:", p)
print("Probabilitatea teoretica de a extrage o bila rosie este:", 19/60, "≈", 0.31666)


#probabilitatea teoretica este: P(r) = P1 * Pr1 + P2 * Pr2 + P3 * Pr3 = 3/6 * 3/10 + 1/6 * 4/10 + 2/6 * 3/10 = 19/60 ≈ 0.31666