from random import randint, sample
from numpy import random

# modes = ['a' for i in range(18)]
# for i, x in enumerate(modes):
#     print(i, x)
# rand_index = random.randint(0, 5, randint(0, 6 - 1))
# rand_index = random.choice(a=range(6), replace=False, size=6)
rand_indexs = sample(range(6), 5)
print(rand_indexs)
