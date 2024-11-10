import random

def generate_vector(id):
    num1 = random.randint(1, 10)
    num2 = random.randint(1, 10)
    num3 = random.randint(1, 10)
    return [id, num1, num2, num3]

vectors = [generate_vector(str(i)) for i in range(1, 121)]

for vector in vectors:
    print(str(vector) + ",")