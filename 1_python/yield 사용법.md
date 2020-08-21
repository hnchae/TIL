### yield

```python
import random

def lottery():
    for i in range(6):
        yield random.randint(1, 40)

    yield random.randint(1, 15)

for random_number in lottery():
    print('next number {}'.format(random_number))


def fib():
    a, b = 1, 1
    while True:
        yield a
        a, b = b, a+b

count =0
for n in fib():
    print(n)                                                                    
    if count == 10:
        break
    count +=1

```

