import random

rolls = []
six_count = 0
one_count = 0
two_sixes_in_row = 0

for i in range(20):   # roll the die 20 times
    roll = random.randint(1, 6)
    rolls.append(roll)

    if roll == 6:
        six_count += 1
    if roll == 1:
        one_count += 1

# count two 6s in a row
for i in range(len(rolls) - 1):
    if rolls[i] == 6 and rolls[i + 1] == 6:
        two_sixes_in_row += 1

print("Die rolls:", rolls)
print("Number of times 6 was rolled:", six_count)
print("Number of times 1 was rolled:", one_count)
print("Number of times two 6s appeared in a row:",
       two_sixes_in_row)