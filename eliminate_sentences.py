f1 = open('/Users/vivanov/translator/en')
f2 = open('/Users/vivanov/translator/ru')

lines1 = f1.readlines()
lines2 = f2.readlines()

lengths1 = [len((lines1[i]).split(' ')) for i in range(len(lines1))]
lengths2 = [len((lines2[i]).split(' ')) for i in range(len(lines2))]

assert (len(lines1) == len(lines2))
allowed_indexes = []
for i in range(len(lines1)):
    if 10 < lengths1[i] < 30:
        if 10 < lengths2[i] < 30:
            allowed_indexes.append(i)

lines2write1 = []
lines2write2 = []


def cmp_indexes(x1, x2):
    return lengths1[x1] < lengths1[x2]

allowed_indexes.sort(cmp=cmp_indexes)

for i in allowed_indexes:
    lines2write1.append(lines1[i])
    lines2write2.append(lines2[i])
f1 = open('/Users/vivanov/translator/en1', 'w')
f2 = open('/Users/vivanov/translator/ru1', 'w')
f1.writelines(lines2write1)
f2.writelines(lines2write2)
