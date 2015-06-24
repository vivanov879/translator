import time_to_sleep
import time
import podai
import threading



f1 = open('/Users/vivanov/translator/en')
f2 = open('/Users/vivanov/translator/ru')
lines1 = f1.readlines()
lines2 = f2.readlines()

assert (len(lines1) == len(lines2))
allowed_indexes = []
for i in range(len(lines1)):
    if 10 < len((lines1[i]).split(' ')) < 30:
        if  10 < len((lines2[i]).split(' ')) < 30:
            allowed_indexes.append(i)

lines2write1 = []
lines2write2 = []
for i in allowed_indexes:
    lines2write1.append(lines1[i])
    lines2write2.append(lines2[i])
f1 = open('/Users/vivanov/translator/en1', 'w')
f2 = open('/Users/vivanov/translator/ru1', 'w')
f1.writelines(lines2write1)
f2.writelines(lines2write2)
