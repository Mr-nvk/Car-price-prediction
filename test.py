import csv
import time
tic = time.clock()
Rd = open("autos.csv","r")
Fd = list(csv.reader(Rd))
Nd = []
for i in range(1,len(Fd)):
    Bl = [float(Fd[i][4]),float(Fd[i][11]),float(Fd[i][9])]
    Nd.append(Bl)

Trainingdata = Nd[0:int((0.75*len(Fd)))]
Testingdata = Nd[int((0.75*len(Fd))):]
print(len(Fd))
print(len(Trainingdata))
print(len(Testingdata))
toc = time.clock()
n = toc - tic
print(n)