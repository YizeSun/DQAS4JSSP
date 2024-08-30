import pennylane as qml
import numpy as np
from math import pi
import qubovert as qv
import re

job = 3
machine = 2
em = 1
timeslots = job+em

productGroup = {1:["0B","1H"],
                2:["0D","1F"],
                3:["0C","1E"],
                4:["0B","1A"],
                5:["0D","1C"],
                6:["0E","1A"],
                7:["0E","1E"],
                8:["0E","1F"],
                9:["0B","1C"],
                10:["0B","1C"],
                11:["0B","1D"],
                12:["0E","1C"],
                13:["0C","1A"],
                14:["0C","1D"],
                15:["0B","1H"],
                16:["0E","1A"],
                17:["0C","1H"],
                18:["0D","1D"],
                19:["0B","1C"],
                20:["0B","1G"],
               }

timeTable = {
    1:[1,2],
    2:[11,11],
    3:[3,3],
    4:[13,15],
    5:[12,13],
    6:[6,6],
    7:[8,8],
    8:[7,7],
    9:[18,0],
    10:[20,0],
    11:[19,0],
    12:[10,12],
    13:[15,16],
    14:[16,18],
    15:[17,17],
    16:[9,9],
    17:[4,10],
    18:[5,5],
    19:[14,14],
    20:[2,4]
}

dueTable = {
    1:[1,2],
    2:[11,11],
    3:[3,3],
    4:[13,15],
    5:[12,13],
    6:[6,6],
    7:[8,8],
    8:[7,7],
    9:[18,20],
    10:[20,21],
    11:[19,19],
    12:[10,12],
    13:[15,16],
    14:[16,17],
    15:[17,18],
    16:[9,9],
    17:[4,10],
    18:[5,5],
    19:[14,14],
    20:[2,4]
}

#end = 14
endJob={
    0:0,
    1:14
}


idleTime = 1
avaiableSlots = [[19,1],[20,1],[21,1]]
machineNumber = 1
timeLen = len(avaiableSlots)

num_shots = 1000
p=2

def transferModel (model):
    m = {}
    for key in model.keys():
        if len(key)==0:
            m[()]=model[key]
        if len(key)==1:
            m[(int(re.search('\d+', key[0]).group()),)]=model[key]
        if len(key)==2:
            m[(int(re.search('\d+', key[0]).group()),int(re.search('\d+', key[1]).group()))]=model[key]
    return m

def transferHamiltonian (dic):
    H = 0
    for key in dic.keys():
        if len(key)==0:
            H+=dic[key]*qml.Identity(1)
        if len(key)==1:
            H+=dic[key]*qml.PauliZ(key[0])@qml.PauliZ(key[0])
        if len(key)==2:
            H+=dic[key]*qml.PauliZ(key[0])@qml.PauliZ(key[1])
    return H


def compute_cvar(probabilities, values, alpha):
    """
    Auxilliary method to computes CVaR for given probabilities, values, and confidence level.

    Attributes:
    - probabilities: list/array of probabilities
    - values: list/array of corresponding values
    - alpha: confidence level

    Returns:
    - CVaR
    """
    sorted_indices = np.argsort(values)
    probs = np.array(probabilities)[sorted_indices]
    vals = np.array(values)[sorted_indices]
    cvar = 0
    total_prob = 0
    for i, (p, v) in enumerate(zip(probs, vals)):
        done = False
        if p >= alpha - total_prob:
            p = alpha - total_prob
            done = True
        # print(total_prob)
        total_prob += p
        cvar += p * v
    cvar /= total_prob
    return cvar

def findJobs (timeTable):
    restJob=[]
    for key in timeTable.keys():
        for i in range(machine):
            if timeTable[key][i]==0:
                restJob.append([key,i])
    return restJob

def findDue (dueTable,restJob):
    dueTime=restJob
    for job in dueTime:
        #print(job)
        job.append(dueTable[job[0]][job[1]])
    return dueTime

def transformBits (x):
    result=[]
    for b in x:
        if (int(b)==0):
            result.append(0)
        else:
            result.append(1)
    return result

def removeZero (x):
    for i in range(len(x)-1,-1,-1):
        if x[i] == 100:
            x.remove(100)
    return x

def one_start_constraint(result):
    for i in range(job):
        n=0
        for j in range(timeLen):
            if result[i][j]==1:
                n+=1
        if n>1 or n==0:
            return False
    return True

def share_time_constraint(result):
    for j1 in range(job):
        for j2 in range(job):
            for t1 in range(timeLen):
                for t2 in range(timeLen):
                    if (result[j1][t1]==1)&(result[j2][t2]==1)&(t1==t2)&(j1!=j2):
                        return False
    return True


def process_order_constraint(result):
    for j in range(job):
        for t in range(timeLen):
            if result[j][t] == 1:
                for m in range(machine):
                    if (timeTable[restJob[j][0]][m] != 0) & (m < restJob[j][1]) & (
                            timeTable[restJob[j][0]][m] > avaiableSlots[t][0]):
                        return False
                    if (timeTable[restJob[j][0]][m] != 0) & (m > restJob[j][1]) & (
                            timeTable[restJob[j][0]][m] < avaiableSlots[t][0]):
                        return False
    for j1 in range(job):
        for j2 in range(job):
            for t1 in range(timeLen):
                for t2 in range(timeLen):
                    if (result[j1][t1] == 1) & (result[j2][t2] == 1) & (j1 != j2) & (
                            restJob[j1][1] < restJob[j2][1]) & (avaiableSlots[t1][0] > avaiableSlots[t2][0]):
                        return False

    return True

def checkValid(result):
    return (one_start_constraint(result)&share_time_constraint(result)&process_order_constraint(result))

restJob=findDue (dueTable,findJobs (timeTable))

N = job*len(avaiableSlots)+em

# create the variables
x = {i: qv.boolean_var('x(%d)' % i) for i in range(N)}

model1 = 0
for i in range(job):
    y=0
    for j in range(timeLen):
        #print(x[i*time+j])
        y+=(1-x[i*timeLen+j])/2
        model1 += (y-1)**2
for i in range(N-em,N):
    #print(x[i])
    model1 += (1-(1-x[i])/2)**2

model2 = 0
for t in range(timeslots):
    if t in range(em):
        for i in range(N-em,N):
            #print(x[i])
            model2+=(1-(1-x[i])/2)
    else:
        y=0
        for j1 in range(job):
            for j2 in range(j1+1,job):
                #print(x[j1*time+t-em],x[j2*time+t-em])
                model2 += (1-x[j1*timeLen+t-em])/2*(1-x[j2*timeLen+t-em])/2

model3 = 0
for i,j1 in enumerate(restJob):
    for j,t in enumerate(avaiableSlots):
        for m in range(machine):
            if (timeTable[j1[0]][m]>=t[0]) & (timeTable[j1[0]][m]!=0)&(m<t[1]):
                #print(t[0],x[i*time+j])
                model3+=(1-x[i*timeLen+j])/2
            if (timeTable[j1[0]][m]<=t[0]) & (timeTable[j1[0]][m]!=0)&(m>t[1]):
                #print(t[0],x[i*time+j])
                model3+=(1-x[i*timeLen+j])/2
for i,j1 in enumerate(restJob):
    for j,j2 in enumerate(restJob):
        for m,t1 in enumerate(avaiableSlots):
            for n,t2 in enumerate(avaiableSlots):
                if (j1[0]==j2[0])&(t1[1]+1==t2[1])&(t1[0]>t2[0]):
                    model3+=(1-x[i*timeLen+m])/2*(1-x[j*timeLen+n])/2

if model3==0:
    model3={}

model4 = 0
for i,j1 in enumerate(restJob):
    for j,t in enumerate(avaiableSlots):
        if t[0]>=j1[2]:
            #print(t[0]-j1[2],x[i*time+j])
            model4 +=(t[0]-j1[2])*(1-x[i*timeLen+j])/2
        else:
            #print(j1[2]-t[0],x[i*time+j])
            model4 +=(j1[2]-t[0])*(1-x[i*timeLen+j])/2

# produce group
model5 = 0
for i,j1 in enumerate(restJob):
    for j,j2 in enumerate(restJob):
        for m1,t1 in enumerate(avaiableSlots):
            for m2,t2 in enumerate(avaiableSlots):
                if (t1[0]+1==t2[0])&(j1[1]==j2[1])&(productGroup[j1[0]][j1[1]]!=productGroup[j2[0]][j2[1]]):
                    #print(t1,t2)
                    #print(x[i*timeLen+m1],x[j*timeLen+m2])
                    model5 += (1-x[i*timeLen+m1])/2*(1-x[j*timeLen+m2])/2

for i,j1 in enumerate(restJob):
    for m,t1 in enumerate(avaiableSlots):
        if (timeTable[endJob[t1[1]]][t1[1]]+1==t1[0])&(productGroup[endJob[t1[1]]][t1[1]]!=productGroup[j1[0]][j1[1]]):
            #print(x[i*time+m])
            model5 += (1-x[i*timeLen+m])/2

dic1=transferModel(model1)
dic2=transferModel(model2)
dic3=transferModel(model3)
dic4=transferModel(model4)
dic5=transferModel(model5)

H1=transferHamiltonian(dic1)
H2=transferHamiltonian(dic2)
H3=transferHamiltonian(dic3)
H4=transferHamiltonian(dic4)
H5=transferHamiltonian(dic5)

H=3*H1+3*H2+3*H3+3*H4+2*H5

dev = qml.device("default.qubit",wires=H.wires)
@qml.qnode(dev)

def eval_bitstring(state):
    qml.BasisState(np.array(state), wires=range(N))
    return qml.expval(H)


class Objective:
    """
    Wrapper for objective function to track the history of evaluations
    """

    def __init__(self, var_form, alpha, optimal=None):
        self.history = []
        self.var_form = var_form
        self.alpha = alpha
        self.optimal = optimal
        self.opt_history = []

    def evaluate(self, thetas):
        # create and run circuit
        counts = self.var_form(thetas)

        # evaluate counts
        probabilities = np.zeros(len(counts))
        values = np.zeros(len(counts))
        for i, (x, p) in enumerate(counts.items()):
            values[i] = eval_bitstring(transformBits(x))
            probabilities[i] = p / num_shots

        # evaluate cvar
        cvar = compute_cvar(probabilities, values, self.alpha)
        self.history += [(cvar-3.5)/(104.0-3.5)]
        return cvar



