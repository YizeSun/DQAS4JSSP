import json
import pennylane as qml
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import JSSP
import time

np.random.seed(1234)
json_file_path = "learned_struc.json"

with open(json_file_path, 'r') as j:
     contents = json.loads(j.read())

print(contents)
num_qubits = JSSP.N
n = 0
layer = 1

operators = []
for content in contents[0]:
    operators.append([content.split(",")[1]])

for i,content in enumerate(contents[1]):
    operators[i].append(content[1])

for i,op in enumerate(operators):
    if op[0] == "U3" or op[0] == "CU3" or op[0] == "CU3-single" or op[0] == "CU33":
        operators[i].append(n)
        n += len(op[1])*3
    elif op[0] == "CNOT-set0":
        operators[i].append(n)
        n += len(op[1])*6
    elif op[0] == "rz-CNOT-rz" or op[0] == "rz-CNOTT-rz":
        operators[i].append(n)
        n += len(op[1])*2
    elif op[0] == "U1" or op[0] == "RX" or op[0] == "RY" or op[0] == "RZ" or op[0] == "CRZ":
        operators[i].append(n)
        n += len(op[1])*1
    else:
        operators[i].append(n)

print(operators)

def transform(params):
    params=params.reshape(layer,n*3)

    theta=[[0] * n] * 3
    phi=[[0] * n] * 3
    delta=[[0] * n] * 3

    for d in range(layer):
        for i in range(3):
            for j in range(n):
                if i==0:
                    theta[d][j]=params[d][3*i+j]
                if i==1:
                    phi[d][j]=params[d][3*i+j]
                if i==2:
                    delta[d][j]=params[d][3*i+j]

    return theta,phi,delta

def encoding_block(num_qubits):  # for cart pole
    for w in range(num_qubits):
        qml.RX(np.pi,wires=w)

def measurement_1():
    return qml.counts()

def measurement_2():
    return qml.expval(JSSP.H)

def entanglement1(op):
    for wire in op:
        qml.CNOT(wires=[wire,(wire+1)%num_qubits])

def entanglement2(op):
    for wire in op:
        if wire % 2 == 0:
            qml.CNOT(wires=[wire, (wire + 1) % num_qubits])
        else:
            continue
    for wire in op:
        if wire % 2 == 1:
            qml.CNOT(wires=[wire, (wire + 1) % num_qubits])
        else:
            continue

def layers(params):
    params = params.reshape(layer,n)
    for d in range(layer):
        for op in operators:
            # -- u3 & cu3 gate --
            if op[0] == "U3":
                for i,wire in enumerate(op[1]):
                     qml.U3(theta=params[d][op[2]+i*3]
                          , phi=params[d][op[2]+i*3+1]
                          , delta=params[d][op[2]+i*3+2]
                          , wires=wire)
            elif op[0] == "CU3":
                for i, wire in enumerate(op[1]):
                    def CU3(p_w, target):
                        qml.U3(theta=p_w[d][op[2]+i*3]
                             , phi=p_w[d][op[2]+i*3+1]
                             , delta=p_w[d][op[2]+i*3+2]
                             , wires=target)
                    qml.ctrl(CU3, control=wire)(params, (wire+1)%num_qubits)
            elif op[0] == "CU3-single":
                for i, wire in enumerate(op[1]):
                    def CU3(p_w, target):
                        qml.U3(theta=p_w[d][op[2]+i*3]
                             , phi=p_w[d][op[2]+i*3+1]
                             , delta=p_w[d][op[2]+i*3+2]
                             , wires=target)
                    qml.ctrl(CU3, control=wire)(params, (wire+1)%num_qubits)
            elif op[0] == "CU33":
                for i, wire in enumerate(op[1]):
                    def CU3(p_w, target):
                        qml.U3(theta=p_w[d][op[2]+i*3]
                             , phi=p_w[d][op[2]+i*3+1]
                             , delta=p_w[d][op[2]+i*3+2]
                             , wires=target)
                    qml.ctrl(CU3, control=wire)(params, (wire+2)%num_qubits)
            # -- simple one qubit gate --
            elif op[0] == "U1":
                for i,wire in enumerate(op[1]):
                    qml.U1(params[d][op[2]+i*1], wires=wire)
            elif op[0] == "RX":
                for i, wire in enumerate(op[1]):
                    qml.RX(params[d][op[2]+i*1], wires=wire)
            elif op[0] == "RY":
                for i, wire in enumerate(op[1]):
                    qml.RY(params[d][op[2]+i*1], wires=wire)
            elif op[0] == "RZ":
                for i, wire in enumerate(op[1]):
                    qml.RZ(params[d][op[2]+i*1], wires=wire)
            elif op[0] == "X":
                for i, wire in enumerate(op[1]):
                    qml.X(wires=wire)
            elif op[0] == "SX":
                for i, wire in enumerate(op[1]):
                    qml.SX(wires=wire)
            elif op[0] == "T":
                for i, wire in enumerate(op[1]):
                    qml.T(wires=wire)
            elif op == "Ta":
                for i, wire in enumerate(op[1]):
                    qml.adjoint(qml.T(wires=wire))
            elif op[0] == "H":
                for i, wire in enumerate(op[1]):
                    qml.Hadamard(wires=wire)
            elif op[0] == "SQH":
                unitary = (1./4+1.j/4) * np.array([-1.j*(np.sqrt(2)+2.j), -1.j*np.sqrt(2)],
                                                [-1.j*np.sqrt(2), 1.j*(np.sqrt(2)-2.j)])
                qml.QubitUnitary(unitary, wires=op[1])
            elif op[0] == "S":
                for i, wire in enumerate(op[1]):
                    qml.S(wires=wire)
            elif op[0] == "E":
                for i, wire in enumerate(op[1]):
                    qml.Identity(wires=wire)
            # -- two qubits gate --
            elif op[0] == "SWAP":
                qml.SWAP(wires=op[1][0])
            elif op[0] == "SQSWAP":
                unitary = np.array([1., 0., 0., 0.],
                                    [0., (1./2+1.j/2), (1./2-1.j/2), 0.],
                                    [0., (1./2-1.j/2), (1./2+1.j/2), 0.],
                                    [0., 0., 0., 1.],)
                qml.QubitUnitary(unitary, wires=op[1])
            elif op[0] == "CZ":
                for i, wire in enumerate(op[1]):
                    qml.CZ(wires=[wire, (wire + 1) % num_qubits])
            elif op[0] == "CRZ":
                for i, wire in enumerate(op[1]):
                    qml.CRZ(params[d][op[2]+i*1], wires=[wire, (wire + 1) % num_qubits])
            elif op[0] == "CNOT":
                entanglement1(op[1])
            elif op[0] == "CNOT-set0":
                for i, wire in enumerate(op[1]):
                    qml.CNOT(wires=[wire, (wire + 1) % num_qubits])
                    qml.U3(theta=params[d][op[2]+i*6]
                         , phi=params[d][op[2]+i*6+1]
                         , delta=params[d][op[2]+i*6+2]
                         , wires=wire)
                    qml.U3(theta=params[d][op[2]+i*6+3]
                         , phi=params[d][op[2]+i*6+4]
                         , delta=params[d][op[2]+i*6+5]
                         , wires=(wire + 1) % num_qubits)
            elif op[0] == "CNOT-set1":
                for i, wire in enumerate(op[1]):
                    qml.CNOT(wires=[wire, (wire + 1) % num_qubits])
                    qml.U3(theta=params[d][op[2]+i*3]
                         , phi=params[d][op[2]+i*3+1]
                         , delta=params[d][op[2]+i*3+2]
                         , wires=(wire + 1) % num_qubits)
            elif op[0] == "rz-CNOT-rz":
                for i, wire in enumerate(op[1]):
                    qml.RZ(params[d][op[2]+i*2], wires=[(wire + 1)%num_qubits])
                    qml.CNOT(wires=[wire, (wire + 1) % num_qubits])
                    qml.RZ(params[d][op[2]+i*2+1], wires=[(wire + 1)%num_qubits])
            elif op[0] == "rz-CNOTT-rz":
                for i, wire in enumerate(op[1]):
                    qml.RZ(params[d][op[2]+i*2], wires=[(wire + 2)%num_qubits])
                    qml.CNOT(wires=[wire, (wire + 2) % num_qubits])
                    qml.RZ(params[d][op[2]+i*2+1], wires=[(wire + 2)%num_qubits])
            elif op[0] == "CNOTT":
                for i, wire in enumerate(op[1]):
                    qml.CNOT(wires=[wire, (wire + 2) % num_qubits])
            elif op[0] == "HCNOT":
                for i, wire in enumerate(op[1]):
                    qml.Hadamard(wires=[(wire + 1) % num_qubits])
                    qml.CNOT(wires=[wire, (wire + 1) % num_qubits])
            elif op[0] == "CNOTH":
                for i, wire in enumerate(op[1]):
                    qml.CNOT(wires=[wire, (wire + 1) % num_qubits])
                    qml.Hadamard(wires=[(wire + 1) % num_qubits])
            elif op[0] == "Ising-ZZ":
                for i, wire in enumerate(op[1]):
                    qml.IsingZZ(params[d][op[2]+i*1], wires=[wire, (wire + 1) % num_qubits])
            elif op[0] == "ZZ":
                for i, wire in enumerate(op[1]):
                    qml.CNOT(wires=[wire, (wire+1)%num_qubits])
                    qml.RZ(params[d][op[2]+i*1], wires=wire)
                    qml.CNOT(wires=[wire, (wire+1)%num_qubits])
            elif op[0] == "QFT":
                qml.QFT(wires=op[1])


#dev = qml.device("default.qubit",wires=num_qubits,shots=1000)
dev_ideal = qml.device('default.mixed', wires=num_qubits,shots=1000)
noise_gate = qml.PhaseDamping
noise_strength = 0.2
dev = qml.transforms.insert(noise_gate, noise_strength)(dev_ideal)
@qml.qnode(dev)
def make_circuit(params):
    p = 2
    params = params.reshape(p, 6)
    for wire in JSSP.H.wires:
        qml.RX(np.pi, wires=wire)
    for n in range(p):
        qml.RY(params[n][0], wires=0)
        qml.RY(params[n][1], wires=1)
        for wire in range(num_qubits - 1):
            qml.CZ(wires=[wire, (wire + 1) % num_qubits])
        qml.RY(params[n][2], wires=5)
        qml.RY(params[n][3], wires=6)
        for wire in range(num_qubits - 1):
            qml.CNOT(wires=[wire, (wire + 1) % num_qubits])
        qml.RY(params[n][4], wires=1)
        qml.RY(params[n][5], wires=2)
    return qml.counts()
    #encoding_block(num_qubits)
    #qml.Barrier(wires=range(num_qubits))
    #layers(params)
    #qml.Barrier(wires=range(num_qubits))
    #return measurement_1()

dev = qml.device("default.qubit",wires=num_qubits,shots=1000)
@qml.qnode(dev)
def result_circuit(params):
    p = 2
    params = params.reshape(p, 6)
    for wire in JSSP.H.wires:
        qml.RX(np.pi, wires=wire)
    for n in range(p):
        qml.RY(params[n][0], wires=0)
        qml.RY(params[n][1], wires=1)
        for wire in range(num_qubits - 1):
            qml.CZ(wires=[wire, (wire + 1) % num_qubits])
        qml.RY(params[n][2], wires=5)
        qml.RY(params[n][3], wires=6)
        for wire in range(num_qubits - 1):
            qml.CNOT(wires=[wire, (wire + 1) % num_qubits])
        qml.RY(params[n][4], wires=1)
        qml.RY(params[n][5], wires=2)
    return qml.counts()
    #encoding_block(num_qubits)
    #qml.Barrier(wires=range(num_qubits))
    #layers(params)
    #qml.Barrier(wires=range(num_qubits))
    #return measurement_1()

#dev = qml.device("default.qubit", wires=num_qubits,shots=1000)
dev_ideal = qml.device('default.mixed', wires=num_qubits,shots=1000)
noise_gate = qml.PhaseDamping
noise_strength = 0.2
dev = qml.transforms.insert(noise_gate, noise_strength)(dev_ideal)
@qml.qnode(dev)
def circuit_base(params):
    params=params.reshape(2,num_qubits)
    encoding_block(num_qubits)
    for n in range(2):
        for wire in range(num_qubits - 1):
            if wire % 2 == 0:
                qml.CNOT(wires=[wire, wire + 1])
            else:
                continue
            if wire % 2 == 1:
                qml.CNOT(wires=[wire, wire + 1])
            else:
                continue
        for wire, param in enumerate(params[n]):
            qml.RY(param, wires=wire)

    return measurement_1()

dev = qml.device("default.qubit", wires=num_qubits, shots=1000)
@qml.qnode(dev)
def resultCircuit_base(params):
    params=params.reshape(2,num_qubits)
    encoding_block(num_qubits)
    for n in range(2):
        for wire in range(num_qubits - 1):
            if wire % 2 == 0:
                qml.CNOT(wires=[wire, wire + 1])
            else:
                continue
            if wire % 2 == 1:
                qml.CNOT(wires=[wire, wire + 1])
            else:
                continue
        for wire, param in enumerate(params[n]):
            qml.RY(param, wires=wire)

    return measurement_1()

energy=[]
def cost_execution(params):

    cost = make_circuit(params)

    energy.append([cost])
    return cost

def train(circuit, res_circuit, initial_point, name):
    alpha = 0.5  # confidence levels to be evaluated
    #objectives = {}  # set of tested objective functions w.r.t. alpha
    epochs = 200
    n = 0
    data = {}
    data_mean = [100 for i in range(200)]
    data_std = [100 for i in range(200)]
    for i in range(200):
        data[i] = []

    time_start = time.time()
    for epoch in range(epochs):
        initial_point = np.random.rand(num_qubits * 2)
        objectives = JSSP.Objective(circuit, alpha)
        out = minimize(objectives.evaluate, x0=initial_point, method="COBYLA", options={"maxiter": 200})
        print(out)
        print(epoch)
        optimal_params = out['x']
        result = res_circuit(optimal_params)
        for key, value in result.items():
            if (value == max(result.values())):
                x = np.array(JSSP.transformBits(key))
                print(x, value)
        result = x[:-JSSP.em].reshape((JSSP.job, JSSP.timeLen))
        if (JSSP.checkValid(result)):
            n += 1

        #plt.plot(range(len(objectives.history)),objectives.history)
        #plt.show()
        for i, result in enumerate(objectives.history):
            data[i] = np.append(data[i], result)
    print(n / epochs)
    time_end = time.time()
    for key in data.keys():
        if len(data[key]) != 0:
            data_mean[key] = np.mean(data[key])
            data_std[key] = np.std(data[key])
    #print(data_mean)
    data_mean = JSSP.removeZero(data_mean)
    data_std = JSSP.removeZero(data_std)
    data_max = [data_mean[i] + data_std[i] for i in range(len(data_std))]
    data_min = [data_mean[i] - data_std[i] for i in range(len(data_std))]
    np.savez(name,data_mean,data_std,data_max,data_min)

    T = time_end - time_start
    print(T)

    return data_mean, data_min, data_max
# add seed
initial_point_base = np.random.rand(num_qubits*2)
initial_point_nas = np.random.rand(12)
#initial_point_nas = np.random.rand(n*layer)
baseline_name = "baseline-noise-10-200.npz"
auto_name = "op10-9-06-l2-5-5-0.10-top1-noise.npz"
#base_data = np.load("baseline-noise.npz")
data_mean_base, data_min_base, data_max_base = train(circuit_base, resultCircuit_base, initial_point_base, baseline_name)
data_mean_nas, data_min_nas, data_max_nas = train(make_circuit, result_circuit, initial_point_nas, auto_name)

plt.plot(range(len(data_mean_base)), data_mean_base, color='deeppink', label='baseline')
plt.fill_between(range(len(data_max_base)), data_min_base, data_max_base, color='violet', alpha=0.5)

#plt.plot(range(len(data_mean_nas)), data_mean_nas, color='royalblue', label='auto')
#plt.fill_between(range(len(data_max_nas)), data_min_nas, data_max_nas, color='blue', alpha=0.5)

y_major_locator = MultipleLocator(0.25)
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
#plt.xlim(0, 100)
plt.ylim(-0.05, 0.75)
plt.legend()
plt.show()