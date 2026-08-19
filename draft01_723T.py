import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import scipy.linalg
import torch
import numqi
import opt_einsum

from utils import get_rotation_axis

from zzz233 import to_pickle_wrapper, from_pickle_wrapper
to_pickle = to_pickle_wrapper('723T.pkl')
from_pickle = from_pickle_wrapper('723T.pkl')

if torch.get_num_threads() != 1:
    torch.set_num_threads(1)

np_rng = np.random.default_rng()
hfb = lambda x: bin(x)[2:].rjust(7,'0')

# what's the smallest ((n,2,3)) code that admit transversal T gate


class QECCn23TModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_qubit = 7
        error_str,error_torch = numqi.qec.make_pauli_error_list_sparse(num_qubit, distance=3, kind='torch-csr01')
        self.num_qubit = num_qubit
        self.error_torch = error_torch.clone().to(torch.complex128)
        # tmp0 = np.sort(np_rng.permutation(2**num_qubit)[:120])
        # tmp0 = np.array([3,5,6,7,9,15,17,23,24,25,27,29,34,36,38,39,40,46,48,54,56,57,58,60,67,69,70,71,73,79,81,87,88,89,91,93,98,100,102,103,104,110,112,118,120,121,122,124])
        # self.basis = torch.tensor(np.eye(2**num_qubit)[:,tmp0], dtype=torch.complex128)
        # self.manifold = numqi.manifold.Stiefel(self.basis.shape[1], rank=2, dtype=torch.float64)
        tmp0 = np.array([5,6,24,27,33,34,37,38,45,46,53,54,69,70,88,91,97,98,101,102,109,110,117,118], dtype=np.int64)
        self.basis0 = torch.tensor(np.eye(2**num_qubit)[:,tmp0], dtype=torch.complex128)
        self.manifold0 = numqi.manifold.Sphere(self.basis0.shape[1], dtype=torch.float64)
        tmp0 = np.array([9,10,17,18,25,26,29,30,36,39,57,58,73,74,81,82,89,90,93,94,100,103,121,122], dtype=np.int64)
        self.basis1 = torch.tensor(np.eye(2**num_qubit)[:,tmp0], dtype=torch.complex128)
        self.manifold1 = numqi.manifold.Sphere(self.basis1.shape[1], dtype=torch.float64)

        self.manifold_su2 = numqi.manifold.SpecialOrthogonal(2, batch_size=num_qubit, dtype=torch.complex128)
        self.gateT = torch.tensor(numqi.gate.rz(np.pi/4), dtype=torch.complex128)

        N = num_qubit
        tmp0 = [y for x in range(N) for y in [(2,2), (N+x,x)]]
        self.contract_expr = opt_einsum.contract_expression([2]*(N+1), list(range(N))+[2*N], [2]*(N+1), list(range(N,2*N))+[2*N+1], *tmp0, [2*N+1,2*N])
        self.lambda_target = None
        self.zero_mask = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1,1,1,0,1,0,1,0,1,1,1,1,0,1,0,1,0,1,1,1,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                    0,1,1,1,0,1,1,1,0,0,1,1,1,0,1,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,
                    1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,0], dtype=torch.bool)
        self.nonzero_value = torch.tensor([0.5, -0.25, 0.25, -0.5,  0.25, 0.25, -0.5, -0.25, -0.25, 0.5, 0.25, -0.25, 0.25, -0.25, -0.5, -0.25, 0.25,
                                            -0.5, -0.25, -0.25, 0.5, -0.25, -0.25, 0.5, -0.25, 0.25, -0.5, 0.25, -0.25, -0.5, 0.25, 1, -0.25], dtype=torch.float64)

    def forward(self):
        # q0 = self.basis @ self.manifold().to(torch.complex128)
        q0 = torch.stack([self.basis0 @ self.manifold0().to(torch.complex128), self.basis1 @ self.manifold1().to(torch.complex128)], axis=1)
        lambda_aij = numqi.qec.knill_laflamme_hermite_mul(self.error_torch, q0)
        # lambda_aij = q0.T.conj() @ (self.error_torch @ q0).reshape(-1, *q0.shape)
        constraint = []
        if torch.any(self.zero_mask):
            constraint.append(lambda_aij[self.zero_mask].reshape(-1))
        tmp0 = lambda_aij[~self.zero_mask]
        constraint += [
            tmp0[:,0,1],
            tmp0[:,0,0].real-self.nonzero_value,
            tmp0[:,1,1].real-self.nonzero_value,
        ]

        q0_tensor = q0.reshape([2]*(self.num_qubit+1))
        su2 = self.manifold_su2()
        logicalU = self.contract_expr(q0_tensor, q0_tensor.conj(), *su2)
        constraint.append(logicalU - self.gateT)

        tmp0 = (lambda_aij[:,0,0].real + lambda_aij[:,1,1].real)/2
        norm2 = torch.dot(tmp0, tmp0)
        if (self.lambda_target is None) or isinstance(self.lambda_target, float):
            loss = sum([torch.vdot(x.reshape(-1), x.reshape(-1)).real for x in constraint])
            if self.lambda_target is not None:
                loss = loss + (norm2-self.lambda_target)**2
            ret = loss
        else:
            assert self.lambda_target in {'min','max'}
            loss = norm2 if (self.lambda_target=='min') else -norm2
            ret = loss, constraint
        return ret


num_qubit = 7
model = QECCn23TModel()
# theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=400, tol=1e-8, early_stop_threshold=1e-6)

model.lambda_target = 4.875
# model.lambda_target = 1/np.sqrt(11.8)
theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=40, tol=1e-10, early_stop_threshold=1e-7)
theta_optim1 = numqi.optimize.minimize(model, theta_optim.x, num_repeat=1, tol=1e-18)

# to_pickle(code=model.manifold().detach().numpy(), theta=theta_optim1.x)
# to_pickle(code0=model.manifold0().detach().numpy(), code1=model.manifold1().detach().numpy(), theta=theta_optim1.x)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        s3 = np.sqrt(3)
        code = np.zeros((2,128), dtype=np.float64)
        code[0,[5,6,24,27,33,34,37,38,45,46,53,54,69,70,88,91,97,98,101,102,109,110,117,118]] = np.array([1,1,s3,-s3,-1,-1,1,1,-1,-1,1,1,-1,-1,s3,-s3,1,1,1,1,1,1,-1,-1])/np.sqrt(32)
        code[1,[9,10,17,18,25,26,29,30,36,39,57,58,73,74,81,82,89,90,93,94,100,103,121,122]] = np.array([1,1,-1,-1,1,1,-1,-1,s3,-s3,1,1,1,1,-1,-1,-1,-1,-1,-1,-s3,s3,1,1])/np.sqrt(32)
        self.code = torch.tensor(code.T.copy().reshape([2]*8), dtype=torch.complex128)

        s3 = np.sqrt(3)
        a = np.sqrt(2*np.sqrt(3)-3)
        su_list = [
            numqi.gate.rx(5*np.pi/4),
            numqi.gate.rz(-3*np.pi/4),
            numqi.gate.rz(-5*np.pi/4),
            numqi.gate.rz(3*np.pi/4),
            numqi.gate.rz(-3*np.pi/4),
            np.array([[1-s3, -a], [-a, s3-1]]),
            np.array([[1-s3, a], [a, s3-1]]),
        ]
        self.su_list = torch.tensor(np.stack(su_list[:5]), dtype=torch.complex128)
        self.theta = torch.nn.Parameter(torch.randn(1, dtype=torch.float64))

        # self.manifold = numqi.manifold.SpecialOrthogonal(2, batch_size=7, dtype=torch.complex128)
        self.gateT = torch.tensor(numqi.gate.rz(np.pi/4), dtype=torch.complex128)
        N = 7
        tmp0 = [y for x in range(N) for y in [(2,2), (N+x,x)]]
        self.contract_expr = opt_einsum.contract_expression([2]*(N+1), list(range(N))+[2*N], [2]*(N+1), list(range(N,2*N))+[2*N+1], *tmp0, [2*N+1,2*N])

    def forward(self):
        ct = torch.cos(self.theta)
        st = torch.sin(self.theta)
        x0 = 1j*torch.stack([st,ct,ct,-st], axis=1).reshape(2,2)
        x1 = 1j*torch.stack([-st,ct,ct,st], axis=1).reshape(2,2)
        # su2 = self.manifold()
        # tmp0 = [su2[0]]*5 + [su2[1]]*2
        logicalU = self.contract_expr(self.code, self.code, *self.su_list, x0, x1)
        tmp0 = (logicalU - self.gateT).reshape(-1)
        loss = torch.vdot(tmp0, tmp0).real
        return loss

model = DummyModel()
theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=20, tol=1e-8, early_stop_threshold=1e-6)
theta_optim1 = numqi.optimize.minimize(model, theta_optim.x, num_repeat=1, tol=1e-30)
z2 = model.manifold().detach().numpy()
print(np.around(z2.reshape(-1,4), 5))
# print(np.angle(np.linalg.eigvals(z2))*180/np.pi/180)

for x in z2:
    axis,theta = get_rotation_axis(x)
    print(axis, theta*180/np.pi)

'''
[[-0.4796128   0.87748023]
 [ 0.87748023  0.4796128 ]]

[[ 0.62395709 -0.78145861]
 [-0.78145861 -0.62395709]]

[[ 0.85936642 -0.51136031]
 [-0.51136031 -0.85936642]]
'''


class DummyModel01(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # search product stabilizer operator
        s3 = np.sqrt(3)
        code = np.zeros((2,128), dtype=np.float64)
        code[0,[24,27,37,38,69,70,97,98,109,110,117,118]] = np.array([s3,-s3,1,1,1,1,-1,-1,-1,-1,1,1])/4
        code[1,[9,10,17,18,29,30,57,58,89,90,100,103]] = np.array([1,1,-1,-1,-1,-1,1,1,1,1,s3,-s3])/4
        self.code = torch.tensor(code.T.copy().reshape([2]*8), dtype=torch.complex128)
        self.manifold = numqi.manifold.SpecialOrthogonal(2, batch_size=7, dtype=torch.complex128)
        N = 7
        tmp0 = [y for x in range(N) for y in [(2,2), (N+x,x)]]
        self.contract_expr = opt_einsum.contract_expression([2]*(N+1), list(range(N))+[2*N], [2]*(N+1), list(range(N,2*N))+[2*N+1], *tmp0, [2*N+1,2*N])
        self.gateI = torch.tensor(np.eye(2), dtype=torch.complex128)

    def forward(self):
        su2 = self.manifold()
        logicalU = self.contract_expr(self.code, self.code, *su2)
        tmp0 = (logicalU - self.gateI).reshape(-1)
        loss = torch.vdot(tmp0, tmp0).real
        return loss

from tqdm import tqdm
model = DummyModel01()
for _ in tqdm(range(100)):
    theta_optim = numqi.optimize.minimize(model, ('uniform',-np.pi,np.pi), num_repeat=20, tol=1e-14, early_stop_threshold=1e-8, print_every_round=0)
    su2 = model.manifold().detach().numpy()
    tag0 = np.abs(su2 - np.eye(2)).max(axis=(1,2)) < 5e-3
    tag1 = np.abs(su2 + np.eye(2)).max(axis=(1,2)) < 5e-3
    if not np.all(np.logical_or(tag0, tag1)):
        break
print(np.around(su2.reshape(-1,4), 5))
get_rotation_axis(su2[-1])
