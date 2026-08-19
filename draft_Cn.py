import numpy as np
import torch
import opt_einsum

import numqi


if torch.get_num_threads() != 1:
    torch.set_num_threads(1)

class QECCn23C2mModel(torch.nn.Module):
    def __init__(self, veca, C2m_m:int, ind0L, ind1L, tag_real:bool=False, sign:str='+'):
        super().__init__()
        assert sign in "+-"
        num_qubit = len(veca)
        error_str,error_torch = numqi.qec.make_pauli_error_list_sparse(num_qubit, distance=3, kind='torch-csr01')
        self.num_qubit = num_qubit
        self.error_torch = error_torch.clone().to(torch.complex128)
        basis0 = np.zeros((len(ind0L), 2**num_qubit), dtype=np.complex128)
        basis0[np.arange(len(ind0L)), ind0L] = 1
        assert np.abs(basis0.conj() @ basis0.T - np.eye(len(basis0))).max() < 1e-12
        basis1 = np.zeros((len(ind1L), 2**num_qubit), dtype=np.complex128)
        basis1[np.arange(len(ind1L)), ind1L] = 1
        assert np.abs(basis1.conj() @ basis1.T - np.eye(len(basis0))).max() < 1e-12
        self.manifold0 = numqi.manifold.Sphere(len(basis0), dtype=(torch.float64 if tag_real else torch.complex128))
        self.manifold1 = numqi.manifold.Sphere(len(basis1), dtype=(torch.float64 if tag_real else torch.complex128))
        self.basis0 = torch.tensor(basis0, dtype=torch.complex128).T.to_sparse_csr()
        self.basis1 = torch.tensor(basis1, dtype=torch.complex128).T.to_sparse_csr()
        tmp0 = 1 if (sign == '+') else -1
        self.logical_gate = torch.tensor(tmp0*numqi.gate.rz(-2*np.pi/C2m_m), dtype=torch.complex128)
        # self.theta_su2 = torch.nn.Parameter(torch.randn(num_qubit, dtype=torch.float64))
        self.theta_su2 = torch.tensor(np.array(veca)*2*np.pi/C2m_m, dtype=torch.float64)
        self.C2m_m = C2m_m

        N = num_qubit
        tmp0 = [y for x in range(N) for y in [(2,2), (N+x,x)]]
        self.contract_expr = opt_einsum.contract_expression([2]*(N+1), list(range(N))+[2*N], [2]*(N+1), list(range(N,2*N))+[2*N+1], *tmp0, [2*N+1,2*N])
        self.mask_lambda = None

    def forward(self, return_info:bool=False):
        coeff0 = self.manifold0().to(torch.complex128)
        coeff1 = self.manifold1().to(torch.complex128)
        # coeff = torch.concat([coeff[:2]*1j, coeff[2:]])
        # coeff = torch.concat([coeff[:2]*1j, coeff[4:]*0], axis=0) + coeff[2:]
        q0 = torch.stack([self.basis0@coeff0, self.basis1@coeff1], axis=1)
        lambda_aij = numqi.qec.knill_laflamme_hermite_mul(self.error_torch, q0)
        if hasattr(self, 'manifold_su2'):
            assert not hasattr(self, 'theta_su2')
            su2 = self.manifold_su2()
        elif hasattr(self, 'theta_su2'):
            su2 = numqi.gate.rz(self.theta_su2)
        else:
            su2 = self.su2
        tmp1 = q0.reshape([2]*(self.num_qubit+1))
        logicalU = self.contract_expr(tmp1, tmp1.conj(), *su2)
        constraint = [
            torch.vdot(q0[:,0], q0[:,1]),
            lambda_aij[:,0,1],
            lambda_aij[:,0,0].real-lambda_aij[:,1,1].real,
            logicalU - self.logical_gate,
        ]
        if return_info:
            info = dict(lambda_aij=lambda_aij.detach().numpy(), q0=q0.detach().numpy(), logicalU=logicalU.detach().numpy(),
                        su2=su2.detach().numpy(), coeff0=coeff0.detach().numpy(), coeff1=coeff1.detach().numpy())
        loss = sum([torch.vdot(x.reshape(-1), x.reshape(-1)).real for x in constraint])
        ret = (loss,info) if return_info else loss
        return ret

'''
823 C84
(3, 4, 8, 10, 20, 24, 27, 29) -
(3, 4, 10, 13, 15, 22, 24, 34) -
(3, 4, 18, 20, 27, 29, 32, 34) +
(3, 8, 10, 13, 15, 18, 20, 38) -
(3, 22, 24, 27, 29, 32, 34, 38) -
(4, 8, 13, 20, 24, 27, 32, 39) +
(4, 8, 15, 18, 22, 29, 32, 39) +
(4, 10, 13, 18, 22, 27, 34, 39) +
'''

# Q823_C84_list = [
#     pass
# ]

num_qubit = 8
C2m_m = 42

veca = np.array([3, 4, 18, 20, 27, 29, 32, 34])
ind0L,ind1L = numqi.qec.get_C2m_submultiset(C2m_m, veca)

model = QECCn23C2mModel(C2m_m=C2m_m, veca=veca, ind0L=ind0L, ind1L=ind1L, sign='+')
early_stop_threshold = 1e-7
theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=100, tol=1e-10,
                        early_stop_threshold=early_stop_threshold, print_freq=0)
if theta_optim.fun < early_stop_threshold:
    theta_optim = numqi.optimize.minimize(model, theta_optim.x, num_repeat=1, tol=1e-20, print_freq=500)

if theta_optim.fun < 1e-10:
    coeff0 = model.manifold0().detach().numpy().copy()
    coeff1 = model.manifold1().detach().numpy().copy()
    info = model(return_info=True)[1]
    code = info['q0'].T
    if np.abs(coeff0.imag).max() > 1e-10:
        tmp0 = np.exp(-1j*np.angle(coeff0[np.argmax(np.abs(coeff0))]))
        coeff0 = coeff0 * tmp0
        coeff1 = coeff1 * tmp0
    if np.abs(code.imag).max() < 1e-10:
        code = code.real
    else:
        print(np.around(coeff0, 4))
        code[0] *= np.exp(-1j*np.angle(code[0,0]))
    hf0a = lambda x,eps: np.nonzero(np.abs(code[0]-x)<eps)[0].tolist()
    hf0 = lambda x,eps=1e-3: (hf0a(x,eps), hf0a(-x,eps),hf0a(1j*x,eps),hf0a(-1j*x,eps))
    print(np.sort(np.abs(coeff0)))

    # qweA,qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
    # print(qweA, qweB, sep='\n')
    print((np.abs(coeff0)**2*model.C2m_m*2))


# basis0 = model.basis0.to_dense().T.numpy()
# basis1 = model.basis1.to_dense().T.numpy()
# N = numqi.utils.hf_num_state_to_num_qubit(basis0.shape[1])
# error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(N, distance=3, kind='scipy-csr01')
# N0 = len(basis0)
# tmp0 = basis0.conj() @ (error_scipy @ basis1.T).reshape(-1,2**N,N0)
# tmp1 = basis0.conj() @ (error_scipy @ basis0.T).reshape(-1,2**N,N0) - basis1.conj() @ (error_scipy @ basis1.T).reshape(-1,2**N,N0)
# z0 = np.concatenate([tmp0, tmp1], axis=0)
# # z0 = (z0 + z0.transpose(0,2,1)) # real coefficient only
# z1 = numqi.qec.pick_indenpendent_vector(z0.reshape(-1,N0*N0), tag_pure_imag=True).reshape(-1,N0,N0)
# z1[np.abs(z1)<1e-10] = 0
# assert np.abs(z1 - np.around(z1).astype(np.int64)).max() < 1e-10
# z1 = np.around(z1).astype(np.int64)
# x0 = np.diagonal(z1, axis1=1, axis2=2)

zc0 = np.angle(coeff0)/np.pi*2
print(np.around(zc0.reshape(-1,1) - zc0, 3))
