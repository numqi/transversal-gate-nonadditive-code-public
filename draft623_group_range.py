import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import scipy.linalg
import concurrent.futures
import torch
import numqi
import opt_einsum

from utils import get_rotation_axis, hf_kron, hfb, SpecialUnitary2XZManifold

from zzz233 import to_pickle_wrapper, from_pickle_wrapper
to_pickle = to_pickle_wrapper('623_group_range.pkl')
from_pickle = from_pickle_wrapper('623_group_range.pkl')

if torch.get_num_threads() != 1:
    torch.set_num_threads(1)


class QECC623TransversalGroupModel(torch.nn.Module):
    def __init__(self, logicalX:str, group:str='BD12'):
        super().__init__()
        assert isinstance(logicalX, str) and (len(logicalX) == 6) and (set(logicalX)<={'I','X'})
        num_qubit = 6
        error_str,error_torch = numqi.qec.make_pauli_error_list_sparse(num_qubit, distance=3, kind='torch-csr01')
        self.num_qubit = num_qubit
        self.error_torch = error_torch.clone().to(torch.complex128)
        basis0 = np.eye(2**num_qubit)
        basis1 = (numqi.qec.hf_pauli(logicalX) @ basis0.T).T
        if group.startswith('C'):
            self.manifold = numqi.manifold.Stiefel(2**num_qubit, rank=2, dtype=torch.complex128)
        else:
            self.manifold = numqi.manifold.Sphere(len(basis0), dtype=torch.complex128)
        self.basis0 = torch.tensor(basis0, dtype=torch.complex128).T.to_sparse_csr()
        self.basis1 = torch.tensor(basis1, dtype=torch.complex128).T.to_sparse_csr()
        # self.manifold_su2 = numqi.manifold.SpecialOrthogonal(2, batch_size=num_qubit, dtype=torch.complex128)
        # assert not group.startswith('C')
        if group.startswith('BD'):
            tmp0 = int(group[2:])
            assert (tmp0%2==0) #and (tmp0>=12)
            m = tmp0//2
            self.logical_gate = torch.tensor(numqi.qec.get_su2_finite_subgroup_generator(f'BD{2*m}')[1], dtype=torch.complex128)
            self.theta_su2 = torch.nn.Parameter(torch.randn(num_qubit, dtype=torch.float64))
            # self.theta_su2 = torch.tensor(np.array([2,3,3,4,4,5,7,9])*2*np.pi/19, dtype=torch.float64)
        elif group.startswith('C'):
            tmp0 = int(group[1:])
            assert (tmp0%2==0) #and (tmp0>=12)
            m = tmp0//2
            self.logical_gate = torch.tensor(numqi.qec.get_su2_finite_subgroup_generator(f'C{2*m}')[0], dtype=torch.complex128)
            self.theta_su2 = torch.nn.Parameter(torch.randn(num_qubit, dtype=torch.float64))
        elif group in {'2Ox','2I'}:
            self.logical_gate = torch.tensor(numqi.qec.get_su2_finite_subgroup_generator(group)[1], dtype=torch.complex128)
            self.manifold_su2 = SpecialUnitary2XZManifold(batch_size=num_qubit)
        elif group=='2T':
            self.logical_gate = torch.tensor(numqi.qec.get_su2_finite_subgroup_generator(group)[1], dtype=torch.complex128)
            self.manifold_su2 = numqi.manifold.SpecialOrthogonal(2, batch_size=num_qubit, dtype=torch.complex128)
        else:
            raise ValueError(f'Unsupported group {group}')
        # self.theta_phase = torch.nn.Parameter(torch.randn(1, dtype=torch.float64))
        self.lambda2_target = None

        N = num_qubit
        tmp0 = [y for x in range(N) for y in [(2,2), (N+x,x)]]
        self.contract_expr = opt_einsum.contract_expression([2]*(N+1), list(range(N))+[2*N], [2]*(N+1), list(range(N,2*N))+[2*N+1], *tmp0, [2*N+1,2*N])
        self.mask_lambda = None

    def set_lambda2_target(self, x:float|None):
        if x is None:
            self.lambda2_target = None
        else:
            self.lambda2_target = torch.tensor(float(x), dtype=torch.float64)

    def forward(self, return_info:bool=False):
        coeff = self.manifold().to(torch.complex128)
        if coeff.ndim==1:
            q0 = torch.stack([self.basis0@coeff, self.basis1@coeff], axis=1)
        else:
            q0 = coeff
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
            logicalU - self.logical_gate*(torch.exp(-1j*self.theta_phase) if hasattr(self, 'theta_phase') else 1),
        ]
        if self.lambda2_target is not None:
            tmp0 = (lambda_aij[:,0,0] + lambda_aij[:,1,1]).real/2
            norm2 = torch.dot(tmp0, tmp0)
            constraint.append(norm2 - self.lambda2_target)
        if return_info:
            info = dict(lambda_aij=lambda_aij.detach().numpy(), q0=q0.detach().numpy(), logicalU=logicalU.detach().numpy(), su2=su2.detach().numpy(), coeff=coeff.detach().numpy())
        loss = sum([torch.vdot(x.reshape(-1), x.reshape(-1)).real for x in constraint])
        ret = (loss,info) if return_info else loss
        return ret


def hf_task(lambda2_list:np.ndarray, logicalX:str, trans_group:str, kwargs0:dict, kwargs1:dict):
    model = QECC623TransversalGroupModel(logicalX, group=trans_group)
    z0 = []
    for lambda2_i in lambda2_list:
        model.set_lambda2_target(lambda2_i)
        theta_optim = numqi.optimize.minimize(model, **kwargs0)
        if theta_optim.fun < 1e-5:
            theta_optim = numqi.optimize.minimize(model, theta_optim.x, **kwargs1)
        z0.append(theta_optim)
    ret = {'lambda2':lambda2_list, 'theta_optim':z0, 'logicalX':logicalX, 'group':trans_group}
    return ret


if __name__=='__main__':
    # group_list = ['2I', '2Ox'] + [f'BD{x}' for x in range(12,38,2)]
    # logicalX_list = ['X'*x+'I'*(7-x) for x in range(1,8)][::-1]
    # group_list = 'BD24 BD26 BD30 BD32 BD34 BD36'.split(' ')
    group_list = 'C4 C6 BD4 C10'.split(' ')
    logicalX_list = ['XXXXXX']
    lambda2_list = np.linspace(0.5, 1.1, 101)
    kwargs0 = dict(theta0='uniform', num_repeat=100, tol=1e-8, early_stop_threshold=1e-5, print_freq=0, print_every_round=0)
    kwargs1 = dict(num_repeat=1, tol=1e-20, print_freq=0, print_every_round=0)

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        task_list = [(x,y) for x in logicalX_list for y in group_list]
        job_list = [executor.submit(hf_task, lambda2_list, x, y, kwargs0, kwargs1) for x,y in task_list]
        for job in concurrent.futures.as_completed(job_list):
            ret = job.result()
            key = f'grp{ret["group"]}_{ret["logicalX"]}'
            to_pickle(**{key:ret})
            print(f'{key} done')



def dump_plot_data():
    import pickle
    # group_list = ['2I', '2Ox'] + [f'BD{x}' for x in range(12,38,2)]
    # group_list += '2T BD6 BD8 BD10'.split(' ')
    group_list = 'C4 C6 BD4 C10'.split(' ')
    logicalX_list = ['X'*6]
    with open('623_group_range.pkl', 'rb') as fid:
        ALL_DATA = pickle.load(fid)
    z0 = dict()
    for group in group_list:
        for logicalX in logicalX_list:
            key = f'grp{group}_{logicalX}'
            if key in ALL_DATA:
                tmp0 = ALL_DATA[key]
                z0[key] = {
                    'xdata': tmp0['lambda2'],
                    'ydata': np.array([x.fun for x in tmp0['theta_optim']]),
                }
    with open('tbd233.pkl', 'wb') as fid:
        pickle.dump(z0, fid)


def hf_plot():
    group = 'C4'
    logicalX = 'XXXXXX'
    data = from_pickle(f'grp{group}_{logicalX}')

    xdata = data['lambda2']
    ydata = np.array([x.fun for x in data['theta_optim']])
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    ax.plot(xdata, ydata, '.-')
    ax.grid()
    ax.set_yscale('log')
    ax.set_xlabel(r'$\lambda^2$')
    ax.set_ylabel('loss')
    ax.set_title(f'623 {group} logicalX={logicalX}')
    fig.tight_layout()
    fig.savefig('tbd00.png')

    print(xdata[np.nonzero(ydata < 1e-5)[0].min()])
    print(xdata[np.nonzero(ydata < 1e-5)[0].max()])
    # BD12 X7 [0, 5.1]
    # BD14 X7 [0.45, 4.7]
    # BD16 X7 [0.6, 4.0] ??? wired

    # 2Ox X7 [0, 4.0]
'''
grpBD12_XXXXXXX done
grpBD14_XXXXXXX done
grpBD16_XXXXXXX done
grpBD18_XXXXXXX done
grpBD20_XXXXXXX done
grpBD22_XXXXXXX done
grpBD24_XXXXXXX done
grpBD26_XXXXXXX done
grpBD28_XXXXXXX done
grpBD30_XXXXXXX done
grp2Ox_XXXXXXX done
grpBD12_XXXXXXI done
grp2I_XXXXXXX done
grpBD14_XXXXXXI done
grp2I_XXXXXXI done
'''
