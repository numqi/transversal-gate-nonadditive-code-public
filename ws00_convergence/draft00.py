import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import multiprocessing
import concurrent.futures
import numpy as np
import torch
import numqi
import opt_einsum
import matplotlib.pyplot as plt


from zzz233 import to_pickle_wrapper, from_pickle_wrapper
to_pickle = to_pickle_wrapper('623_group_range01.pkl')
from_pickle = from_pickle_wrapper('623_group_range01.pkl')

if torch.get_num_threads()!=1:
    torch.set_num_threads(1)
if torch.get_num_interop_threads()!=1:
    torch.set_num_interop_threads(1)


class QECC623TransversalGroupModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        group:str = 'C4'
        num_qubit = 6
        error_str,error_torch = numqi.qec.make_pauli_error_list_sparse(num_qubit, distance=3, kind='torch-csr01')
        self.num_qubit = num_qubit
        self.error_torch = error_torch.clone().to(torch.complex128)
        self.manifold = numqi.manifold.Stiefel(2**num_qubit, rank=2, dtype=torch.complex128)
        tmp0 = int(group[1:])
        assert (tmp0%2==0) #and (tmp0>=12)
        m = tmp0//2
        self.logical_gate = torch.tensor(numqi.qec.get_su2_finite_subgroup_generator(f'C{2*m}')[0], dtype=torch.complex128)
        self.manifold_su2 = numqi.manifold.SpecialOrthogonal(2, batch_size=num_qubit, dtype=torch.complex128)
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
        q0 = coeff
        lambda_aij = numqi.qec.knill_laflamme_hermite_mul(self.error_torch, q0)
        su2 = self.manifold_su2()
        tmp1 = q0.reshape([2]*(self.num_qubit+1))
        logicalU = self.contract_expr(tmp1, tmp1.conj(), *su2)
        constraint = [
            torch.vdot(q0[:,0], q0[:,1]),
            lambda_aij[:,0,1],
            lambda_aij[:,0,0].real-lambda_aij[:,1,1].real,
            logicalU - self.logical_gate,
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


def collect_data_hf0(lambda2:float):
    model = QECC623TransversalGroupModel()
    kwargs0 = dict(num_repeat=1, tol=1e-8, early_stop_threshold=1e-5, print_freq=0, print_every_round=0)
    kwargs1 = dict(num_repeat=1, tol=1e-20, print_freq=0, print_every_round=0)
    model.set_lambda2_target(lambda2)
    np_rng = np.random.default_rng()
    num_parameter = numqi.optimize.get_model_flat_parameter(model).size
    z0 = None
    for IND0 in range(100):
        x0 = np_rng.uniform(-1, 1, size=num_parameter)
        theta_optim = numqi.optimize.minimize(model, theta0=x0, **kwargs0)
        if (z0 is None) or (theta_optim.fun<z0[1].fun):
            print(IND0, theta_optim.fun)
            z0 = (x0,theta_optim)
    callback = numqi.optimize.MinimizeCallback(print_freq=1, tag_print=False)
    theta_optim = numqi.optimize.minimize(model, theta0=z0[0], **kwargs1, callback=callback)
    return callback.state['fval']

def collect_convergence_data():
    lambda2_list = [0.65, 0.66, 2/3, 0.67]
    z0 = []
    for x in lambda2_list:
        z0.append(collect_data_hf0(x))
        print(x, z0[-1][-1])
    convergence = dict(lambda2_list=lambda2_list, z0=z0)
    to_pickle(convergence=convergence)

def plot_convergence_data():
    tmp0 = from_pickle('convergence')
    lambda2_list = tmp0['lambda2_list']
    z0 = tmp0['z0']

    fig,ax = plt.subplots()
    label_list = [r'$\lambda^2=0.65$', r'$\lambda^2=0.66$', r'$\lambda^2=2/3$', r'$\lambda^2=0.67$']
    for i0 in range(len(label_list)):
        x0 = z0[i0]
        ax.plot(np.arange(len(x0))+1, x0, label=label_list[i0])
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, 3000)
    ax.grid()
    ax.set_xlabel('step')
    ax.set_ylabel('loss')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def hf_task(lambda2):
    kwargs0 = dict(theta0='uniform', num_repeat=100, tol=1e-8, early_stop_threshold=1e-5, print_freq=0, print_every_round=0)
    kwargs1 = dict(num_repeat=1, tol=1e-20, print_freq=0, print_every_round=0)
    model = QECC623TransversalGroupModel()
    model.set_lambda2_target(lambda2)
    theta_optim = numqi.optimize.minimize(model, **kwargs0)
    if theta_optim.fun < 1e-5:
        theta_optim = numqi.optimize.minimize(model, theta_optim.x, **kwargs1)
    return {'lambda2':lambda2, 'theta_optim':theta_optim}


def generate_C4_range_data():
    multiprocessing.set_start_method('spawn')
    lambda2_list = np.linspace(0.5, 1.1, 101).tolist()
    z0 = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
        job_list = [executor.submit(hf_task, x) for x in lambda2_list]
        for job in concurrent.futures.as_completed(job_list):
            tmp0 = job.result()
            z0.append(tmp0)
            print(tmp0['lambda2'], tmp0['theta_optim'].fun)
    z0 = sorted(z0, key=lambda x:x['lambda2'])
    z0 = [x['theta_optim'] for x in z0]
    tmp0 = dict(lambda2_list=lambda2_list, z0=z0)
    to_pickle(c4_range=tmp0)

def plot_C4_range():
    tmp0 = from_pickle('c4_range')
    z0 = tmp0['z0']
    lambda2_list = tmp0['lambda2_list']

    xdata = np.array(lambda2_list)
    ydata = np.array([x.fun for x in z0])
    fig,ax = plt.subplots()
    ax.axvline(2/3, color='orange', linestyle='dashed', linewidth=2)
    ax.text(0.68, 3e-11, r'$\lambda^2=2/3$', color='orange')
    ax.plot(xdata, ydata, '.-')
    ax.grid()
    ax.set_yscale('log')
    ax.set_ylim(3e-17, 1e-1)
    ax.set_xlim(0.5, 1.1)
    ax.set_xlabel(r'$\lambda^2$')
    ax.set_ylabel('loss')
    fig.tight_layout()
    fig.savefig('tbd00.png')


def plot_in_one_figure():
    # Nature Portfolio/npj production size: 183 mm (double column).  The
    # explicit style keeps the exported figure independent of user rc files.
    figure_style = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.linewidth': 1.0,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'legend.fontsize': 7,
        'lines.linewidth': 1.25,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'mathtext.fontset': 'dejavusans',
    }

    # Color-blind-safe colors are paired with distinct line styles so that
    # all four traces also remain identifiable in grayscale.
    series_style = [
        dict(color='#666666', linestyle=(0, (1.5, 1.5))),
        dict(color='#56B4E9', linestyle=(0, (4.0, 1.8))),
        dict(color='#0072B2', linestyle='-', linewidth=1.6),
        dict(color='#D55E00', linestyle=(0, (5.0, 1.5, 1.2, 1.5))),
    ]
    label_list = [
        r'$\lambda^2 = 0.65$',
        r'$\lambda^2 = 0.66$',
        r'$\lambda^2 = 2/3$',
        r'$\lambda^2 = 0.67$',
    ]

    with plt.rc_context(figure_style):
        fig, (ax0, ax1) = plt.subplots(
            1, 2, figsize=(183/25.4, 82/25.4), sharey=True,
        )

        z0 = from_pickle('convergence')['z0']
        for values, label, style in zip(z0, label_list, series_style):
            ax0.plot(np.arange(len(values)) + 1, values,
                     label=label, **style)
        ax0.set_xscale('log')
        ax0.set_yscale('log')
        ax0.set_xlim(1, 3000)
        ax0.set_ylim(3e-17, 3e1)
        ax0.set_xlabel('Optimization step')
        ax0.set_ylabel('Optimization loss')
        ax0.legend(
            loc='lower left', ncol=1, frameon=False,
            handlelength=2.8, handletextpad=0.6, labelspacing=0.35,
        )

        range_data = from_pickle('c4_range')
        xdata = np.asarray(range_data['lambda2_list'])
        ydata = np.asarray([result.fun for result in range_data['z0']])
        ax1.plot(
            xdata, ydata, color='#0072B2', marker='o', markersize=2.3,
            markeredgewidth=0, linewidth=1.25,
        )
        ax1.axvline(
            2/3, color='#D55E00', linestyle=(0, (4.0, 2.0)),
            linewidth=1.25, zorder=3,
        )
        ax1.annotate(
            r'$\lambda^2 = 2/3$', xy=(2/3, 2e-9), xytext=(0.695, 2e-9),
            ha='left', va='center', color='#222222',
            arrowprops=dict(
                arrowstyle='-', color='#444444', linewidth=1.0,
                shrinkA=2, shrinkB=2,
            ),
        )
        ax1.set_xlim(0.5, 1.1)
        ax1.set_xlabel(r'$\lambda^2$')
        ax1.tick_params(axis='y', which='both', labelleft=False)

        for panel_label, ax in zip(('a', 'b'), (ax0, ax1)):
            ax.text(
                -0.15, 1.035, panel_label, transform=ax.transAxes,
                fontsize=8, fontweight='bold', ha='left', va='bottom',
            )
            ax.grid(axis='y', which='major', color='#E0E0E0',
                    linewidth=1.0, zorder=0)
            ax.tick_params(which='both', direction='out', top=False,
                           right=False, length=3)
            ax.tick_params(which='minor', length=1.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig.subplots_adjust(
            left=0.090, right=0.980, bottom=0.165, top=0.925, wspace=0.16,
        )
        fig.savefig('tbd00.png', dpi=600, facecolor='white')
        fig.savefig('convergence.pdf', facecolor='white')
        fig.savefig('convergence.svg', facecolor='white')
        plt.close(fig)


if __name__=='__main__':
    # generate_C4_range_data()
    # plot_C4_range()
    plot_in_one_figure()
