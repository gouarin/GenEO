import os
import sys
import subprocess
from textwrap import dedent

def run_simu(path, nxdomains, nydomains, n, E1, E2, nu1, nu2, stripe_nb, taueigmax, neigs, option):

    if not os.path.exists(path):
        os.mkdir(path)
    with open('options.txt', 'w') as f:
        f.writelines(dedent(f"""
        -E1 {E1}
        -E2 {E2}
        -Lx {nxdomains}
        -Ly {nydomains}
        -n {n}
        -stripe_nb {stripe_nb}
        -test_case {path}
        {option}
        -PCBNN_verbose False
        -PCBNN_GenEO_verbose False
        -PCNew_view True
        -PCBNN_view True
        -PCBNN_GenEO True
        -PCNew_GenEO True
        -PCNew_ComputeRitzApos True
        -PCBNN_GenEO_taueigmax {taueigmax}
        -PCBNN_GenEO_taueigmin 0.1
        -PCBNN_GenEO_nev {neigs}
        -ksp_Apos_ksp_converged_reason
        -ksp_Apos_ksp_rtol 1e-10
        -computeRitz True
        -global_ksp_ksp_monitor_true_residual
        -global_ksp_ksp_rtol 1e-10
        -global_ksp_ksp_converged_reason
        -global_ksp_ksp_max_it 150
        -PCNew_viewV0 False
        -PCNew_viewGenEO False
        -PCNew_viewminV0 False
        -PCNew_viewnegV0 False
        -PCBNN_viewV0 False
        -PCBNN_viewGenEO False
        -PCBNN_viewminV0 False
        -PCBNN_kscaling False
        """))

    print(f"mpiexec -np {nxdomains*nydomains} --oversubscribe python demo_AGenEO_2d.py -options_file options.txt")
    result = subprocess.run(["mpiexec", "-np", f"{nxdomains*nydomains}", "--oversubscribe", "python", "./demo_AGenEO_2d.py", "-options_file", "options.txt"], capture_output=True)
    if result.stderr:
        with open(f"{path}/stderr_execution.txt", 'w') as f:
            f.write(result.stderr.decode('utf-8'))
    with open(f"{path}/output_execution.txt", 'w') as f:
        f.write(result.stdout.decode('utf-8'))
    # os.system(f"mpiexec -np {ndomains} --oversubscribe python ./demo_AGenEO_2d.py -options_file options.txt")


case = 2

if case == 1:
    nxdomains= [3]
    nydomains= [3]
    n = [10]#, 42]

    E1 = [1e10]
    E2 = [1e6]
    neigs = 30
    taueigmax = [0, 1e-3, 1e-2, 5e-2, 1e-1, 0.2, 0.5]
    stripe_nb = [3]
elif case == 2:
    nxdomains= [4]
    nydomains= [1]
    n = [22]#, 42]

    E1 = [1e10]
    E2 = [1e6]
    neigs = 30
    taueigmax = [0, 1e-3, 1e-2, 5e-2, 1e-1, 0.2, 0.5]
    stripe_nb = [3]
elif case == 3:
    nxdomains= [4, 8]
    nydomains= [1, 1]
    E1 = [1e6, 1e8, 1e10, 1e12, 1e6,  1e6,  1e6]
    E2 = [1e6, 1e6,  1e6,  1e6, 1e8, 1e10, 1e12]
    neigs = 10
    taueigmax = [0.1]
    stripe_nb = [1, 2, 3]
    n = [22]#, 42]
elif case == 4:
    nxdomains= [4, 8]
    nydomains= [1, 1]
    E1 = [1e6, 1e8, 1e10, 1e12, 1e6,  1e6,  1e6]
    E2 = [1e6, 1e6,  1e6,  1e6, 1e8, 1e10, 1e12]
    neigs = 10
    taueigmax = [0.1]
    stripe_nb = [1, 2, 3]
    n = [22]#, 42]

nu1 = [0.3]*len(E1)
nu2 = [0.3]*len(E1)

if case < 4:
    PCNew_options = [('pcnew', "-PCNew True \n-PCNew_switchtoASM False \n-PCNew_switchtoASMpos False \n-PCNew_H2CoarseProjection True \n-PCNew_H3CoarseProjection False"),
                    ('pcbnn', "-PCNew False \n-PCBNN_switchtoASM False \n-PCBNN_CoarseProjection True"),
    ]
else:
    PCNew_options = [('pcnew', "-PCNew True \n-PCNew_switchtoASM False \n-PCNew_switchtoASMpos False \n-PCNew_H2CoarseProjection True \n-PCNew_H3CoarseProjection False"),
                     ('pcnew', "-PCNew True \n-PCNew_switchtoASM True \n-PCNew_switchtoASMpos False \n-PCNew_H2CoarseProjection True \n-PCNew_H3CoarseProjection False"),
                     ('pcnew', "-PCNew True \n-PCNew_switchtoASM False \n-PCNew_switchtoASMpos True \n-PCNew_H2CoarseProjection True \n-PCNew_H3CoarseProjection False"),
                     ('pcnew', "-PCNew True \n-PCNew_switchtoASM False \n-PCNew_switchtoASMpos True \n-PCNew_H2CoarseProjection False \n-PCNew_H3CoarseProjection False"),
                     ('pcnew', "-PCNew True \n-PCNew_switchtoASM False \n-PCNew_switchtoASMpos False \n-PCNew_H2CoarseProjection True \n-PCNew_H3CoarseProjection True"),
                     ('pcnew', "-PCNew True \n-PCNew_switchtoASM True \n-PCNew_switchtoASMpos False \n-PCNew_H2CoarseProjection True \n-PCNew_H3CoarseProjection True"),
                     ('pcnew', "-PCNew True \n-PCNew_switchtoASM False \n-PCNew_switchtoASMpos True \n-PCNew_H2CoarseProjection True \n-PCNew_H3CoarseProjection True"),
                     ('pcnew', "-PCNew True \n-PCNew_switchtoASM False \n-PCNew_switchtoASMpos True \n-PCNew_H2CoarseProjection False \n-PCNew_H3CoarseProjection True"),
                     ('pcbnn', "-PCNew False \n-PCBNN_switchtoASM True \n-PCBNN_CoarseProjection True"),
                     ('pcbnn', "-PCNew False \n-PCBNN_switchtoASM True \n-PCBNN_CoarseProjection False"),
                     ('pcbnn', "-PCNew False \n-PCBNN_switchtoASM False \n-PCBNN_CoarseProjection True"),
    ]

if not os.path.exists('output.d'):
    os.mkdir('output.d')

id = 0
for nn in n:
    for nx, ny in zip(nxdomains, nydomains):
        for io, (name, option) in enumerate(PCNew_options):
            for s in stripe_nb:
                for tau in taueigmax:
                    for i in range(len(E1)):
                        path = f'output.d/case_{case}_{name}_ ' + '_'.join([str(el) for el in [nx, ny, nn, E1[i], E2[i], nu1[i], nu2[i], s, tau]])
                        run_simu(path, nx, ny, nn, E1[i], E2[i], nu1[i], nu2[i], s, tau, neigs, option)
                        id += 1
