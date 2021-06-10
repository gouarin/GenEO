import os
import sys
import subprocess
from textwrap import dedent

def run_simu(path, nxdomains, nydomains, n, E1, E2, nu1, nu2, stripe_nb, taueigmax, Aposrtol, neigs, option):

    if not os.path.exists(path):
        os.mkdir(path)
    with open('options.txt', 'w') as f:
        f.writelines(dedent(f"""
        -E1 {E1}
        -E2 {E2}
        -nu1 {nu1}
        -nu2 {nu2}
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
        -ksp_Apos_ksp_rtol {Aposrtol}
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


case = 7

if case == 1:
    error('dont erase case 1')
    nxdomains= [3]
    nydomains= [3]
    n = [21]#, 42]
    Aposrtol = [1e-10]
    E1 = [1e11]
    E2 = [1e7]
    neigs = 30
    taueigmax = [0, 1e-3, 1e-2, 5e-2, 1e-1, 0.2, 0.5]
    stripe_nb = [3]
    nu1 = [0.3]*len(E1)
    nu2 = [0.3]*len(E1)
elif case == 2:
    error('dont erase case 2')
    nxdomains= [3]
    nydomains= [3]
    n = [21]#, 42]
    Aposrtol = [1e-10]
    E1 = [1e11]
    E2 = [1e11]
    neigs = 30
    taueigmax = [5e-2]
    stripe_nb = [0]
    nu1 = [0.2, 0.3, 0.35, 0.4, 0.45, 0.49]
    nu2 = nu1 
    E1 = [1e11]*len(nu1)
    E2 = [1e11]*len(nu2)
elif case == 3:
    error('dont erase case 3')
    nxdomains= [8]
    nydomains= [1]
    Aposrtol = [1e-10]
    E1 = [1e11, 1e11, 1e11, 1e11, 1e9,  1e7,  1e5]
    E2 = [1e5, 1e7,  1e9,  1e11, 1e11, 1e11, 1e11]
    neigs = 30
    taueigmax = [0.1]
    stripe_nb = [3]
    n = [21]#, 42]
    nu1 = [0.3]*len(E1)
    nu2 = [0.3]*len(E1)
elif case == 4: #options takes several values (see below)
    error('dont erase case 4')
    nxdomains= [3]
    nydomains= [3]
    E1 = [1e11]
    E2 = [1e7]
    Aposrtol = [1e-10]
    neigs = 20
    taueigmax = [0.1]
    stripe_nb = [3]
    n = [21]#, 42]
    nu1 = [0.3]*len(E1)
    nu2 = [0.3]*len(E1)
elif case == 5: #options takes several values (see below)
    nxdomains= [3]
    nydomains= [3]
    E1 = [1e11]
    E2 = [1e7]
    Aposrtol = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    neigs = 20
    taueigmax = [0.1]
    stripe_nb = [3]
    n = [21]#, 42]
    nu1 = [0.3]*len(E1)
    nu2 = [0.3]*len(E1)
elif case == 6:
    nxdomains= [3]
    nydomains= [3]
    Aposrtol = [1e-10]
    E1 = [1e11]
    E2 = [1e7]
    neigs = 30
    taueigmax = [0.1]
    stripe_nb = [0, 1, 2, 3, 4] #4 will set E = E2
    n = [21]#, 42]
    nu1 = [0.3]*len(E1)
    nu2 = [0.3]*len(E1)
elif case == 7:
    error('dont erase case 7')
    nxdomains= [2,4,8,16,32,64]
    #nxdomains= [32,64]
    nydomains= [1]*len(nxdomains)
    Aposrtol = [1e-10]
    E1 = [1e11]
    E2 = [1e7]
    neigs = 30
    taueigmax = [0.1]
    stripe_nb = [3] #4 will set E = E2
    n = [14]#, 42]
    nu1 = [0.3]*len(E1)
    nu2 = [0.3]*len(E1)


if case != 4:
    PCNew_options = [('pcnew', "-PCNew True \n-PCNew_switchtoASM False \n-PCNew_switchtoASMpos False \n-PCNew_H2CoarseProjection True \n-PCNew_H3CoarseProjection False"),
                    ('pcbnn', "-PCNew False \n-PCBNN_switchtoASM False \n-PCBNN_CoarseProjection True"),
    ]
else:
    PCNew_options = [('pcnew_NNhyb_ad', "-PCNew True \n-PCNew_switchtoASM False \n-PCNew_switchtoASMpos False \n-PCNew_H2CoarseProjection True \n-PCNew_H3CoarseProjection False"),
                     ('pcnew_AShyb_ad', "-PCNew True \n-PCNew_switchtoASM True \n-PCNew_switchtoASMpos False \n-PCNew_H2CoarseProjection True \n-PCNew_H3CoarseProjection False"),
                     ('pcnew_ASposhyb_ad', "-PCNew True \n-PCNew_switchtoASM False \n-PCNew_switchtoASMpos True \n-PCNew_H2CoarseProjection True \n-PCNew_H3CoarseProjection False"),
                     ('pcnew_ASposad_ad', "-PCNew True \n-PCNew_switchtoASM False \n-PCNew_switchtoASMpos True \n-PCNew_H2CoarseProjection False \n-PCNew_H3CoarseProjection False"),
                     ('pcnew_NNhyb_hyb', "-PCNew True \n-PCNew_switchtoASM False \n-PCNew_switchtoASMpos False \n-PCNew_H2CoarseProjection True \n-PCNew_H3CoarseProjection True"),
                     ('pcnew_AShyb_hyb', "-PCNew True \n-PCNew_switchtoASM True \n-PCNew_switchtoASMpos False \n-PCNew_H2CoarseProjection True \n-PCNew_H3CoarseProjection True"),
                     ('pcnew_ASposhyb_hyb', "-PCNew True \n-PCNew_switchtoASM False \n-PCNew_switchtoASMpos True \n-PCNew_H2CoarseProjection True \n-PCNew_H3CoarseProjection True"),
                     ('pcnew_ASposad_hyb', "-PCNew True \n-PCNew_switchtoASM False \n-PCNew_switchtoASMpos True \n-PCNew_H2CoarseProjection False \n-PCNew_H3CoarseProjection True"),
                     ('pcbnn_AShyb', "-PCNew False \n-PCBNN_switchtoASM True \n-PCBNN_CoarseProjection True"),
                     ('pcbnn_ASad', "-PCNew False \n-PCBNN_switchtoASM True \n-PCBNN_CoarseProjection False"),
                     ('pcbnn_NNhyb', "-PCNew False \n-PCBNN_switchtoASM False \n-PCBNN_CoarseProjection True"),
                     ('pcbnn_AS_onelevel', "-PCNew False \n-PCBNN_switchtoASM True \n-PCBNN_CoarseProjection False \n  PCBNN_addCoarseSolve False "),
    ]

if not os.path.exists('output.d'):
    os.mkdir('output.d')

id = 0
for nn in n:
    for nx, ny in zip(nxdomains, nydomains):
        for io, (name, option) in enumerate(PCNew_options):
            for s in stripe_nb:
                for tau in taueigmax:
                    for rtol in Aposrtol:
                        for i in range(len(E1)):
                            path = f'output.d/case_{case}_{name}_' + '_'.join([str(el) for el in [nx, ny, nn, E1[i], E2[i], nu1[i], nu2[i], s, tau, rtol]])
                            run_simu(path, nx, ny, nn, E1[i], E2[i], nu1[i], nu2[i], s, tau, rtol, neigs, option)
                            id += 1
