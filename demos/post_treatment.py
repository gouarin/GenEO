import pyvista as pv
import numpy as np
import h5py
import os
import re
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from io import BytesIO
import pandas as pd

pv.set_plot_theme("document")

def get_cell_value(y, E1, E2, stripe_nb,Ly):
    stripe_nb = 2
    if stripe_nb == 0:
        return E2
    elif stripe_nb == 1:
        if 1./7 <= y - np.floor(y) <= 2./7:
            return E1
        else:
            return E2
    elif stripe_nb == 2:
        if (1./7 <= y - np.floor(y) <= 2./7) or (3./7 <= y  - np.floor(y) <= 4./7):
            return E1
        else:
            return E2
    elif stripe_nb == 3:
        if (1./7 <= y  - np.floor(y) <= 2./7) or (3./7 <= y  - np.floor(y) <= 4./7) or (5./7 <= y  - np.floor(y) <= 6./7):
            return E1
        else:
            return E2

def plot_solution(path, data, E1, E2, stripe_nb, Ly):
    scale = 100000

    p = pv.Plotter(off_screen=True)

    filename = os.path.join(path, 'solution_2d.vts')
    if os.path.exists(filename):
        grid = pv.read(os.path.join(path, 'solution_2d.vts'))
        E = np.zeros((grid.n_cells))
        for ic in range(grid.n_cells):
            cell = grid.cell_points(ic)
            center_y = .5*(cell[0, 1] + cell[2, 1])
            E[ic] = get_cell_value(center_y, E1, E2, stripe_nb,Ly)

        grid.cell_arrays.update({'E': E})
        disp = grid.point_arrays['Unnamed']

        print(grid.x.shape, disp.shape)
        grid.x.T.flat[:] += scale*disp[:, 0]
        grid.y.T.flat[:] += scale*disp[:, 1]

        p.add_mesh(grid, scalars='E', show_edges=True, show_scalar_bar=False, n_colors=2, edge_color='#464646', cmap=['#dbd9d3', '#a2acbd'])
        print(os.path.join(path, 'solution.png'))
        p.show(cpos='xy', screenshot=os.path.join(path, 'solution.png'))

def plot_coarse_vec(path, data, E1, E2, stripe_nb,Ly):
    scale = 0.5
    for k, v in data.items():
        print(k, v)
        ncol = int(np.ceil(np.sqrt(len(v))))

        p = pv.Plotter(shape=(ncol, ncol), off_screen=True)

        with open(os.path.join(path, f'properties_{k}.txt')) as json_file:
            prop = json.load(json_file)

        iplot = 0
        for i in range(ncol):
            for j in range(ncol):
                p_individual = pv.Plotter(off_screen=True)
                if iplot < len(v):
                    h5 = h5py.File(os.path.join(path, v[iplot]))
                    coords = h5['coordinates']
                    coarse_vec = h5['coarse_vec']

                    nx = 0
                    dim = len(coords[:].shape)
                    if dim == 1:
                        coords = coords[:].copy()
                        coords.shape = (coords.size//2, 2)

                    xx = coords[:, 0]
                    while xx[nx] < xx[nx + 1]:
                        nx += 1
                    nx += 1
                    ny = coords.size//2//nx

                    x = np.zeros((ny, nx))
                    x.flat = coords[:, 0]  #+ scale*coarse_vec[::2]
                    y = np.zeros((ny, nx))
                    y.flat = coords[:, 1] #+ scale*coarse_vec[1::2]

                    scalar = np.zeros((ny, nx))
                    grid = pv.StructuredGrid(x, y, np.zeros((ny, nx)))

                    E = np.zeros((grid.n_cells))
                    for ic in range(grid.n_cells):
                        cell = grid.cell_points(ic)
                        center_y = .5*(cell[0, 1] + cell[1, 1])
                        E[ic] = get_cell_value(center_y, E1, E2, stripe_nb,Ly)

                    grid.cell_arrays.update({'E': E})

                    if dim == 1:
                        coarse_vec = coarse_vec[:].copy()
                        coarse_vec.shape = (coarse_vec.size//2, 2)

                    grid.x[:] += scale*np.reshape(coarse_vec[:, 0], (ny, nx, 1))
                    grid.y[:] += scale*np.reshape(coarse_vec[:, 1], (ny, nx, 1))

                    p.subplot(i, j)
                    p.add_text(f"{prop['eigs'][iplot]}", position='upper_edge', font_size=4)
                    p.add_mesh(grid, show_edges=True, show_scalar_bar=False, n_colors=2, edge_color='#464646', cmap=['#dbd9d3', '#a2acbd'])

                    p_individual.add_mesh(grid, show_edges=True, show_scalar_bar=False, n_colors=2, edge_color='#464646', cmap=['#dbd9d3', '#a2acbd'])
                    p_individual.show(cpos='xy', window_size=[800, 800], screenshot=os.path.join(path, f'coarse_vec_{k}_{iplot}.png'))
                    iplot += 1

        p.show(cpos='xy', screenshot=os.path.join(path, f'coarse_vec_{k}.png'))

def plot_eigenvalues(path, eigs):
    fig, ax = plt.subplots()
    for i, eig in enumerate(eigs):
        ax.scatter(i*np.ones_like(eig), eig)
    ax.set_xlabel('Subdomain number')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel('Generalized eigenvalues (log scale)')
    ax.set_yscale('log')
    fig.savefig(os.path.join(path, 'eigenvalues.png'), dpi=300)

def plot_condition_number(path, condition):
    fig, ax = plt.subplots()

    for k, v in condition.items():
        data = np.asarray(v)
        index = np.argsort(data[:, 0])
        data = data[index]
        ax.plot(data[:, 1], data[:, 2], label=k, marker='.')

    ax.set_xlabel('Coarse space size')
    ax.set_ylabel('Condition number (log scale)')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(os.path.join(path, 'condition_number.png'), dpi=300)

regexp = re.compile('coarse_vec_(\d+)_(\d+).h5')
regexp_case_9 = re.compile('case_9_*')
regexp_case_8 = re.compile('case_8_*')
regexp_case_7 = re.compile('case_7_*')
regexp_case_6 = re.compile('case_6_*')
regexp_case_5 = re.compile('case_5_*')
regexp_case_4 = re.compile('case_4_*')
regexp_case_2 = re.compile('case_2_*')
regexp_case_3 = re.compile('case_3_*')

path = 'output.d'

fig, ax = plt.subplots()

condition = {
    'classical GenEO': [],
    'AWG': [],
    # 'pcnew_neg': []
}

dfs = []

for (dirpath, dirnames, filenames) in os.walk(path):
    if 'results.json' in filenames:
        data = {}
        for f in filenames:
            res = regexp.search(f)
            if res:
                i, rank = res.groups()
                d = data.get(rank, [])
                d.append(f)
                data[rank] = d

        with open(os.path.join(dirpath, 'results.json')) as json_file:
            results = json.load(json_file)

        res_case_4 = regexp_case_4.search(os.path.split(dirpath)[-1])

        if res_case_4:
            name = open(os.path.join(dirpath, 'name.txt')).read()
            data_case4 = {'name': name,
                    'kappa': results['kappa'][0],
                    'labdamin': results['lambdamin'][0],
                    'lambdamax': results['lambdamax'][0],
                    'V0dim': int(results['V0dim']),
                    # 'vneg': results['vneg'],
                    'sum_gathered_nneg': int(results['sum_gathered_nneg']),
            }
            dfs.append(data_case4)
        # plot_coarse_vec(dirpath, data, results['E1'], results['E2'], results['stripe_nb'],results['Ly'])
        # plot_solution(dirpath, data, results['E1'], results['E2'], results['stripe_nb'],results['Ly'])

        # plot_eigenvalues(dirpath, results['GenEOV0_gathered_Lambdasharp'])

#####CASE 2
#        res_case_2 = regexp_case_2.search(os.path.split(dirpath)[-1])
#        if res_case_2:
#            #name = open(os.path.join(dirpath, 'name.txt')).read()
#            data = {'nu': results['nu1'],
#                    'kappa': results['kappa'][0],
#                    'iter': len(results['precresiduals']),
#                    'lambdamin': results['lambdamin'][0],
#                    'lambdamax': results['lambdamax'][0],
#                    'V0dim': int(results['V0dim']),
#                    'sum_gathered_nneg': int(results['sum_gathered_nneg']),
#            }
#            dfs.append(data)
#            #plot_coarse_vec(dirpath, data, results['E1'], results['E2'], results['stripe_nb'],results['Ly'])
#            plot_eigenvalues(dirpath, results['GenEOV0_gathered_Lambdasharp'])
#####END CASE 2
#####CASE 3
        #res_case_3 = regexp_case_3.search(os.path.split(dirpath)[-1])
        #if res_case_3:
        #    name = open(os.path.join(dirpath, 'name.txt')).read()
        #    data = {' ': name,
        #            'E_1': results['E1'],
        #            'E_2': results['E2'],
        #            'kappa': results['kappa'][0],
        #            'iter': len(results['precresiduals']),
        #            'lambdamin': results['lambdamin'][0],
        #            'lambdamax': results['lambdamax'][0],
        #            'V0dim': int(results['V0dim']),
        #            'sum_gathered_nneg': int(results['sum_gathered_nneg']),
        #    }
        #    dfs.append(data)
        #    #plot_coarse_vec(dirpath, data, results['E1'], results['E2'], results['stripe_nb'],results['Ly'])
        #    plot_eigenvalues(dirpath, results['GenEOV0_gathered_Lambdasharp'])
#####END CASE 3
###### I deleted  CASE 4 by accident, it is in Loic's commit from June 10th 2021

#######CASE 1
#        if os.path.basename(dirpath).split('_')[1] == '1':
#            if os.path.basename(dirpath).split('_')[2] == 'pcbnn':
#                condition['classical GenEO'].append((results['taueigmax'], results['V0dim'], results['kappa'][0]))
#            else:
#                condition['AWG'].append((results['taueigmax'], results['V0dim'], results['kappa'][0]))
######## END CASE 1
#######CASE 9
#        if os.path.basename(dirpath).split('_')[1] == '9':
#            if os.path.basename(dirpath).split('_')[2] == 'pcbnn':
#                condition['classical GenEO'].append((results['taueigmax'], results['V0dim'], results['kappa'][0]))
#            else:
#                condition['AWG'].append((results['taueigmax'], results['V0dim'], results['kappa'][0]))
######## END CASE 9
#####CASE 5 and 8
#        res_case_5 = regexp_case_5.search(os.path.split(dirpath)[-1])
#        res_case_8 = regexp_case_8.search(os.path.split(dirpath)[-1])
#        if res_case_5 or res_case_8:
#            name = open(os.path.join(dirpath, 'name.txt')).read()
#            data = {' ': name,
#                    'nu': results['nu1'],
#                    'rtol': results['Aposrtol'],
#                    'kappa': results['kappa'][0],
#                    'iter': len(results['precresiduals']),
#                    'lambdamin': results['lambdamin'][0],
#                    'lambdamax': results['lambdamax'][0],
#                    'V0dim': int(results['V0dim']),
#                    'sum_gathered_nneg': int(results['sum_gathered_nneg']),
#            }
#            dfs.append(data)
#            #plot_coarse_vec(dirpath, data, results['E1'], results['E2'], results['stripe_nb'],results['Ly'])
#            plot_eigenvalues(dirpath, results['GenEOV0_gathered_Lambdasharp'])
#####END CASE 5 and 8
        # ax.plot(results['precresiduals'])
######CASE 6
        res_case_6 = regexp_case_6.search(os.path.split(dirpath)[-1])
        if res_case_6:
            name = open(os.path.join(dirpath, 'name.txt')).read()
            data = {' ': name,
                    'nbstripes': results['stripe_nb'],
                    'kappa': results['kappa'][0],
                    'iter': len(results['precresiduals']),
                    'lambdamin': results['lambdamin'][0],
                    'lambdamax': results['lambdamax'][0],
                    'V0dim': int(results['V0dim']),
                    'sum_gathered_nneg': int(results['sum_gathered_nneg']),
            }
            dfs.append(data)
            #plot_coarse_vec(dirpath, data, results['E1'], results['E2'], results['stripe_nb'],results['Ly'])
            # plot_eigenvalues(dirpath, results['GenEOV0_gathered_Lambdasharp'])
#######END CASE 6
#####CASE 7
        res_case_7 = regexp_case_7.search(os.path.split(dirpath)[-1])
        if res_case_7:
            name = open(os.path.join(dirpath, 'name.txt')).read()
            data = {' ': name,
                    'N': len(results['minV0_gathered_dim']),
                    'kappa': results['kappa'][0],
                    'iter': len(results['precresiduals']),
                    'lambdamin': results['lambdamin'][0],
                    'lambdamax': results['lambdamax'][0],
                    'V0dim': int(results['V0dim']),
                    'sum_gathered_nneg': int(results['sum_gathered_nneg']),
            }
            dfs.append(data)
            #plot_coarse_vec(dirpath, data, results['E1'], results['E2'], results['stripe_nb'],results['Ly'])
            plot_eigenvalues(dirpath, results['GenEOV0_gathered_Lambdasharp'])
#####END CASE 7





# ax.set_xlabel('iteration number')
# ax.set_ylabel('residual')
# ax.set_yscale('log')
# fig.savefig(os.path.join(path, 'residuals.png'), dpi=300)

#plot_condition_number(path, condition) #CASE 1 and 9
df = pd.DataFrame(dfs)
print(df.to_latex())
