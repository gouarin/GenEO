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

def get_cell_value(y, E1, E2, stripe_nb,Ly):
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

def plot_coarse_vec(path, data, E1, E2, stripe_nb,Ly):
    scale = 0.5
    for k, v in data.items():
        ncol = int(np.ceil(np.sqrt(len(v))))

        p = pv.Plotter(shape=(ncol, ncol), off_screen=True)

        with open(os.path.join(path, f'properties_{k}.txt')) as json_file:
            prop = json.load(json_file)

        iplot = 0
        for i in range(ncol):
            for j in range(ncol):
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
                    p.add_mesh(grid, show_edges=True)
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
regexp_case_4 = re.compile('case_4_*')

path = 'output.d'

fig, ax = plt.subplots()

condition = {
    'pcbnn': [],
    'pcnew': [],
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
            data = {'name': name,
                    'kappa': results['kappa'][0],
                    'labdamin': results['lambdamin'][0],
                    'lambdamax': results['lambdamax'][0],
                    'V0dim': int(results['V0dim']),
                    # 'vneg': results['vneg'],
                    'sum_gathered_nneg': int(results['sum_gathered_nneg']),
            }
            dfs.append(data)
        # plot_coarse_vec(dirpath, data, results['E1'], results['E2'], results['stripe_nb'],results['Ly'])
        # plot_eigenvalues(dirpath, results['GenEOV0_gathered_Lambdasharp'])

        # ax.plot(results['precresiduals'])

        if os.path.basename(dirpath).split('_')[1] == '1':
            if os.path.basename(dirpath).split('_')[2] == 'pcbnn':
                condition['pcbnn'].append((results['taueigmax'], results['V0dim'], results['kappa'][0]))
            else:
                condition['pcnew'].append((results['taueigmax'], results['V0dim'], results['kappa'][0]))
                # condition['pcnew_neg'].append((results['taueigmax'], results['sum_gathered_nneg'], results['kappa'][0]))
        #if os.path.basename(dirpath).split('_')[1] == '5':
        #    if os.path.basename(dirpath).split('_')[2] == 'pcbnn':
        #        condition5['pcbnn'].append((results['Aposrtol'], results['V0dim'], results['kappa'][0]))
        #    else:
        #        condition5['pcnew'].append((results['taueigmax'], results['V0dim'], results['kappa'][0]))


ax.set_xlabel('iteration number')
ax.set_ylabel('residual')
ax.set_yscale('log')
fig.savefig(os.path.join(path, 'residuals.png'), dpi=300)

plot_condition_number(path, condition)
#plot_condition_number(path, condition5)
df = pd.DataFrame(dfs)
print(df.to_latex())
