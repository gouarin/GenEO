import pyvista as pv
import numpy as np
import h5py
import os
import re
import json
from matplotlib import mathtext
from io import BytesIO

path = 'output.d'
files = []
for (dirpath, dirnames, filenames) in os.walk(path):
    files.extend(filenames)
    break

data = {}
regexp = re.compile('coarse_vec_(\d+)_(\d+).h5')
for f in files:
    res = regexp.search(f)
    if res:
        i, rank = res.groups()
        d = data.get(rank, [])
        d.append(f)
        data[rank] = d

scale = 0.5
for k, v in data.items():
    ncol = int(np.ceil(np.sqrt(len(v))))

    p = pv.Plotter(shape=(ncol, ncol), off_screen=True)

    with open(os.path.join(path, f'properties_{k}.txt')) as json_file:
        prop = json.load(json_file)

    iplot = 0
    for i in range(ncol):
        for j in range(ncol):
            # if k == '1' and iplot < 1:
            if iplot < len(v):
                h5 = h5py.File(os.path.join(path, v[iplot]))
                coords = h5['coordinates']
                coarse_vec = h5['coarse_vec']
                # la = h5['lambda'][::2]
                # print(la)
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

                print(nx, ny)
                x = np.zeros((ny, nx))
                # x.flat = coords[::2]  #+ scale*coarse_vec[::2]
                x.flat = coords[:, 0]  #+ scale*coarse_vec[::2]
                y = np.zeros((ny, nx))
                # y.flat = coords[1::2] #+ scale*coarse_vec[1::2]
                y.flat = coords[:, 1] #+ scale*coarse_vec[1::2]
                # print(coords[1::2])
                # print(y)

                # test = np.zeros((ny, nx))
                # test.flat = la

                # print(test)

                scalar = np.zeros((ny, nx))
                # scalar[1, :] = 1

                grid = pv.StructuredGrid(x, y, np.zeros((ny, nx)))

                E = np.zeros((grid.n_cells))
                for ic in range(grid.n_cells):
                    cell = grid.cell_points(ic)
                    center_y = .5*(cell[0, 1] + cell[1, 1])
                    # if  .2 <= center_y <= 0.4 or .6 <= center_y <= 0.8:
                    # if  1./7 <= center_y <= 2./7:
                    #     E[ic] = prop['E1']
                    # else:
                    #     E[ic] = prop['E2']
                    E[ic] = prop['E2']

                # grid.add_field_array(la, 'lambda')
                # la.shape = (nx, ny)
                # grid.point_arrays.update({'lambda': test.T.flatten()})
                grid.cell_arrays.update({'E': E})

                # grid.x[:] += scale*np.reshape(coarse_vec[::2], (ny, nx, 1))
                # grid.y[:] += scale*np.reshape(coarse_vec[1::2], (ny, nx, 1))
                # import ipdb; ipdb.set_trace()
                if dim == 1:
                    coarse_vec = coarse_vec[:].copy()
                    coarse_vec.shape = (coarse_vec.size//2, 2)

                grid.x[:] += scale*np.reshape(coarse_vec[:, 0], (ny, nx, 1))
                grid.y[:] += scale*np.reshape(coarse_vec[:, 1], (ny, nx, 1))
                # grid.plot(scalars='force', show_edges=True, cpos="xy")

                p.subplot(i, j)
                # tex, value = prop['eigs'][iplot].split('=')

                # buffer = BytesIO()
                # mathtext.math_to_image(tex, buffer, dpi=1000, format='png')
                # image = pv.read(buffer)
                # p.add_text(f'{tex}\n{value}', position='upper_edge', font_size=4)
                p.add_text(f"{prop['eigs'][iplot]}", position='upper_edge', font_size=4)
                # p.add_text(r'$\alpha$', position='upper_edge', font_size=12)
                p.add_mesh(grid, show_edges=True)
                iplot += 1
    p.show(cpos='xy', screenshot=f'coarse_vec_{k}.png')
