import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from pythreejs import *
import ipywidgets as widgets
from collections import OrderedDict
from glob import glob

def scalar2rgb(x, cmap='rainbow'):
    import matplotlib.cm, matplotlib.colors
    cNorm = matplotlib.colors.Normalize(vmin=x.min(), vmax=x.max())
    colormap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    return colormap.to_rgba(x)[..., :3]


def read_data(filename):
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    output = reader.GetOutput()
    data_arrays = output.GetPointData()
    numpy_arrays = [vtk_to_numpy(data_arrays.GetArray(i)) for i in range(data_arrays.GetNumberOfArrays())]
    coords = vtk_to_numpy(output.GetPoints().GetData())
    dim = output.GetDataDimension()
    ncells = reader.GetNumberOfCells()
    elements = np.zeros((ncells, 2 ** dim), dtype=np.uint16)
    for ic in range(ncells):
        c = output.GetCell(ic)
        elements[ic] = [c.GetPointId(ip) for ip in range(c.GetNumberOfPoints())]

    if dim == 2:
        fieldnames = [
         'u', 'v', 'lambda', 'rank']
    else:
        fieldnames = [
         'u', 'v', 'v', 'lambda', 'rank']
    return (
     dim, coords, elements, numpy_arrays, fieldnames)


def read_coarse_vec(filename):
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    output = reader.GetOutput()
    data_arrays = output.GetPointData()
    numpy_arrays = [vtk_to_numpy(data_arrays.GetArray(i)) for i in range(data_arrays.GetNumberOfArrays())]
    return numpy_arrays


def get_faces_and_edges(index):
    if index.shape[1] == 4:
        faces = np.concatenate((index[:, :3], index[:, (0, 2, 3)]))
        edges = np.concatenate((index[:, (0, 1)], index[:, (1, 2)], index[:, (2, 3)], index[:, (3, 0)]))
    else:
        t = ((0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (1, 5, 6), (1, 6, 2), (0, 4, 7),
             (0, 7, 3), (3, 2, 6), (3, 6, 7), (0, 1, 5), (0, 5, 4))
        f = []
        for i in t:
            f.append(index[:, i])

        t = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4),
             (1, 5), (3, 7), (2, 6))
        e = []
        for i in t:
            e.append(index[:, i])

        faces = np.concatenate(f)
        edges = np.concatenate(e)
    return (faces, edges)


def plot_solution(path, filename):
    coarse_vecs = []
    coarse_files = glob(path + '/coarse_vec*.vts')
    coarse_files.sort()
    for f in coarse_files:
        coarse_vecs.append(read_coarse_vec(f))

    dim, coords, index, numpy_arrays, fieldnames = read_data(path + '/' + filename)
    work = np.zeros_like(coords)
    faces_, edges_ = get_faces_and_edges(index)
    irank = fieldnames.index('rank')
    faces_by_rank = []
    edges_by_rank = []
    for i in range(int(numpy_arrays[irank].max()) + 1):
        mask = numpy_arrays[irank] == i
        ranges = np.arange(mask.size)[mask]
        indices = np.in1d(faces_, ranges)
        indices = np.all(indices.reshape(faces_.shape), axis=1)
        faces_by_rank.append(faces_[indices])
        indices = np.in1d(edges_, ranges)
        indices = np.all(indices.reshape(edges_.shape), axis=1)
        edges_by_rank.append(edges_[indices])

    bcolor = BufferAttribute(array=scalar2rgb(numpy_arrays[0])[faces_],
      normalized=False)
    vertices = BufferAttribute(array=coords[faces_].reshape(-1, 3),
      normalized=False)
    geometry = BufferGeometry(attributes={'position':vertices, 
     'color':bcolor})
    material = MeshBasicMaterial(color='0xffffff', wireframe=True)
    material_color = MeshLambertMaterial(side='DoubleSide',
      color='0xF5F5F5',
      vertexColors='VertexColors',
      transparent=True,
      opacity=0.5)
    mesh_color = Mesh(geometry, material_color)
    edges = BufferAttribute(array=coords[edges_].reshape(-1, 3),
      normalized=False)
    geom_edges = BufferGeometry(attributes={'position': edges})
    material = LineBasicMaterial(color='0xffffff', linewidth=1)
    mesh = LineSegments(geom_edges, material)
    view_width = 600
    view_height = 600
    camera = PerspectiveCamera(fov=50, position=[0, 0, 13], aspect=view_width / view_height)
    ambient_light = AmbientLight()
    scene = Scene()
    scene.add(mesh_color)
    scene.add(mesh)
    scene.add(ambient_light)
    minCoords = coords.min(axis=0)
    maxCoords = coords.max(axis=0)
    midCoords = 0.5 * (minCoords + maxCoords)
    scene.position = tuple(-midCoords)
    scene.background = 'black'
    controller = OrbitControls(controlling=camera)
    if dim == 2:
        controller.enableRotate = False
    renderer = Renderer(camera=camera, scene=scene, controls=[controller], width=view_width,
      height=view_height)
    fields = OrderedDict()
    for v, k in enumerate(fieldnames):
        fields[k] = v

    select = widgets.Dropdown(options=fields,
      value=0,
      description='Fields')
    irank = fieldnames.index('rank')
    domain = OrderedDict({'all': -1})
    for i in range(int(numpy_arrays[irank].max()) + 1):
        domain[('domain {}').format(i)] = i

    select_dom = widgets.Dropdown(options=domain,
      value=-1,
      description='domains')
    coarse_vecs_label = OrderedDict({'none': -1})
    for i in range(len(coarse_vecs)):
        coarse_vecs_label[('coarse vec {}').format(i)] = i

    select_coarse_vecs = widgets.Dropdown(options=coarse_vecs_label,
      value=-1,
      description='coarse vecs')
    show_displacement = widgets.Checkbox(value=False,
      description='Show displacement',
      disabled=False)
    show_mesh = widgets.Checkbox(value=True,
      description='Show mesh',
      disabled=False)
    scale = widgets.IntSlider(value=100,
      min=1,
      max=1000,
      step=1,
      description='Scale factor for coarse vec displacement',
      disabled=False,
      continuous_update=False,
      orientation='horizontal')

    def draw_mesh(change):
        mesh.visible = not mesh.visible

    def update_domain(change):
        rank = select_dom.value
        work[:] = coords
        if show_displacement.value:
            if select_coarse_vecs.value == -1:
                for i in range(dim):
                    work[(..., i)] += numpy_arrays[i]

            else:
                for i in range(dim):
                    work[(..., i)] += scale.value * coarse_vecs[select_coarse_vecs.value][i]

            if rank >= 0:
                new_coords = work[faces_by_rank[rank]].reshape(-1, 3)
                minCoords = new_coords.min(axis=0)
                maxCoords = new_coords.max(axis=0)
                midCoords = 0.5 * (minCoords + maxCoords)
                scene.position = tuple(-midCoords)
                vertices.array = work[faces_by_rank[rank]].reshape(-1, 3)
                bcolor.array = scalar2rgb(numpy_arrays[select.value])[faces_by_rank[rank]]
                edges.array = work[edges_by_rank[rank]].reshape(-1, 3)
            else:
                vertices.array = work[faces_].reshape(-1, 3)
                bcolor.array = scalar2rgb(numpy_arrays[select.value])[faces_]
                edges.array = work[edges_].reshape(-1, 3)
                minCoords = work.min(axis=0)
                maxCoords = work.max(axis=0)
                midCoords = 0.5 * (minCoords + maxCoords)
                scene.position = tuple(-midCoords)

    select.observe(update_domain, names=['value'])
    select_dom.observe(update_domain, names=['value'])
    select_coarse_vecs.observe(update_domain, names=['value'])
    scale.observe(update_domain, names=['value'])
    show_displacement.observe(update_domain, names=['value'])
    show_mesh.observe(draw_mesh, names=['value'])
    show_displacement.value = True
    return widgets.HBox([renderer, widgets.VBox([select, select_dom, select_coarse_vecs, show_displacement, show_mesh, scale])])
# okay decompiling plot.cpython-36.pyc
