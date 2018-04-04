
#       File author(s): Christian Fournier <Christian.Fournier@supagro.inra.fr>


""" A high level class interface to RATP
"""
import json
import numpy
import pandas

from collections import Iterable

import alinea.astk.scene.pgl_scene as pgls
from alinea.astk.scene.display import display_property
from alinea.astk.scene.surfacic_point_cloud import SurfacicPointCloud
from alinea.astk.scene.tagged_mesh import unit_square_mesh
from alinea.pyratp.grid import Grid
from alinea.pyratp.interface.clumping_index import get_clumping
from alinea.pyratp.interface.post_processing import aggregate_light, aggregate_grid
from alinea.pyratp.interface.smart_grid import SmartGrid
from alinea.pyratp.micrometeo import MicroMeteo
from alinea.pyratp.runratp import runRATP
from alinea.pyratp.skyvault import Skyvault, elevations46, azimuths46, omegas_46
from alinea.pyratp.vegetation import Vegetation


def sample_database():
    """ to test if th creative commons database of french city fits
    """
    return {'Montpellier': {'city': 'Montpellier', 'latitude':43.61,
                            'longitude':3.87}}


class RatpScene(object):
    """ High level class interface for RATP lighting model
    """

    localisation_db = sample_database()
    timezone = 0  # consider UTC/GMT time for date inputs
    idecaly = 0   # regular grid
    nitrogen = 2  # dummy value (g.m-2) for nitrogen per leaf area

    def __init__(self, scene=None, grid=None, entities=None, rleaf=0.15,
                 rsoil=0.2, nbinclin=9, orientation=0, localisation='Montpellier',
                 scene_unit='m', mu=None, distinc=None, **grid_kwds):
        """
        Initialise a RatpScene.

        Arguments:
        scene: a pyratp.interface.surfacic_point_cloud or a plantgl.Scene or a
        {shape_id: vertices, faces} scene mesh dict encoding the 3D scene
        grid: a pyratp.interface.smart_grid instance encoding the voxel grid.
         If None (default), the grid is adjusted to fit the input scene.
        entities: a {shape_id: entity} dict defining entities in the scene. If
         None (default), all shapes are considered as members of the same entity
        rleaf: leaf reflectance or a {entity: leaf_reflectance} property dict.
        rsoil: soil reflectance
        nbinclin: the number of angular classes for leaf angle distribution
        orientation (float): the angle (deg, positive clockwise) from X+ to
         North (default: 0)
        localisation : a string referencing a city of the localisation database
         (class variable), or a dict{'longitude':longitude, 'latitude':latitude}
        scene_unit: a string indicating unit used for scene coordinates
        mu : a list of clumping indices, indexed by density.If None (default)
         clumping is automatically evaluated using clark-evans method.
        distinc: a list of list giving the frequency of the nbinclin leaf angles
         classes (from horizontal to vertical) per entity in the canopy.
         If None (default), distinc is computed automatically from the scene

        Other named args to this function are used for controlling grid shape
        when grid=None

        """

        # scene_box will be used to construct grid, if grid is None
        scene_box = ((0, 0, 0), (1, 1, 1))

        if scene is None:
            scene = unit_square_mesh()

        if isinstance(scene, SurfacicPointCloud):
            self.scene = scene
            self.scene_mesh = scene.as_scene_mesh()
            if grid is None:
                scene_box = scene.bbox()
        elif pgls.is_pgl_scene(scene):
            self.scene_mesh = pgls.as_scene_mesh(scene)
            self.scene = SurfacicPointCloud.from_scene_mesh(self.scene_mesh,
                                                            scene_unit=scene_unit)
            if grid is None:
                scene_box = pgls.bbox(scene, scene_unit)
        else:
            try:
                self.scene_mesh = scene
                self.scene = SurfacicPointCloud.from_scene_mesh(scene,
                                                                scene_unit=scene_unit)
                if grid is None:
                    vertices, faces = zip(*scene.values())
                    vertices = reduce(lambda pts, new: pts + list(new), vertices, [])
                    x, y, z = zip(*vertices)
                    scene_box = (
                    (min(x), min(y), min(z)), (max(x), max(y), max(z)))
            except Exception as details:
                print details
                raise ValueError("Unrecognised scene format: should be one of pgl.Scene, "
                      "SurfacicPointCloud or {sh_id:(vertices, faces)} dict")

        if grid is None:
            self.smart_grid = SmartGrid(scene_box=scene_box, **grid_kwds)
        elif isinstance(grid, SmartGrid):
            self.smart_grid = grid
        else:
            raise ValueError('Unrecognised grid format: should be None or a '
                             'SmartGrid instance')

        if entities is None:
            entities = {sh_id: 'default' for sh_id in self.scene_mesh}
        self.entities = entities
        # RATP entity code starts at 1
        ent = list(set(self.entities.values()))
        self.entity_map = dict(zip(ent, range(1, len(ent) + 1)))
        self.entity_code = numpy.array(
            [self.entity_map[entities[sh]] for sh in self.scene.shape_id])

        if not isinstance(rleaf, dict):
            # if not hasattr(rleaf, '__len__'):
            #     rleaf = [rleaf]
            rleaf = {'default': rleaf}
        self.rleaf = {self.entity_map[k]: rleaf[k] for k in self.entity_map}

        if not hasattr(rsoil, '__len__'):
            rsoil = [rsoil]
        self.rsoil = rsoil

        self.orientation = orientation

        if not isinstance(localisation, dict):
            try:
                localisation = RatpScene.localisation_db[localisation]
            except KeyError:
                print 'Warning : localisation', localisation, \
                    'not found in database, using default localisation', \
                    RatpScene.localisation_db.iter().next()
                localisation = RatpScene.localisation_db.itervalues().next()
        self.localisation = localisation

        if mu is None:
            mu = self.clumping()
            print(' '.join(['clumping evaluated:'] + [str(m) for m in
                                                      mu]))
        self.set_clumping(mu)

        if distinc is None:
            distinc = self.inclination_distribution(nbinclin)
        self.set_inclin(distinc)

        self.ratp_grid, self.grid_indices = self.grid()

    def n_entities(self):
        """ return the number of distinct entities in the canopy"""
        return max(self.entity_code)

    def set_clumping(self, mu):
        nent = self.n_entities()
        if not isinstance(mu, Iterable):
            mu = [mu] * nent
        assert len(mu) == nent
        self.mu = mu

    def set_inclin(self, distinc):
        nent = self.n_entities()
        assert isinstance(distinc, Iterable)
        assert len(distinc) == nent
        nbinclin = len(distinc[0])
        assert all(map(lambda x: len(x) == nbinclin, distinc))
        self.distinc = distinc
        self.nbinclin = nbinclin

    def clumping(self, expand=True):
        spc = self.scene
        grid = self.smart_grid
        x, y, z = spc.x, spc.y, spc.z
        xv, yv, zv = grid.within_cell_position(x, y, z, normalise=True)
        jx, jy, jz = grid.grid_index(x, y, z)
        entity = self.entity_code
        data = pandas.DataFrame(
            {'entity': entity, 'x': xv, 'y': yv, 'z': zv, 's': spc.area,
             'n': spc.normals,
             'jx': jx, 'jy': jy, 'jz': jz})
        mu = []
        domain = ((0, 0, 0), (1, 1, 1))
        grouped = data.groupby('entity')
        for e, dfe in grouped:
            gvox = dfe.groupby(('jx', 'jy', 'jz'))
            clumps = []
            for k, df in gvox:
                clumping = get_clumping(df['x'], df['y'], df['z'], df['s'],
                                        df['n'], domain=domain, expand=expand)
                min_mu = df['s'].mean() / df[
                    's'].sum()  # minimal mu in the case of perfect clumping
                if numpy.isfinite(clumping):
                    clumps.append(max(min_mu, clumping))
            mu.append(numpy.mean(clumps))

        return mu

    def grid(self):

        """ Create and fill a RATP grid
        """

        spc = self.scene
        grid_pars = {'latitude': self.localisation['latitude'],
                     'longitude': self.localisation['longitude'],
                     'timezone': self.timezone,
                     'idecaly': self.idecaly,
                     'orientation': self.orientation,
                     'rs': self.rsoil,
                     'nent': self.n_entities()
                     }
        grid_pars.update(self.smart_grid.ratp_grid_parameters())
        ratp_grid = Grid.initialise(**grid_pars)
        grid_indices = self.smart_grid.grid_index(spc.x, spc.y, spc.z,
                                                  check=True)
        jx, jy, jz = self.smart_grid.ratp_grid_index(*grid_indices)
        entity = self.entity_code - 1  # Grid.fill expect python indices
        nitrogen = [self.nitrogen] * spc.size
        ratp_grid, _ = Grid.fill_from_index(entity, jx, jy, jz, spc.area,
                                       nitrogen, ratp_grid)
        # RATPScene grid indices of individual surfacic points
        jx, jy, jz = grid_indices
        grid_indices = pandas.DataFrame({'point_id': range(len(jx)),
                                              'jx': jx, 'jy': jy, 'jz': jz})

        return ratp_grid, grid_indices

    def inclinations(self):
        df = self.scene.inclinations()
        df_ent = pandas.DataFrame({'shape_id': self.scene.shape_id,
                                   'entity': self.entity_code}).drop_duplicates()
        df = df.merge(df_ent)
        return [group['inclination'].tolist() for name, group in
                df.groupby('entity')]

    def inclination_distribution(self, nbinclin=9):

        def _dist(inc):
            dist = numpy.histogram(inc, nbinclin, (0, 90))[0]
            return (dist.astype('float') / dist.sum()).tolist()
        inclinations = self.inclinations()
        distinc = map(_dist, inclinations)

        return distinc

    def voxel_index(self):
        """Mapping between RATP (filled) VoxelId and RatpScene grid indices"""

        grid = self.ratp_grid

        if grid is None:
            return pandas.DataFrame({})

        nveg = grid.nveg

        if nveg == 0:
            return pandas.DataFrame({})

        index = numpy.arange(1, nveg + 1)
        # RATP fortan indices
        numx, numy, numz = grid.numx[:nveg], grid.numy[:nveg], grid.numz[:nveg]
        # associated smart_grid indices
        jx, jy, jz = self.smart_grid.decode_ratp_indices(numx - 1, numy - 1,
                                                         numz - 1)
        xc, yc, zc = self.smart_grid.voxel_centers(jx, jy, jz)

        return pandas.DataFrame(
            {'VoxelId': index, 'jx': jx, 'jy': jy, 'jz': jz, 'xc': xc, 'yc': yc,
             'zc': zc})

    def do_irradiation(self, sun_sources=None, sky_sources=None,
                       scattering_indicatrix=None, mu=None, distinc=None):
        """ Run a simulation of light interception for one wavelength

            Parameters:
                sun_sources: elevation (degrees), azimuth (degrees, from North
                    positive clockwise), horizontal irradiance of sources
                    representing the sun.
                sky_sources: elevation (degrees), azimuth (degrees, from North
                    positive clockwise), horizontal irradiance of sources
                    representing the sky.
                    If both sun_sources and sky_sources are None, a 46
                    directions, normalised soc sky is used
                scattering_indicatrix: relative weights associated to sky_sources
                    directions used to compute interception coefficient of scattered
                    light. If None uniform contribution is used if sky_sources
                    is not None or default RATP turtle 46 if sky_sources is None
                mu: clumping index of vegetation. If None, clumping index is
                    estimated after modified clark evans index
                distinc: a list of list giving the frequency of the nbinclin
                    leaf angles classes (from horizontal to vertical) per entity
                    in the canopy.If None (default), distinc is computed
                    automatically from the scene
        """

        nent = self.n_entities()

        if mu is not None:
            self.set_clumping(mu)

        if distinc is not None:
            self.set_inclin(distinc)

        entities = [{'rf': [self.rleaf[i + 1]], 'distinc': self.distinc[i],
                     'mu': self.mu[i]} for i
                    in range(nent)]
        vegetation = Vegetation.initialise(entities, nblomin=1)

        if sun_sources is None and sky_sources is None:
            sky = Skyvault.initialise()
            ghi = 1
        else:
            hmoy, azmoy, omega, pc = [], [], [], []
            if sun_sources is not None:
                el, az, irr = sun_sources
                hmoy.extend(el)
                azmoy.extend(az)
                pc.extend(irr)
                # sun directions are not used to estimate interception coeff of
                # scattered light 'rka' (mod_Shortwave_Balance.f90, line 142 +
                # mod_dir_interception.f90, line 112)
                omega.extend([0.] * len(irr))
            if sky_sources is not None:
                el, az, irr = sky_sources
                hmoy.extend(el)
                azmoy.extend(az)
                pc.extend(irr)
                if scattering_indicatrix is None:
                    omega.extend([1.] * len(irr))
                else:
                    omega.extend(scattering_indicatrix)
            else:
                hmoy.extend(elevations46)
                azmoy.extend(azimuths46)
                omega.extend(omegas_46)
                # only use for omega / rediff, not fo lighting
                pc.extend([0] * len(omegas_46))

            # pc is used (mod_Hemi_Interception, line 124) to weight Rdif
            ghi = sum(pc)
            if ghi == 0:
                raise ValueError('Irradiance of sources is null!!!')
            pc = numpy.array(pc) / ghi
            # omega are Sr solid angles (mod_dir_interception.f90, line 112)
            omega = numpy.array(omega)
            omega /= omega.sum()
            omega *= 2 * numpy.pi
            # RATP sources azimuths are from South, positive clockwise
            # (mod_Shortwave_Balance, line 83) and orientation is not taken into
            # account for diffuse sources, unike RATP native sun sources
            # (mod_Shortwave_Balance.f90, line 136)
            azmoy = numpy.array(azmoy) - 180 + self.orientation
            sky = Skyvault.initialise(hmoy=hmoy, azmoy=azmoy, omega=omega, pc=pc)
        met = MicroMeteo.initialise(doy=1, hour=12, Rglob=ghi, Rdif=ghi)
        res = runRATP.DoIrradiation(self.ratp_grid, vegetation, sky, met)

        VegetationType, Iteration, day, hour, VoxelId, ShadedPAR, SunlitPAR, \
        ShadedArea, SunlitArea = res.T
        # 'PAR' is expected in  Watt.m-2 in RATP input, whereas output is in
        #  micromol => convert back to W.m2 (cf shortwavebalance, line 306)
        dfvox = pandas.DataFrame({'VegetationType': VegetationType,
                                  'Iteration': Iteration,
                                  'day': day,
                                  'hour': hour,
                                  'VoxelId': VoxelId,
                                  'ShadedPAR': ShadedPAR / 4.6,
                                  'SunlitPAR': SunlitPAR / 4.6,
                                  'ShadedArea': ShadedArea,
                                  'SunlitArea': SunlitArea,
                                  'Area': ShadedArea + SunlitArea,
                                  'PAR': (
                                         ShadedPAR * ShadedArea + SunlitPAR *
                                         SunlitArea) / (
                                         ShadedArea + SunlitArea) / 4.6,
                                  })

        return pandas.merge(dfvox, self.voxel_index())

    def scene_lightmap(self, dfvox, spatial='point_id', temporal=True):
        """  Aggregate light outputs along spatial domain of scene input

        Args:
            dfvox: a pandas data frame with ratp outputs
            spatial: a string indicating the aggregation level: 'point_id' or
             'shape_id' .
            temporal: should iterations be aggregated ?

        Returns:
            a pandas dataframe with aggregated outputs

        """
        if spatial == 'point_id':
            scene_map = self.scene.area_map()
        else:
            scene_map = self.scene.shape_map().merge(self.scene.area_map())
        grid_map = self.grid_indices

        return aggregate_light(dfvox, scene_map=scene_map, grid_map=grid_map,
                               temporal=temporal)

    def xy_lightmap(self, dfvox):
        """ Aggregate light outputs along x y cells"""
        grid_map = self.grid_indices
        area_map = self.scene.area_map()
        return aggregate_grid(dfvox, grid_map, area_map)

    def plot(self, dfvox, by='point_id', minval=None, maxval=None):
        lmap = self.scene_lightmap(dfvox, spatial=by)
        if by == 'shape_id':
            df = lmap.loc[:, ('shape_id', 'PAR')].set_index('shape_id')
            prop = {k: df['PAR'][k] for k in df.index}
        else:
            # add shape_id
            lmap = lmap.merge(self.scene.shape_map())
            lmap = lmap.sort_values('point_id')
            prop = {sh_id: df.to_dict('list')['PAR'] for sh_id, df in
                    lmap.groupby('shape_id')}
        return display_property(self.scene_mesh, prop, minval=minval,
                                maxval=maxval)

    def parameters(self):
        """return a dict of intantiation parameters"""
        ent = self.entities
        rl = self.rleaf
        if self.n_entities() == 1:
            ent = ent.itervalues().next()
            rl = rl.itervalues().next()
            if ent == 'default':
                ent = None
        d = {'grid': self.smart_grid.as_dict(), 'entities': ent, 'rleaf': rl,
             'rsoil': self.rsoil, 'nbinclin': self.nbinclin,
             'orientation': self.orientation, 'localisation': self.localisation,
             'mu': self.mu, 'distinc':self.distinc}
        return d

    @staticmethod
    def read_parameters(file_path):
        with open(file_path, 'r') as input_file:
            pars = json.load(input_file)
        return pars

    def save_parameters(self, file_path):
        saved = self.parameters()
        with open(file_path, 'w') as output_file:
            json.dump(saved, output_file, sort_keys=True, indent=4,
                      separators=(',', ': '))

    @staticmethod
    def load_ratp_scene(scene, parameters):
        # copy to avoid altering parameters
        pars = {k:v for k,v in parameters.iteritems()}
        grid = SmartGrid.from_dict(pars.pop('grid'))
        return RatpScene(scene, grid=grid, **pars)