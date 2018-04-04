import numpy
from alinea.pyratp.interface.ratp_scene import RatpScene, pgls
from alinea.astk.scene.surfacic_point_cloud import SurfacicPointCloud


def test_instanciation():
    ratp = RatpScene()
    numpy.testing.assert_almost_equal(ratp.smart_grid.bbox(),
                                      ((-0.05, -0.05, -0.55),
                                       (1.05, 1.05, 0.55)), decimal=2)
    numpy.testing.assert_almost_equal(ratp.scene.normals,
                                      [[0., 0., 1.], [0., 0., 1.]])
    for w in ('city', 'latitude', 'longitude'):
        assert w in ratp.localisation
    assert isinstance(ratp.entities, dict)
    assert len(ratp.entity_code) == ratp.scene.size
    assert ratp.ratp_grid.s_canopy == 1
    numpy.testing.assert_almost_equal(ratp.ratp_grid.total_ground_area, 1.2345678)
    numpy.testing.assert_array_equal(ratp.distinc[0],
                                     [1., 0., 0., 0., 0., 0., 0., 0., 0.])
    vindex = ratp.voxel_index()

    ratp = RatpScene(shape=[5, 5, 10])
    numpy.testing.assert_array_equal(ratp.smart_grid.shape, [5, 5, 10])

    if pgls.pgl_imported:
        scene = pgls.pgl.Scene([pgls.pgl.Sphere()])
        ratp = RatpScene(scene)
        numpy.testing.assert_almost_equal(ratp.smart_grid.bbox(), (
        (-0.55, -0.55, -0.55), (0.55, 0.55, 0.55)), decimal=2)
        assert 'shape_id' in ratp.scene.as_data_frame().columns
        assert ratp.scene.size == 112
        assert len(ratp.entity_code) == 112


def test_parameters():
    ratp = RatpScene()
    pars = ratp.parameters()
    assert 'grid' in pars
    assert isinstance(pars['grid'], dict)


def test_load_ratp_scene():
    ratp = RatpScene()
    pars = ratp.parameters()
    scene = ratp.scene
    ratp = RatpScene.load_ratp_scene(scene, pars)


def test_light():
    ratp = RatpScene(resolution=(1, 1, 1))
    dfvox = ratp.do_irradiation()
    dfpoints = ratp.scene_lightmap(dfvox, 'point_id')
    dfshape = ratp.scene_lightmap(dfvox, 'shape_id')
    dfxy = ratp.xy_lightmap(dfvox)
    if pgls.pgl_imported:
        ratp.plot(dfvox)
        ratp.plot(dfvox, 'shape_id')
        ratp = RatpScene(pgls.unit_sphere_scene())
        dfvox = ratp.do_irradiation()
        ratp.plot(dfvox)


def test_sky_source_type():
    # small Horizontal surface in  big voxel
    spc = SurfacicPointCloud(0.5, 0.5, 0.5, area=0.1, normals=[(0, 0, 1)])
    # sun_source
    ratp = RatpScene(spc, resolution=(1, 1, 1), rsoil=0, nbinclin=90)
    dfsun = ratp.do_irradiation(sun_sources=([90], [0], [1]))
    dfsky = ratp.do_irradiation(sky_sources=([90], [0], [1]))
    assert dfsky.PAR.values > dfsun.PAR.values
    dfscat = ratp.do_irradiation(sun_sources=([90], [0], [1]),
                                 sky_sources=([90], [0], [0]),
                                 scattering_indicatrix=[1])
    numpy.testing.assert_almost_equal(dfsky.PAR.values, dfscat.PAR.values,
                                      decimal=2)


def test_sources():
    # small Horizontal surface in  big voxel
    spc = SurfacicPointCloud(0.5, 0.5, 0.5, area=0.1, normals=[(0, 0, 1)])
    # nearly horizontal square, zenith light
    ratp = RatpScene(spc, resolution=(1, 1, 1), rsoil=0, nbinclin=90)
    dfv = ratp.do_irradiation(sun_sources=([90], [0], [1]))
    numpy.testing.assert_almost_equal(dfv.PAR.values, 0.96, decimal=2)
    # zenith light, irrad=10
    dfv = ratp.do_irradiation(sun_sources=([90], [0], [10]))
    numpy.testing.assert_almost_equal(dfv.PAR.values, 9.66, decimal=2)
    # inclined light
    dfv = ratp.do_irradiation(
        sun_sources=([45], [0], [numpy.sin(numpy.radians(45))]))
    numpy.testing.assert_almost_equal(dfv.PAR.values, 0.69, decimal=2)
    # inclined + vertical light
    dfv = ratp.do_irradiation(
        sun_sources=([45, 90], [0, 0], [numpy.sin(numpy.radians(45)), 1]))
    numpy.testing.assert_almost_equal(dfv.PAR.values, 1.66, decimal=2)
    # vertical surface, zenith light
    vd = [0] * 89 + [1]
    vh = [1] + [0] * 89
    dfv = ratp.do_irradiation(sun_sources=([90], [0], [1]), distinc=[vd])
    numpy.testing.assert_almost_equal(dfv.PAR.values, 0.008, decimal=3)
    # vertical surface, near horizontal light
    dfv = ratp.do_irradiation(
        sun_sources=([1], [0], [numpy.sin(numpy.radians(1))]), distinc=[vd])
    numpy.testing.assert_almost_equal(dfv.PAR.values, 0.63, decimal=2)

    # Test Orientation
    # Three filled voxels with lad heterogeneity along X+ (North/South)
    spc = SurfacicPointCloud([0.5, 1.5, 2.5], [0.5] * 3, [0.5] * 3,
                             [0.1, 0.1, 100])
    ratp = RatpScene(spc, resolution=(1, 1, 1), rsoil=0, rleaf=0, nbinclin=90,
                     distinc=[vd], mu=1)
    # radiant source coming from X+
    dfv = ratp.do_irradiation(
        sun_sources=([1], [0], [numpy.sin(numpy.radians(1))]))
    v1 = dfv.loc[dfv.jx==1,'PAR'].values
    # source coming from X-
    dfv = ratp.do_irradiation(
        sun_sources=([1], [180], [numpy.sin(numpy.radians(1))]))
    v2 = dfv.loc[dfv.jx == 1, 'PAR'].values
    # compare irrad on central voxel
    assert v2 > v1
    # perpendicular source coming from Y-
    dfv = ratp.do_irradiation(
        sun_sources=([1], [90], [numpy.sin(numpy.radians(1))]))
    numpy.testing.assert_almost_equal(dfv.loc[dfv.jx == 1, 'PAR'].values,
                                      dfv.loc[dfv.jx == 0, 'PAR'].values,
                                      decimal=2)
    # Test left-right / East-West
    # a scene with high lad on Y- (east), north being towards X+
    spc = SurfacicPointCloud([0.5] * 3, [0.5, 1.5, 2.5], [0.5] * 3,
                             [100, 0.1, 0.1])
    ratp = RatpScene(spc, resolution=(1, 1, 1), rsoil=0, rleaf=0, nbinclin=90,
                     distinc=[vd], mu=1)
    # radiant source coming from east (north, positive clockwise convention)
    dfv = ratp.do_irradiation(
        sun_sources=([1], [90], [numpy.sin(numpy.radians(1))]))
    v1 = dfv.loc[dfv.jy==1,'PAR'].values
    # source coming from west
    dfv = ratp.do_irradiation(
        sun_sources=([1], [-90], [numpy.sin(numpy.radians(1))]))
    v2 = dfv.loc[dfv.jy == 1, 'PAR'].values
    assert v2 > v1
    # a scene with high lad on X+ (east), north being towards Y+
    spc = SurfacicPointCloud([0.5, 1.5, 2.5], [0.5] * 3, [0.5] * 3,
                             [0.1, 0.1, 100])
    ratp = RatpScene(spc, resolution=(1, 1, 1), rsoil=0, rleaf=0, nbinclin=90,
                     distinc=[vd], mu=1, orientation=-90)
    # radiant source coming from east (north, positive clockwise convention)
    dfv = ratp.do_irradiation(
        sun_sources=([1], [90], [numpy.sin(numpy.radians(1))]))
    v1 = dfv.loc[dfv.jx==1,'PAR'].values
    # source coming from west
    dfv = ratp.do_irradiation(
        sun_sources=([1], [-90], [numpy.sin(numpy.radians(1))]))
    v2 = dfv.loc[dfv.jx == 1, 'PAR'].values
    assert v2 > v1