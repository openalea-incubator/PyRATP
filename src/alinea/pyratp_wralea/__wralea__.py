
# This file has been generated at Mon Feb 14 11:30:22 2011

from openalea.core import *


__name__ = 'PyRATP'

__editable__ = True
__description__ = ''
__license__ = 'CeCILL-C'
__url__ = 'http://openalea.gforge.inria.fr'
__alias__ = []
__version__ = '0.9.0'
__authors__ = ''
__institutes__ = None
__icon__ = ''


__all__ = ['ratp_read_vgx', 'ratp_read_grid', 'ratp_fill_grid', 'ratp_toto_grid']



ratp_read_vgx = Factory(name='plant from vegestar',
                authors=' (wralea authors)',
                description='',
                category='Unclassified',
                nodemodule='ratp',
                nodeclass='read_vgx',
                inputs=[{'interface': IFileStr(filter="*.vgx", save=False), 'name': 'filename', 'value': None, 'desc': 'Vegestar 3d Grid file'}],
                outputs=[{'interface': ISequence, 'name': 'entity', 'desc': 'No output for the moment'}, {'interface': IFloat, 'name': 'x', 'desc': ''}, {'interface': None, 'name': 'y', 'desc': ''}, {'interface': None, 'name': 'z', 'desc': ''}, {'interface': None, 'name': 's', 'desc': ''}, {'interface': None, 'name': 'n', 'desc': ''}],
                widgetmodule=None,
                widgetclass=None,
               )




ratp_read_grid = Factory(name='read grid',
                authors=' (wralea authors)',
                description='Build a RATP Grid',
                category='simulation, ecophysiology',
                nodemodule='ratp',
                nodeclass='read_grid',
                inputs=[{'interface': IFileStr(filter="*.grd", save=False), 'name': 'filename', 'value': None, 'desc': '3d Grid file'}],
                outputs=[{'interface': None, 'name': 'grid', 'desc': 'No output for the moment'}],
                widgetmodule=None,
                widgetclass=None,
               )




ratp_fill_grid = Factory(name='fill grid',
                authors=' (wralea authors)',
                description='fill a RATP Grid with vegetation',
                category='Unclassified',
                nodemodule='ratp',
                nodeclass='fill_grid',
                inputs=[{'interface': IInt, 'name': 'entity', 'value': None, 'desc': '3d Grid file'}, {'interface': IFloat, 'name': 'x', 'value': None, 'desc': ''}, {'interface': IFloat, 'name': 'y', 'value': None, 'desc': ''}, {'interface': IFloat, 'name': 'z', 'value': None, 'desc': ''}, {'interface': IFloat, 'name': 's', 'value': None, 'desc': ''}, {'interface': IFloat, 'name': 'n', 'value': None, 'desc': ''}],
                outputs=[{'interface': None, 'name': 'grid', 'desc': 'No output for the moment'}],
                widgetmodule=None,
                widgetclass=None,
               )




ratp_toto_grid = Factory(name='toto grid',
                authors=' (wralea authors)',
                description='toto',
                category='Unclassified',
                nodemodule='ratp',
                nodeclass='toto_grid',
                inputs=[{'interface': ISequence, 'name': 'entity', 'value': None, 'desc': '3d Grid file'}, {'interface': IFloat, 'name': 'x', 'value': None, 'desc': ''}, {'interface': IFloat, 'name': 'y', 'value': None, 'desc': ''}, {'interface': IFloat, 'name': 'z', 'value': None, 'desc': ''}, {'interface': IFloat, 'name': 's', 'value': None, 'desc': ''}, {'interface': None, 'name': 'n', 'value': None, 'desc': ''}],
                outputs=[{'interface': None, 'name': 'grid', 'desc': 'No output for the moment'}],
                widgetmodule=None,
                widgetclass=None,
               )




