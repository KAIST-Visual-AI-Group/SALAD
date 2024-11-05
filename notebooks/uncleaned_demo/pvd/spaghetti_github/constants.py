import os
import sys

IS_WINDOWS = sys.platform == 'win32'
get_trace = getattr(sys, 'gettrace', None)
DEBUG = get_trace is not None and get_trace() is not None
EPSILON = 1e-4
DIM = 3

#### < Seungwoo >
# Added 'docker_home' directory in the middle.
PROJECT_ROOT = "/home/juil/docker_home/projects/3D_CRISPR/spaghetti_github"
# PROJECT_ROOT = "/home/juil/projects/3D_CRISPR/spaghetti_github"
####

DATA_ROOT = f'{PROJECT_ROOT}/assets/'
CHECKPOINTS_ROOT = f'{DATA_ROOT}checkpoints/'
CACHE_ROOT = f'{DATA_ROOT}cache/'
UI_OUT = f'{DATA_ROOT}ui_export/'
UI_RESOURCES = f'{DATA_ROOT}/ui_resources/'
Shapenet_WT = f'{DATA_ROOT}/ShapeNetCore_wt/'
Shapenet = f'{DATA_ROOT}/ShapeNetCore.v2/'
MAX_VS = 100000
MAX_GAUSIANS = 32

