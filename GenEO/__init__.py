# Authors:
#     Loic Gouarin <loic.gouarin@cmap.polytechnique.fr>
#     Nicole Spillane <nicole.spillane@cmap.polytechnique.fr>
#
# License: BSD 3 clause
from .assembling import buildElasticityMatrix, buildIdentityMatrix, buildMassMatrix
from .bc import bcApplyWest, bcApplyEast, bcApplyWest_vec
from .rhs import buildRHS
from .utils import buildVecWithFunction, buildCellArrayWithFunction
from .precond import PCBNN, PCNew
from .projection import projection, GenEO_V0, minimal_V0, coarse_operators
from .cg import cg, KSP_PCG, KSP_AMPCG
