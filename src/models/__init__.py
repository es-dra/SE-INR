from .models import register, make
from . import edsr
from . import rdn
from . import mlp
from . import liif
from . import liif_sp
from . import lte
from . import sc_inr_adaptive
from . import sc_inr_exact
try:
    from . import swinir  # needs timm
except ImportError:
    pass
