# Core models with stable dependencies
from . import Autoformer, Transformer, TimesNet, Nonstationary_Transformer
from . import DLinear, FEDformer, Informer, LightTS, Reformer, ETSformer
from . import Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer
from . import Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN
from . import TemporalFusionTransformer, SCINet, PAttn, TimeXer
from . import WPMixer, MultiPatchFormer, KANAD, MSGNet, TimeFilter

# Optional models - gracefully skip if dependencies are missing
try:
    from . import Mamba
    MambaSimple = Mamba
except ImportError:
    MambaSimple = None

try:
    from . import Sundial
except ImportError:
    Sundial = None

try:
    from . import TimeMoE
except ImportError:
    TimeMoE = None

try:
    from . import Chronos
except ImportError:
    Chronos = None

try:
    from . import Moirai
except ImportError:
    Moirai = None

try:
    from . import TiRex
except ImportError:
    TiRex = None

try:
    from . import TimesFM
except ImportError:
    TimesFM = None

try:
    from . import Chronos2
except ImportError:
    Chronos2 = None

__all__ = [
    'Autoformer', 'Transformer', 'TimesNet', 'Nonstationary_Transformer',
    'DLinear', 'FEDformer', 'Informer', 'LightTS', 'Reformer', 'ETSformer',
    'Pyraformer', 'PatchTST', 'MICN', 'Crossformer', 'FiLM', 'iTransformer',
    'Koopa', 'TiDE', 'FreTS', 'TimeMixer', 'TSMixer', 'SegRNN',
    'MambaSimple', 'TemporalFusionTransformer', 'SCINet', 'PAttn', 'TimeXer',
    'WPMixer', 'MultiPatchFormer', 'KANAD', 'MSGNet', 'TimeFilter',
    'Sundial', 'TimeMoE', 'Chronos', 'Moirai', 'TiRex', 'TimesFM', 'Chronos2'
]