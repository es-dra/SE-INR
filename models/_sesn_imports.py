import sys
import os
import types
import importlib.util

_SESN_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'sesn')
_SESN_IMPL = os.path.join(_SESN_ROOT, 'models', 'impl')


def _patch_sesn_basis(mod):
    orig_normalize = getattr(mod, 'normalize_basis_by_min_scale', None)
    if orig_normalize is None:
        return

    def patched_normalize(basis):
        num_funcs = basis.shape[0]
        num_scales = basis.shape[1]
        norm_per_func = basis.pow(2).sum([2, 3], keepdim=True).sqrt()
        ref_norm = norm_per_func[:, [0]]
        safe_mask = ref_norm > 1e-10
        ref_norm = ref_norm.clamp(min=1e-10)
        result = basis / ref_norm
        result[~safe_mask.expand_as(result)] = 0
        return result

    mod.normalize_basis_by_min_scale = patched_normalize


def _load_sesn_modules(force_reload=False):
    if not force_reload and '_sesn_loaded' in globals() and globals()['_sesn_loaded']:
        return globals()

    if force_reload:
        for k in list(sys.modules.keys()):
            if k.startswith('_se_sesn_impl'):
                del sys.modules[k]
        globals()['_sesn_loaded'] = False

    impl_pkg = types.ModuleType('_se_sesn_impl')
    impl_pkg.__path__ = [_SESN_IMPL]
    sys.modules['_se_sesn_impl'] = impl_pkg

    spec_b = importlib.util.spec_from_file_location(
        '_se_sesn_impl.ses_basis', os.path.join(_SESN_IMPL, 'ses_basis.py'))
    mod_b = importlib.util.module_from_spec(spec_b)
    sys.modules['_se_sesn_impl.ses_basis'] = mod_b
    spec_b.loader.exec_module(mod_b)

    _patch_sesn_basis(mod_b)

    spec_c = importlib.util.spec_from_file_location(
        '_se_sesn_impl.ses_conv', os.path.join(_SESN_IMPL, 'ses_conv.py'))
    mod_c = importlib.util.module_from_spec(spec_c)
    sys.modules['_se_sesn_impl.ses_conv'] = mod_c
    spec_c.loader.exec_module(mod_c)

    globals()['SESConv_Z2_H'] = mod_c.SESConv_Z2_H
    globals()['SESConv_H_H'] = mod_c.SESConv_H_H
    globals()['SEMaxProjection'] = mod_c.SESMaxProjection
    globals()['SESConv_H_H_1x1'] = mod_c.SESConv_H_H_1x1
    globals()['steerable_A'] = mod_b.steerable_A
    globals()['steerable_B'] = mod_b.steerable_B
    globals()['_sesn_loaded'] = True
    return globals()


def get_sesn_classes():
    m = _load_sesn_modules()
    return {
        'SESConv_Z2_H': m['SESConv_Z2_H'],
        'SESConv_H_H': m['SESConv_H_H'],
        'SEMaxProjection': m['SEMaxProjection'],
        'SESConv_H_H_1x1': m.get('SESConv_H_H_1x1'),
        'steerable_A': m['steerable_A'],
        'steerable_B': m['steerable_B'],
    }
