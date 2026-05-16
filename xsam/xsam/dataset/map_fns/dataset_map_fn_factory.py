from functools import partial

from mmengine.utils.misc import get_object_from_string
from xtuner.registry import MAP_FUNC


def dataset_map_fn_factory(fn, **kwargs):
    """Build a partial dataset map function.

    Args:
        fn: Callable map function or a registered/importable string.
        **kwargs: Keyword arguments bound into the partial.

    Returns:
        A partially applied callable map function.
    """
    if isinstance(fn, str):
        fn = MAP_FUNC.get(fn) or get_object_from_string(fn)
    if not callable(fn):
        raise TypeError(f"`fn` must be callable after resolution, but got {type(fn)!r}: {fn!r}")
    return partial(fn, **kwargs)
