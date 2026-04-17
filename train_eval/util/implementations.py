import importlib
import inspect
import logging
import pkgutil


def _all_subclasses(cls):
    if cls is None:
        return []  # or raise ValueError("cls must be provided")
    seen = set()
    stack = list(cls.__subclasses__())
    while stack:
        sub = stack.pop()
        if sub in seen:
            continue
        seen.add(sub)
        yield sub
        stack.extend(sub.__subclasses__())


def _concrete_subclasses(cls):
    if cls is None:
        return []
    return [c for c in _all_subclasses(cls) if not inspect.isabstract(c)]


def find_implementations_in_package(stage_base_cls, root_pkg):
    if stage_base_cls is None:
        return []
    _import_package_tree(root_pkg)
    impls = []
    for cls in _all_subclasses(stage_base_cls):
        if cls.__module__.startswith(root_pkg.__name__) and not inspect.isabstract(cls):
            impls.append(cls)
    return impls


def _import_package_tree(pkg):
    prefix = pkg.__name__ + "."
    for finder, name, ispkg in pkgutil.walk_packages(pkg.__path__, prefix):
        importlib.import_module(name)


def print_stacktrace():
    stack = inspect.stack()
    for frame in stack[1:]:  # skip this function itself
        logging.warning(f"Function {frame.function} "
                        f"in {frame.filename}:{frame.lineno}")
