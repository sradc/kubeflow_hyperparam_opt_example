# Copyright 2022 Sidney Radcliffe
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Helpers to make it easier to work with looped Kubeflow component outputs.

Likely to be replaced by a more general solution in the future.
"""
import ast
import functools
import importlib
import importlib.util
import inspect
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from typing import Callable, List, Optional

from kfp.v2 import dsl


def pseudo_tuple_component(
    func: Optional[Callable] = None,
    *,
    base_image: Optional[str] = None,
    target_image: Optional[str] = None,
    packages_to_install: List[str] = None,
    pip_index_urls: Optional[List[str]] = None,
    output_component_file: Optional[str] = None,
    install_kfp_package: bool = True,
    kfp_package_path: Optional[str] = None,
    globals_: Optional[dict] = None,
    locals_: Optional[dict] = None,
) -> Callable:
    """Decorator to create a Kubeflow component from a
    function with PseudoTuple arguments.
    """
    if not globals_:
        raise ValueError("globals_ must be provided.")
    if not locals_:
        raise ValueError("locals_ must be provided.")
    if func is None:
        return functools.partial(
            pseudo_tuple_component,
            base_image=base_image,
            target_image=target_image,
            packages_to_install=packages_to_install,
            pip_index_urls=pip_index_urls,
            output_component_file=output_component_file,
            install_kfp_package=install_kfp_package,
            kfp_package_path=kfp_package_path,
            locals_=locals_,
            globals_=globals_,
        )
    with expand_func(func, locals_, globals_) as expanded_func:
        return dsl.component(
            expanded_func,
            base_image=base_image,
            target_image=target_image,
            packages_to_install=packages_to_install,
            pip_index_urls=pip_index_urls,
            output_component_file=output_component_file,
            install_kfp_package=install_kfp_package,
            kfp_package_path=kfp_package_path,
        )


@contextmanager
def expand_func(func: Callable, globals_: dict, locals_: dict) -> Callable:
    """Context manager that returns a function with PseudoTuple
    arguments expanded into multiple arguments.

    A context manager is used because Kubeflow runs inspect.getsource,
    which only works if the function is from a file.
    a tempfile is used to store the expanded function, and the
    tempfile is deleted when the context manager exits.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py") as f:
        f_path = Path(f.name)
        expanded_func_code = expand_PseudoTuple_annotated_args_to_str(
            func, locals_, globals_
        )
        f_path.write_text(expanded_func_code)
        spec = importlib.util.spec_from_file_location(f_path.stem, f.name)
        temp_module = importlib.util.module_from_spec(spec)
        sys.modules[f_path.stem] = temp_module
        # Add the local variables to the module as needed, because the module being created
        # may not have the same globals as the module that is calling this function.
        # E.g. type annotations are not available in the imported module
        # unless we do this, because they are not in the source code we've created.
        # Note it will fail if we try and add all the locals_/globals_.
        locals_copy = locals_.copy()
        globals_copy = globals_.copy()
        while True:
            try:
                spec.loader.exec_module(temp_module)
            except NameError as e:
                missing_name = str(e).split(" ")[1][1:-1]
                if missing_name in locals_copy:
                    temp_module.__dict__[missing_name] = locals_copy[missing_name]
                elif missing_name in globals_copy:
                    temp_module.__dict__[missing_name] = globals_copy[missing_name]
                else:
                    raise e
            else:
                break
        func = getattr(temp_module, func.__name__)
        yield func


class PseudoTuple:
    """To be used as a type annotation."""

    def __init__(self, size=1, type=None):
        assert size > 0
        assert isinstance(size, int)
        self.size = size
        self.type = type


def expand_PseudoTuple_annotated_args_to_str(
    func: Callable, globals_: dict, locals_: dict
) -> str:
    """Returns the source code of the function with PseudoTuple arguments expanded.

     Parameters
     ----------
     func : Callable
         A function with PseudoTuple annotated arguments.

     locals_: dict
        E.g. The output of locals().

    globals_: dict
        E.g. The output of globals().

     Returns
     -------
     Callable
         Code for the function with expanded arguments.
    """
    source = dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]
    # Remove the decorators that use this function
    for decorator in func_def.decorator_list:
        decorator_string = ast.unparse(decorator)
        if (
            decorator_string.startswith(pseudo_tuple_component.__name__ + "(")
            or decorator_string == pseudo_tuple_component.__name__
        ):
            func_def.decorator_list.remove(decorator)
    # Get the indent of the first line, to be used for the lines we add
    first_line_col_offset: int = func_def.body[0].col_offset
    new_args = []
    new_body = [*func_def.body]
    for arg in func_def.args.args:
        name = arg.arg
        annotation = arg.annotation
        if isinstance(annotation, ast.Call) and annotation.func.id == "PseudoTuple":
            if isinstance(annotation.args[0], ast.Num):
                size = annotation.args[0].n
            else:
                size = eval(
                    compile(ast.Expression(annotation.args[0]), "<string>", "eval"),
                    globals_,
                    locals_,
                )
            type = annotation.args[1].id if len(annotation.args) > 1 else None
            for i in range(size):
                new_args.append(
                    ast.arg(arg=f"{name}_{i}", annotation=ast.Name(id=type))
                )
            assign_stmt = ast.Assign(
                targets=[ast.Name(id=name)],
                value=ast.Tuple(
                    elts=[ast.Name(id=f"{name}_{i}") for i in range(size)],
                    ctx=ast.Load(),
                ),
            )  # put expanded args into tuple with the name of the original arg
            assign_stmt.col_offset = first_line_col_offset
            new_body.insert(0, assign_stmt)
        else:
            new_args.append(arg)
    func_def.args.args = new_args
    for i, line in enumerate(new_body):
        line.lineno = i + 1
    func_def.body = new_body
    expanded_func_code: str = ast.unparse(func_def)  # our result
    return expanded_func_code


def func_str_to_Callable(func_str: str) -> Callable:
    """Returns a function from a string of its source code."""
    d = {}
    exec(func_str, globals(), d)
    (new_func,) = d.values()  # expect only one value
    return new_func
