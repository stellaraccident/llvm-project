# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Sequence

import os

_this_dir = os.path.dirname(__file__)


def get_lib_dirs() -> Sequence[str]:
  """Gets the lib directory for linking to shared libraries.

  On some platforms, the package may need to be built specially to export
  development libraries.
  """
  return [_this_dir]


def get_include_dirs() -> Sequence[str]:
  """Gets the include directory for compiling against exported C libraries.

  Depending on how the package was build, development C libraries may or may
  not be present.
  """
  return [os.path.join(_this_dir, "include")]


# Eagerly load the main _mlir extension and configure it with our actual
# package location.
def _configure_native_extension():
  import logging
  from . import _mlir
  mlir_libs_suffix = "._mlir_libs"
  if __package__.endswith(mlir_libs_suffix):
    # Form a new relative name to the .dialects package, which should be
    # a peer of this _mlir_libs package.
    dialects_package = __package__[0:-len(mlir_libs_suffix)] + ".dialects"
    logging.debug("MLIR dialects search package: %s", dialects_package)
    _mlir.globals.append_dialect_search_prefix(dialects_package)
  else:
    logging.warning("Expected package to be of form *._mlir_libs but got %s",
                    __package__)


_configure_native_extension()
