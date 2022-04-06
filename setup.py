# -*- coding: utf-8 -*-
import os
import subprocess
import sys
from multiprocessing import cpu_count

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


def get_version() -> str:
  # https://packaging.python.org/guides/single-sourcing-package-version/
  with open(os.path.join("pie", "__init__.py"), "r") as f:
    init = f.read().split()
  return init[init.index("__version__") + 2][1:-1]


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):

  def __init__(self, name, sourcedir=""):
    Extension.__init__(self, name, sources=[])
    self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
  """cmake setup helper class"""

  def build_extension(self, ext):
    extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

    # required for auto-detection of auxiliary "native" libs
    if not extdir.endswith(os.path.sep):
      extdir += os.path.sep

    # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
    # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
    # from Python.
    cmake_args = [
      f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
      f"-DPYTHON_EXECUTABLE={sys.executable}",
      "-DCMAKE_BUILD_TYPE=Release",
    ]
    build_args = ["-j", f"{cpu_count()}"]

    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    subprocess.check_call(
      ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
    )
    subprocess.check_call(["make"] + build_args, cwd=self.build_temp)


def package_files(directory):
  paths = []
  for (path, _, filenames) in os.walk(directory):
    for filename in filenames:
      paths.append(os.path.join("..", path, filename))
  return paths


def get_description():
  with open("README.md", encoding="utf8") as f:
    return f.read()


setup(
  name="pie",
  version=get_version(),
  author="Jiayi Weng",
  author_email="trinkle23897@gmail.com",
  url="https://github.com/Trinkle23897/Fast-Poisson-Image-Editing",
  license="MIT",
  description="A fast poisson image editing algorithm implementation.",
  long_description=get_description(),
  long_description_content_type="text/markdown",
  packages=find_packages(exclude=["tests", "tests.*"]),
  package_data={"pie": ["pie*.so"]},
  entry_points={
    "console_scripts": ["pie=pie.cli:main"],
  },
  install_requires=[
    "opencv-python>=4.5",
    "numpy>=1.20",
  ],
  extras_require={
    "dev": [
      "pylint",
      "isort",
      "mypy",
      "tqdm",
    ],
  },
  ext_modules=[CMakeExtension("pie_core_openmp")],
  cmdclass={"build_ext": CMakeBuild},
  zip_safe=False,
  classifiers=[
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    "Development Status :: 3 - Alpha",
    # Indicate who your project is intended for
    "Intended Audience :: Science/Research",
    "Topic :: Software Development :: Libraries :: Python Modules",
    # Pick your license as you wish (should match "license" above)
    "License :: OSI Approved :: MIT License",
    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Operating System :: POSIX",
  ],
)