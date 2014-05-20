
import numpy
from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext

setup(
    name = "climf",
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("climf_fast", ["climf_fast.pyx"],
    						 libraries=["m"],
    						 include_dirs=[numpy.get_include()])],
    version = "0.1",
    author = "Mark Levy",
    author_email = "??",
    description = ("Implementation of CLiMF"),
    license = "??",
    keywords = "",
    url = "https://github.com/gamboviol/climf",
    packages=[],
    install_requires=[],
    long_description="Implementation of CLiMF",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities"
    ],
)
