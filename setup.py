#! /usr/bin/env python
# -*- coding:Utf8 -*-

#--------------------------------------------------------------------------------------------------------------
# All necessary import:
#--------------------------------------------------------------------------------------------------------------
import os

try:
	import commands
except:
	import subprocess as commands

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

#--------------------------------------------------------------------------------------------------------------
# For adding support of pkg-config:
#--------------------------------------------------------------------------------------------------------------
def pkgconfig(*packages, **kw):
	flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
	for token in commands.getoutput("pkg-config --libs --cflags %s" % ' '.join(packages)).split():
		kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
	return kw

#--------------------------------------------------------------------------------------------------------------
# Sources File:
#--------------------------------------------------------------------------------------------------------------
gene_src = [ "InitialCond/Generation/generation_py.pyx", "InitialCond/Generation/generation.c"]
tree_src = [ "InitialCond/OctTree/tree_py.pyx", "InitialCond/OctTree/tree.c", "InitialCond/types.c"]

#, "InitialCond/Generation/gadget.c"

#--------------------------------------------------------------------------------------------------------------
# Compilation option:
#--------------------------------------------------------------------------------------------------------------
#	-> General:
opt                            = dict(include_dirs = ['.', 'include/'])
#opt                            = dict(include_dirs = ['/home/plum/.local/lib/python3.3/site-packages/Cython/Includes/', '.', 'include/'])

#	-> Tree Package:
tree_opt                       = opt

#	-> Generation Package:
gene_opt                       = pkgconfig("king")
gene_opt["include_dirs"]      += opt["include_dirs"]

#--------------------------------------------------------------------------------------------------------------
# Creation of Extension class:
#--------------------------------------------------------------------------------------------------------------
tree       = Extension("OctTree",  tree_src, **tree_opt)
generation = Extension("Generate", gene_src, **gene_opt)

#--------------------------------------------------------------------------------------------------------------
# Call the setup function:
#--------------------------------------------------------------------------------------------------------------
setup(
	name        = 'InitialCond',
	version     = '1.0',
	description = 'Python Module for generating initial condition for gadget simulation.',
	author      = 'Guillaume Plum',
	cmdclass    = {'build_ext': build_ext},
	packages    = ['InitialCond', 'InitialCond.Generation', 'InitialCond.OctTree'],
	ext_modules = [
		generation,
		tree
	]
)

#vim:spelllang=
