#! /usr/bin/env python
# -*- coding:Utf8 -*-

#--------------------------------------------------------------------------------------------------------------
# All necessary import:
#--------------------------------------------------------------------------------------------------------------
import os, sys, stat

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

def scandir(dir, files=[]):
	for file in os.listdir(dir):
		path = os.path.join(dir, file)
		if os.path.isfile(path) and path.endswith(".pyx"):
			files.append(path.replace(os.path.sep, ".")[:-4])
		elif os.path.isdir(path):
			scandir(path, files)
	return files

def makeExtension(extName):
	extPath = extName.replace(".", os.path.sep)+".pyx"
	return extPath
	#return Extension(
		#extName,
		#[extPath],
		#include_dirs = [libdvIncludeDir, "."],   # adding the '.' to include_dirs is CRUCIAL!!
		#extra_compile_args = ["-O3", "-Wall"],
		#extra_link_args = ['-g'],
		#libraries = ["dv",],
	#)

extNames = scandir("InitialCond")
extensions = [makeExtension(name) for name in extNames]
print(extNames, extensions, sep='\n')

#--------------------------------------------------------------------------------------------------------------
# Sources File:
#--------------------------------------------------------------------------------------------------------------
gene_src  = [ "InitialCond/Generation/cGeneration.pyx", "InitialCond/Generation/generation.c" ]
tree_src  = [ "InitialCond/Tree/cTree.pyx", "InitialCond/Tree/tree.c" ] #, "InitialCond/types.c" ]
types_src = [ "InitialCond/cTypes.pyx" ]

#, "InitialCond/Generation/gadget.c"

#--------------------------------------------------------------------------------------------------------------
# Compilation option:
#--------------------------------------------------------------------------------------------------------------
#	-> General:
opt                            = dict(include_dirs = ['.', 'include/'], extra_compile_args=["-std=c99"])
#opt                            = dict(include_dirs = ['/home/plum/.local/lib/python3.3/site-packages/Cython/Includes/', '.', 'include/'])

#	-> Tree Package:
tree_opt                       = opt.copy()

#	-> Types Package:
types_opt                      = opt.copy()

#	-> Generation Package:
gene_opt                       = pkgconfig("king")
gene_opt["include_dirs"]      += opt["include_dirs"]
if "extra_compile_args" in gene_opt:
	gene_opt["extra_compile_args"] += opt["extra_compile_args"]
else:
	gene_opt["extra_compile_args"] = opt["extra_compile_args"]

#--------------------------------------------------------------------------------------------------------------
# Creation of Extension class:
#--------------------------------------------------------------------------------------------------------------
tree       = Extension("InitialCond.Tree.cTree",  tree_src,  **tree_opt)
generation = Extension("InitialCond.Generation.cGeneration", gene_src,  **gene_opt)
types      = Extension("InitialCond.cTypes",    types_src, **types_opt)

#--------------------------------------------------------------------------------------------------------------
# Call the setup function:
#--------------------------------------------------------------------------------------------------------------
setup(
	name        = 'InitialCond',
	version     = '1.0',
	description = 'Python Module for generating initial condition for gadget simulation.',
	author      = 'Guillaume Plum',
	cmdclass    = {'build_ext': build_ext},
	#packages    = ['InitialCond' ], #, 'InitialCond.Generate', 'InitialCond.OctTree', 'InitialCond.Types'],
	ext_modules = [
		types,
		tree,
		generation
	]
)

#vim:spelllang=
