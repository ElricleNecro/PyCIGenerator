#! /usr/bin/env python
# -*- coding:Utf8 -*-

#--------------------------------------------------------------------------------------------------------------
# All necessary import:
#--------------------------------------------------------------------------------------------------------------
import os, sys, stat

try:
	import King
except ImportError:
	print("It seems that the python Binding for the king librairies is not installed on your system.\n You should install it as it is a necessary dependancy!")
	sys.exit(-1)

try:
	import commands
except:
	import subprocess as commands

from distutils.core import setup
#from distutils.extension import Extension
from Cython.Distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

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

def makeExtension(extName, test=False, **kwargs):
	extPath = [ extName.replace(".", os.path.sep)+".pyx" ]
	cfile   = extName.split(".")
	dir     = os.path.join(*cfile[:-1])
	cfile   = "c" + cfile[-1] + ".c"
	cfile   = os.path.join(dir, cfile)

	if os.path.isfile(cfile):
		extPath += [ cfile ]

	opt_dict = dict(
		include_dirs = ["."],   # adding the '.' to include_dirs is CRUCIAL!!
		extra_compile_args = ["-std=c99"],
		extra_link_args = ['-g'],
		libraries = [],
		cython_include_dirs = [os.path.join(os.getenv("HOME"), '.local/lib/python' + ".".join([ str(a) for a in sys.version_info[:2]]) + '/site-packages/Cython/Includes')]
	)

	for key in kwargs.keys():
		if key in opt_dict:
			opt_dict[ key ] += kwargs[ key ]
		else:
			opt_dict[ key ] = kwargs[ key ]

	if test:
		return extPath, opt_dict
	else:
		return Extension(
			extName,
			extPath,
			**opt_dict
		)

extNames = scandir("InitialCond")

#extensions = [makeExtension(name, include_dirs = [ "include/" ]) for name in extNames]
extensions = []
for name in extNames:
	if "Generation" in name:
		opt = pkgconfig("king")
		opt["cython_include_dirs"] = [ King.get_include() ]
		if "include_dirs" in opt:
			opt["include_dirs"] += [ "include/" ]
		else:
			opt["include_dirs"]  = [ "include/" ]
		extensions.append( makeExtension(name, **opt) )
	else:
		extensions.append( makeExtension(name, include_dirs = [ "include/" ]) )

#--------------------------------------------------------------------------------------------------------------
# Packages names:
#--------------------------------------------------------------------------------------------------------------
packages = [ 'InitialCond', 'InitialCond.Tree', 'InitialCond.Generation', 'InitialCond.Gadget' ]

#--------------------------------------------------------------------------------------------------------------
# Call the setup function:
#--------------------------------------------------------------------------------------------------------------
setup(
	name        = 'InitialCond',
	version     = '1.0',
	description = 'Python Module for generating initial condition for gadget simulation.',
	author      = 'Guillaume Plum',
	cmdclass    = {'build_ext': build_ext},
	packages    = packages,
	data_files  = [
		('bin', ['ci_py'])
	],
	ext_modules = cythonize( extensions ,
			 include_path = [ '.', King.get_include() ]
	)
)

#vim:spelllang=
