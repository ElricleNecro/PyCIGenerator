#!/usr/bin/env bash

export CC="gcc -std=c99"
export CFLAGS="-Wno-unused-result -DDYNAMIC_ANNOTATIONS_ENABLED=1 -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -march=x86-64 -mtune=generic -O2 -pipe -fstack-protector --param=ssp-buffer-size=4"

python setup.py build
python setup.py install --user

