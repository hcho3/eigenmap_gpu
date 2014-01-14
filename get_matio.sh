#!/bin/bash

MATIO_URL=\
http://sourceforge.net/projects/matio/files/matio/1.5.2/matio-1.5.2.tar.gz
MATIO_VER=matio-1.5.2
INST_DIR=$PWD/matio

wget $MATIO_URL
tar xvf $MATIO_VER.tar.gz
mv $MATIO_VER matio
cd matio
./configure --prefix=$INST_DIR
make && make install
