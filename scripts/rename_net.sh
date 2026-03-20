#!/bin/bash

# Renames file to YaneuraOu NN binary format.
name=nn-$(sha256sum $1 | cut -c1-12).bin
echo ${name}
mv $1 ${name}
