#!/usr/bin/env bash

# TODO: Find a better way to install SE3Transformer code!
# (Thought, could copy se3_transformer in the same directory as trip and then it isn't need to be installed?)
cd SE3Transformer && python setup.py install & cd ..