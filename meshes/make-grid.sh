#!/bin/bash

set -e

gmsh -2 -algo pack spheric-needles-1.geo -o spheric-needles-1.msh
