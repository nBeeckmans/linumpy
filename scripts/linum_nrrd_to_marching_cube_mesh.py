#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert a nrrd volume into a mesh (stl format) using marching cube
agorithm"""

import argparse
import sys

import numpy as np
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import marching_cubes
from skimage.transform import rescale
from stl import mesh
import nrrd


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("input", help="Full path to a 3D .nrrd file")
    p.add_argument("output", help="Full path to the .stl file")
    p.add_argument(
        "--resize",
        help="Resize factor to apply to the image "
        + "before running the marching cube algorithm",
        type=float,
        required=False,
    )

    p.add_argument(
        "--add-gaussian-filter",
        help="Apply a gaussian filter before running marching cube."
        + "This leads to a less precise render but removes many holes"
        + " from mesh",
        type=bool,
        required=False,
    )

    return p


def is_not_three_dimensions(image):
    number_of_dimensions = len(image.shape)
    if number_of_dimensions == 3:
        return True

    if has_unused_fourth_dimension(image):
        return True

    return False


def has_unused_fourth_dimension(image):
    return image.shape == 4 and image[3] == 1


def main():
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()

    # Load the data
    img, _ = nrrd.read(args.input)
    if is_not_three_dimensions(img.shape):
        print("NRRD file has to be a 3-dimension image")
        sys.exit(1)

    if has_unused_fourth_dimension(img.shape):
        img = img.squeeze()

    # This leads to less precision but also less holes in the mesh
    if args.add_gaussian_filter:
        img = gaussian(img)

    if args.resize:
        img = rescale(img, args.resize)

    # To separate foreground from background, I chose to use otsu.
    # The marching cube algorithm will use the threshold as its
    # level.
    threshold = threshold_otsu(img)

    verts, faces, _, _ = marching_cubes(img, level=threshold)

    # Recentering the vertices.
    centroid = np.mean(verts, axis=0)
    verts = verts - centroid

    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    # from : https://numpy-stl.readthedocs.io/_/downloads/en/latest/pdf/, Chapter 1, page 8
    for i, f in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = verts[f[j], :]

    stl_mesh.save(args.output)


if __name__ == "__main__":
    main()
