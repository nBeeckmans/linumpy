#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from linumpy.io.test_data import get_data


def test_help(script_runner):
    ret = script_runner.run(["linum_convert_nrrd_to_marching_cube_mesh.py", "--help"])
    assert ret.success


def test_execution(script_runner, tmp_path):
    input = get_data("mosaic_3d_nrrd")
    output = tmp_path / "output.stl"

    ret = script_runner.run(["linum_convert_nifti_to_nrrd.py", input, output])
    assert ret.success
