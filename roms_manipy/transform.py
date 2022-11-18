
"""Generic transform tools"""
from typing import Optional as _opt
import xesmf as xe
import xarray as xr
import numpy as np



def xesmf_regridder(ds_source : _opt[xr.core.dataset.Dataset],
                    ds_target : _opt[xr.core.dataset.Dataset],
                    rename_vars_source: _opt[dict] = None,
                    rename_dims_source: _opt[dict] = None,
                    rename_dims_target: _opt[dict] = None,
                    xesmf_regridder_args: _opt[tuple] =None):
    """Interpolation from source grid to target grid using xesmf.
    xesmf convention requires dimensions names to follow a standard, where
    longitude and latitude are named as 'lon' and 'lat'.

    Args:
        ds_source (xr.core.dataset.Dataset): dataset that will be interpolated
        ds_target (xr.core.dataset.Dataset): dataset containing the targeted grid
        rename_vars_source (dict, optional): rename your variables as {'name': rename}.
                                Defaults to None.
        rename_dims_source (dict, optional): rename your source dimensions as (for instance)
                                {'longitude':'lon', 'latitude':'lat'}. Defaults to None.
        rename_dims_target (dict, optional): rename your target dimensions as (for instance)
                                {'longitude':'lon', 'latitude':'lat'}. Defaults to None.

    Returns:
        xr.core.dataset.Dataset: Interpolated dataset
    """

    if rename_vars_source is None:
        rename_vars_source = {}
    if rename_dims_source is None:
        rename_dims_source = {}
    if rename_dims_target is None:
        rename_dims_target = {}
    if xesmf_regridder_args is None:
        xesmf_regridder_args = ('bilinear')

    # -- interpolation -- #
    # aviso
    ds_source = ds_source.copy()
    ds_target = ds_target.copy()

    ds_source = ds_source.rename_dims(rename_dims_source)
    ds_source = ds_source.rename_vars(rename_vars_source)
    ds_target = ds_target.rename(rename_dims_target)

    # horizontal interpolation from aviso to roms grid
    regridder = xe.Regridder(ds_source, ds_target, xesmf_regridder_args)
    interpolated = regridder(ds_source)  # interpolating

    return interpolated


def rotation2d(uvar, vvar, angle):
    """rotates vectors

    Args:
        uvar: horizontal direction component
        vvar: vertical direction component
        angle: rotation angle
    """
    urot = uvar * np.cos(angle) - vvar * np.sin(angle)
    vrot = uvar * np.sin(angle) + vvar * np.cos(angle)

    return urot, vrot
