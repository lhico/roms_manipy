"""Roms post processing tool"""
from typing import Optional as _opt
import xarray as _xr
import xroms as _xroms
import roms_manipy.transform as _transform
import roms_manipy.visualize as _visualize
from matplotlib import figure as _figure
from matplotlib import axes as _axes
# import numpy as np
# import xesmf as xe
# import matplotlib
# import xcmocean


class ReadAdjust:
    """Adjusts roms file with grid information"""
    def __init__(self,
                fpath : str,
                fgrid : str = ''):
        self.fpath = fpath

        print(f"creating {self.__class__.__name__}.ds")

        self.dataset = _xroms.open_netcdf(fpath)

        if fgrid != '':
            self.dsgrid = _xr.open_dataset(fgrid)

    def set_variables(self, listvar : _opt[list] = None):
        """set variables of the grid into roms file

        Args:
            listvar (_opt[list], optional): _description_. Defaults to ['angle'].
        """

        if listvar is None:
            listvar=['angle']

        for i in listvar:
            self.dataset[i] = self.dsgrid[i]


class Transform(object):
    """Tranformation class"""
    def __init__(self,
                 dataset : _opt[_xr.core.dataset.Dataset],
                 grid : _opt[_xr.core.dataset.Dataset]=None):

        self.dataset = dataset

        if grid is not None:
            self.grid = grid


    def interpolate(self,
                    ds_source : _opt[_xr.core.dataset.Dataset],
                    rename_vars_source: _opt[dict] = None,
                    rename_dims_source: _opt[dict] = None,
                    rename_dims_target: _opt[dict] = None,
                    xesmf_regridder_args: _opt[tuple] =None):
        """Interpolation from source grid to object instance grid using xesmf.
        xesmf convention requires dimensions names to follow a standard, where
        longitude and latitude are named as 'lon' and 'lat'.

        Args:
            ds_source (xr.core.dataset.Dataset): dataset that will be interpolated
            rename_vars_source (dict, optional): rename your variables as {'name': rename}.
                                    Defaults to None.
            rename_dims_source (dict, optional): rename your source dimensions as (for instance)
                                    {'longitude':'lon', 'latitude':'lat'}. Defaults to None.
            rename_dims_target (dict, optional): rename your target dimensions as (for instance)
                                    {'longitude':'lon', 'latitude':'lat'}. Defaults to None.

        Returns:
            xr.core.dataset.Dataset: Interpolated dataset
        """
        interpolated = _transform.xesmf_regridder(ds_source, self.grid,
                                rename_vars_source=rename_vars_source,
                                rename_dims_source=rename_dims_source,
                                rename_dims_target=rename_dims_target,
                                xesmf_regridder_args=xesmf_regridder_args)
        return interpolated



    def rotation(self,
                ustr: str,
                vstr: str,
                ustrnew: str,
                vstrnew: str, 
                angle: _opt[float] = None,
                rot_type: _opt[str] = None,
                geostrophic: bool = False):
        """Rotates vector and set them to rho grid in the dataset

        Args:
            ustr (str): variable name (horizontal direction)
            vstr (str): variable name (vertical direction)
            ustrnew (str): variable name (horizontal direction
            vstrnew (str): variable name (vertical direction)
            angle (float, optional): rotation angle. Defaults to angle variable in the dataset.

        """

        if rot_type is None:
            rot_type='2d'

        if angle is None:
            angle  = self.dataset['angle']

        # rotation handler
        rotation = {
            '2d': _transform.rotation2d
        }


        if geostrophic:
            uvar = self.dataset.xroms.ug
            vvar = self.dataset.xroms.vg
        else:
            uvar = self.dataset[ustr]
            vvar = self.dataset[vstr]


        urot = _xroms.to_rho(uvar, self.dataset.xroms.grid)
        vrot = _xroms.to_rho(vvar, self.dataset.xroms.grid)

        urot, vrot = rotation[rot_type](urot, vrot, angle)


        self.dataset[ustrnew] = urot
        self.dataset[vstrnew] = vrot

        print(f"'{ustr} and {vstr} included in \
                {self.__class__.__name__}.dataset")



class Visualize(object):
    """Visualization class
    """    
    def __init__(self, dataset : _opt[_xr.core.dataset.Dataset]):
        self.dataset = dataset

    def plot(self,
             varb   : str,
             icoords: _opt[dict]=None,
             fig    : _opt[_figure.Figure]=None,
             ax     : _opt[_axes._subplots.Axes]=None,
             ptype  : str ='pcolor',
             plotkwargs: _opt[dict] =None):

        """Generic 2D plot which encapsulates xarray/matplotlib plot
        capabilities.

        Args:
            varb (_type_): _description_
            ds1 (_type_, optional): _description_. Defaults to None.
            itime (_type_, optional): _description_. Defaults to None.
            time (_type_, optional): _description_. Defaults to None.
            fig (_type_, optional): _description_. Defaults to None.
            ax (_type_, optional): _description_. Defaults to None.
            ptype (str, optional): _description_. Defaults to 'pcolor'.
            plotkwargs (dict, optional): _description_. Defaults to {}.

        Returns:
            _type_: _description_
        """
        fig, ax, pt = _visualize.plot(self.dataset,
                                     varb,
                                     icoords=icoords,
                                     fig=fig,
                                     ax=ax,
                                     ptype=ptype,
                                     plotkwargs=plotkwargs)

        return fig, ax, pt


    def plot_cf(self,
                varb: str,
                X  : _opt[int]=None,
                Y  : _opt[int]=None,
                Z  : _opt[int]=None,
                T  : _opt[int]=None,
                fig: _opt[_figure.Figure]=None,
                ax : _opt[_axes._subplots.Axes]=None,
                ptype: str ='pcolor',
                plotkwargs: _opt[dict]=None):
        """Generic 2D plot which encapsulates xroms/xcmocean plot
        capabilities.

        Args:
            varb (string): _description_
            ds1 (xarray.core.dataset.Dataset, optional): dataset created
                    with _xroms. Defaults to None.
            X (int, optional): CF Axes. Defaults to None.
            Y (int, optional): CF Axes. Defaults to None.
            Z (int, optional): CF Axes. Defaults to None.
            T (int, optional): CF Axes. Defaults to None.
            fig (figure.Figure, optional): Defaults to None.
            ax (axes._subplots.AxesSubplot, optional): Defaults to None.
            ptype (str, optional): plot options ('pcolor', 'contour', 'contourf).
                                                                  Defaults to 'pcolor'.
            plotkwargs (dict, optional): dictionary with keyword arguments
                        for the plot. Two coordinates must be specified:
                                    - 2D map: {'x':'longitude', 'y':'latitude, ..}
                                    - section: {'x':'longitude', 'y':'vertical, ...}
                                               {'x':'latitude', 'y':'vertical, ...}

                        Defaults to {}.

        Raises:
            ValueError: Selected dimensions are not 2D

        Returns:
            (figure, ax, plot instance)_
        """
        fig, ax, pt = _visualize.plot_cf(
            self.dataset, varb, X=X, Y=Y, Z=Z, T=T,
                    fig=fig,ax=ax,ptype=ptype,plotkwargs=plotkwargs)

        return fig, ax, pt


    def cross_plot(self,
                   var1 : str,
                   var2 : str,
                   c    : str,
                   icoords: _opt[dict]=None,
                   fig    : _opt[_figure.Figure]=None,
                   ax     : _opt[_axes._subplots.Axes]=None,
                   ptype  : str ='scatter',
                   plotkwargs: _opt[dict]=None):
        """Compares data from two plots using  matplotlib.pyplot's scatter/plot

        Args:
            var1 (str): variable name (x-coordinate)_
            var2 (str): variable name (y-coordinate)
            c (str, optional): variable name (for color/size reference in sactter plot).
                            Defaults to None.
            icoords (dict, optional): .isel coordinates dictionary in coordinates.
                                Defaults to None.
            fig (_figure.Figure, optional): _description_. Defaults to None.
            ax (_axes._subplots.Axes, optional): _description_. Defaults to None.
            ptype (str, optional): _description_. Defaults to 'scatter'.
            plotkwargs (dict, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        fig, ax, sc = _visualize.cross_plot(self.dataset, var1, var2,
                                           c=c, icoords=icoords,
                                           fig=fig,ax=ax,
                                           ptype=ptype,
                                           plotkwargs=plotkwargs)

        return fig, ax , sc

