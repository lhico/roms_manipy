from typing import Optional as _opt
import matplotlib.pyplot as plt
from matplotlib import figure as _figure
from matplotlib import axes as _axes
import cartopy.crs as ccrs
import xarray as _xr
import numpy as np
import xcmocean

try:
    from sectionate.section import create_section, create_section_composite
except:
    raise ValueError("Please, install sectionate, otherwise Transect will be not available.")


def fig_ax_ok(fig: _opt[_figure.Figure],
              ax : _opt[_axes._subplots.Axes]):
    """Creates fig and ax if they are not defined.
    fig and ax should be created with matplotlib.pyplot

    Args:
        fig (matplotlib.figure.Figure)     : figure instance
        ax (matplotlib.axes._subplots.Axes): axes instance



    Returns:
        _type_: _description_
    """    
    # control figure
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
    elif (fig is None) and (ax is not None):
        raise IOError('ax should not be empty')
    elif (fig is not None) and (ax is None):
        raise IOError('fig should not be empty')
    else:
        pass
    return fig, ax


def cross_plot(ds1  :_opt[_xr.core.dataset.Dataset],
               var1 : str,
               var2 : str,
               c      : _opt[str] =None,
               icoords: _opt[dict]=None,
               fig    : _opt[_figure.Figure]=None,
               ax     : _opt[_axes._subplots.Axes]=None,
               ptype  : str ='scatter',
               plotkwargs: _opt[dict]=None):
    """Compares data from two plots using  matplotlib.pyplot's scatter/plot

    Args:
        ds1 (_xr.core.dataset.Dataset): xarray dataset
        var1 (str): variable name (x-coordinate)_
        var2 (str): variable name (y-coordinate)
        c (str, optional): variable name (for color/size reference in sactter plot).
                           Defaults to None.
        icoords (dict, optional): .isel coordinates dictionary in coordinates.
                            Defaults to None.
        fig (matplotlib.figure.Figure, optional): _description_. Defaults to None.
        ax (matplotlib.axes._subplots.Axes, optional): _description_. Defaults to None.
        ptype (str, optional): _description_. Defaults to 'scatter'.
        plotkwargs (dict, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    
    dict_plot ={
        'scatter'  : lambda x : x.scatter,
        'plot'     : lambda x : x.plot,
    }

    fig, ax = fig_ax_ok(fig, ax)
    ds1 = ds1.isel(**icoords)

    if c is not None:
        sc = dict_plot[ptype](ax)(ds1[var1].values, ds1[var2].values, c=ds1[c].values, **plotkwargs)
    else:
        sc = dict_plot[ptype](ax)(ds1[var1].values, ds1[var2].values, **plotkwargs)
    return fig, ax, sc


def plot_cf(ds1: _opt[_xr.core.dataset.Dataset],
            varb: str,
            X  : _opt[int]=None,
            Y  : _opt[int]=None,
            Z  : _opt[int]=None,
            T  : _opt[int]=None,
            fig: _opt[_figure.Figure]=None,
            ax : _opt[_axes._subplots.Axes]=None,
            ptype: str ='pcolor',
            plotkwargs: _opt[dict] =None):
    """Generic 2D plot encapsulating xroms/xcmocean plot
    capabilities. Set ONLY two optional variables in X,Y,Z,Y.

    Args:
        varb (string): _description_
        ds1 (xarray.core.dataset.Dataset, optional): dataset created
                with _xroms. Defaults to None.
        X (int, optional): CF Axes. Defaults to None.
        Y (int, optional): CF Axes. Defaults to None.
        Z (int, optional): CF Axes. Defaults to None.
        T (int, optional): CF Axes. Defaults to None.
        fig (matplotlib.figure.Figure, optional): Defaults to None.
        ax (matplitlib.axes._subplots.AxesSubplot, optional): Defaults to None.
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

    # handler for plotting functions
    dict_plot ={
        'pcolor'  : lambda x : x.cmo.cfpcolormesh,
        'contour' : lambda x : x.cmo.cfcontour,
        'contourf': lambda x : x.cmo.cfcontourf,
    }

    # selecting not None CF
    cfkw = dict(X=X, Y=Y, Z=Z, T=T)
    cfkw = {k:cfkw[k] for k in cfkw if cfkw[k] is not None}
    ndim = len(cfkw)  # check for dimensions

    # guarantee axes are a 2D
    if ndim != 2:
        raise ValueError(f'Please select dimension indexes (X,Y,Z,Y) such as ' +
                            f'{self.__class__.__name__}.ds.{varb} is 2D.')

    # create fig an ax if not defined
    fig, ax = fig_ax_ok(fig, ax)

    # cutting data
    ds1 = ds1[varb].cf.isel(**cfkw)

    ds1 = ds1.where(~ds1.isnull(), drop=True)

    # handles different plot types and their arguments
    # this is equivalent to ds1.cmo.cfpcolormesh(ax=ax, **plotkwargs)
    # if ptype == 'pcolor'
    pt = dict_plot[ptype](ds1)(ax=ax, **plotkwargs)

    return fig, ax, pt



def plot(ds1    : _opt[_xr.core.dataset.Dataset],
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
    dict_plot ={
        'pcolor'  : lambda x : x.plot,
        'contour' : lambda x : x.plot.contour,
        'contourf': lambda x : x.plot.contourf,
    }

    fig, ax = fig_ax_ok(fig, ax)


    ds1 = ds1[varb].isel(icoords)
    ndim = ds1.ndim
    if ndim != 2:
      raise ValueError(f'Please select dimension indexes (X,Y,Z,Y) such as ' +
                            f'{self.__class__.__name__}.ds.{varb} is 2D.')

    pt = dict_plot[ptype](ds1)(ax=ax, **plotkwargs)

    return fig, ax, pt


class Transect(object):
    """
    Example usage
    --------------    
    >>> lons = [-46.37016667, -46.13716667, -45.83833333, -45.59666667]
    >>> lats = [-24.16083333, -24.43333333, -24.781     , -25.06333333]
    >>> ds = xroms.open_netcdf('roms_avg.nc', chunks={'ocean_time': 50})
    >>> transect = Transect(ds, xstr='lon_rho', ystr='lat_rho')
    >>> transect.create_sections(lons, lats, varbs=['temp', 'salt'])
    >>> transect._plot(varb='temp', x='latitude', y='vertical', cmap=cmo.thermal)

    """

    def __init__(self, data, xstr='lon_rho', ystr='lat_rho'):
        """
        TODO: add docstring adequadamente
        xstr,ystr: lon_rho,lat_rho, but also could be lon_psi,lat_psi depending on the 
                   variable you are working with
        """
        self.data = data
        # extracting numerical grid
        self.lon = self.data[xstr]
        self.lat = self.data[ystr]

    ##################################################################################
    ##---- functions to create a cross section based on two given coordinates ------##
    ##################################################################################
    def create_transect(self, begin=None, final=None, lons=None, lats=None, varbs=None, 
                              times=None, find=True, forced_limits=False):
        """
            todo: documentation
        """
        if begin and final:
            print('Searching the nearest indexes for a straight line between two coordinates ...')
            # initially, we search for the nearest grid cells to a straight line between
            # the initial and final coordinates provided
            lon0, lat0 = begin
            lon1, lat1 = final

            self.lons = [lon0, lon1]
            self.lats = [lat0, lat1]

            if find == True:
                self._find_indexes(lon0, lat0,
                                lon1, lat1,
                                forced_limits=forced_limits)
        elif lons.any() and lats.any():
            print('Searching the nearest indexes for each location provided ...')
            self.lons, self.lats = lons, lats
            
            # build transect from multiples profiles' locations
            if find == True:
                self._find_indexes_composite(lons, lats)

        # second step is to extract the vertical profiles of each location found
        if type(varbs) == str:
            print(f'extracting transect for {varbs}')
            # just extract a DataArray
            self._extract_profiles(varb=varbs)
            
        elif type(varbs) == list:
            for k,varb in enumerate(varbs):
                print(f'extracting transect for {varb}')
                # extract section
                self._extract_profiles(varb=varb)
                
                if k == 0:
                    # if it is the first variable, then we must convert a DataArray 
                    # to Dataset, and then save the other variables
                    self.ds = self.ds_section.to_dataset()
                else:
                    # save the transect into an already existent Dataset
                    self.ds[varb] = self.ds_section
        else:
            raise ValueError(f"Type of varbs not available: {type(varbs)}")
        
        # TODO: implement temporal search
        # third, and optional, extract profiles temporally closest to a list of datetimes
        # self._get_temporal_profiles(times) --- UNDER CONSTRUCTION

    # find transect based on a list of segments, ideal to reconstruct cross sections from
    # CTD profiles' locations
    def _find_indexes_composite(self, lons, lats):
        plt.ioff()
        
        isec,jsec,xsec,ysec = create_section_composite(self.lon, self.lat,
                                                        lons, lats)
        # save the locations found
        self.ds_locs = xr.Dataset()
        self.ds_locs['lon'] = xr.DataArray(data=xsec, dims=('location'))
        self.ds_locs['lat'] = xr.DataArray(data=ysec, dims=('location'))
        self.isec = isec
        self.jsec = jsec

        # create the begin and the end of the transect in latlon pairs
        self.begin_transect = (ysec[0], xsec[0])
        self.final_transect = (ysec[-1], xsec[-1])

    # find transect based on a begin and final location. It cross a straight line and find
    # the nearests cells to it
    def _find_indexes(self, lon0, lat0, lon1, lat1, forced_limits=False):
        """
            Given an inital and final geographical location, creates a best fit linear
            transect and found all values near this transect in a numerical grid domain.
            This function uses the sectionate code, developed by Raphael Dussin, that can
            be found at:
            https://github.com/raphaeldussin/sectionate/blob/master/sectionate/section.py
        """
        plt.ioff() # in case plt.ion() is active

        isec,jsec,xsec,ysec = create_section(self.lon, self.lat,
                                             lon0, lat0,
                                             lon1, lat1)
        # save the locations found
        self.ds_locs = xr.Dataset()
        self.ds_locs['lon'] = xr.DataArray(data=xsec, dims=('location'))
        self.ds_locs['lat'] = xr.DataArray(data=ysec, dims=('location'))
        self.isec = isec
        self.jsec = jsec

        if forced_limits:
            # if activated, then we force the limits of the points found
            self.ds_locs = self.ds_locs.where((self.ds_locs['lon'] >= min([lon0, lon1]))
                                             & (self.ds_locs['lon'] <= max([lon0, lon1]))
                                             & (self.ds_locs['lat'] <= max([lat0, lat1]))
                                             & (self.ds_locs['lat'] <= max([lat0, lat1])),
                                             drop=True)

        # create the begin and the end of the transect in latlon pairs
        self.begin_transect = (ysec[0], xsec[0])
        self.final_transect = (ysec[-1], xsec[-1])

    # extract vertical profiles for a given variable (varb)
    def _extract_profiles(self, varb):
        """
        """
        
        # get list of locations
        sec_lons, sec_lats = self.ds_locs['lon'].values, self.ds_locs['lat'].values
        
        # TODO: optimize processing. Taking too long 'cause of this for loop
        # extract each profile
        list_ds_section = []
        for lon0,lat0 in zip(sec_lons, sec_lats):
            profile = self.data[varb].xroms.sel2d(lon0, lat0)
            list_ds_section.append(profile)
            
        self.ds_section = xr.concat(list_ds_section, dim='profiles')

    # TODO: cada tempo deve estar associado à uma estação, então a solução não é trivial
    # find the closest model profile to a given list of datetimes, to recreate the conditions
    # which the observed transect was interpolated
    def _get_temporal_profiles(self, times):
        """
        """
        if hasattr(self, 'ds_section'):
            list_profiles = []
            for t in times:
                profile = self.ds_section.sel(ocean_time=t, method='nearest')
                list_profiles.append(profile)
            
            # build final transect
            self.transect = xr.concat(list_profiles, dim='profiles')
    
    # simple data visualization 
    def _plot(self, varb=None, T=-1, x='latitude', y='vertical', figsize=(10,6), cmap='magma', invertx=True):
        """
        """
        plt.ion()
        
        self.fig = plt.figure(figsize=figsize)

        self.ax_cross = self.fig.add_subplot(1,2,1)
        self.ax_trans = self.fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
        
        # plot transect
        if varb:
            self.section = self.ds[varb].cf.isel(T=T)
        else:
            self.section = self.ds_section.cf.isel(T=T)
            
        self.section.where(~self.section.isnull(), drop=True).cf.plot(ax=self.ax_cross, x=x, y=y, cmap=cmap)
        
        # matplotlib plots the vertical section creating the x axis from low to high. Sometimes, we need
        # the opposite. For that, activate invertx
        if invertx == True:
            self.ax_cross.invert_xaxis()
        
        # plot the transect location
        self.ax_trans.coastlines('10m')

        self.ax_trans.plot(self.lon, self.lat, 'k', alpha=.1)
        self.ax_trans.plot(self.lon.T, self.lat.T, 'k', alpha=.1)

        self.ax_trans.set_xlim(np.nanmin(self.lons)-.5, np.nanmax(self.lons)+.5)
        self.ax_trans.set_ylim(np.nanmin(self.lats)-.5, np.nanmax(self.lats)+.5)

        self.ax_trans.scatter(self.ds_locs['lon'], self.ds_locs['lat'], s=20,
                   marker='o', c='red', label='Virtual profiles')
        self.ax_trans.scatter(self.lons, self.lats, s=20, marker='s', c='green', label='Real profiles', zorder=10)

        self.ax_trans.legend()
        plt.tight_layout()
        plt.show()
