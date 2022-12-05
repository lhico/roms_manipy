from typing import Optional as _opt
import matplotlib.pyplot as plt
from matplotlib import figure as _figure
from matplotlib import axes as _axes
import xarray as _xr
import xcmocean

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
