#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 10:38:39 2022

@author: tobraun
"""

#------------------ PACKAGES ---------------------------#

import numpy as np
import scipy.spatial.distance as scidist
from scipy.signal import find_peaks

#------------------ FUNCTIONS ---------------------------#




def recflow_embed(ts, Ndim, delays, rr, theil, pthresh=None):
    """
    Embeds a univariate time series 'ts' based on recurrence flow.
    If the embedding should be optimal, prior information on the embedding dimension
    (e.g. false nearest neighbour based) is required to specify 'Ndim'.
    For the computation of recurrence plots, the vicinity threshold 
    needs to be specified by fixing a recurrence rate. A minimum temporal separation of
    time series values can be introduced by specifying a Theiler window.
     
        
    Parameters
    ----------    
        ts: (NxM) array (float)
            (float) time series 
        Ndim: int
            number of dimensions that should be considered
        delays: 1D array (int)
            selection of delays
        js: 1D array (int)
            selection of time series, see TDE()
        rr: float
            fixed recurrence rate to fix vicinity threshold
        theiler: int
            Theiler window (ignore by setting to None)
        pthresh: float
            minimum value for prominence of detected peaks

    Returns
    -------
        a_flow : 2D array (float)
            recurrence flow for each dimension and delay
        l_tau: 1D list (int)
            'Nd' optimal delay values
    
    Dependencies
    -------
        Uses a peak detection function from the Scipy package.
    
    """
    Ntau = delays.size
    l_tau = [0]
    a_flow = np.zeros((Ndim,Ntau))
    for i in np.arange(Ndim-1):
        ## Recurrence Flow
        tmp_flow = np.zeros(Ntau)
        for j in (range(Ntau)):
            tmp_tau = np.hstack([l_tau, delays[j]]).astype(int)
            tmp_rp2 = RP(ts, tmp_tau, js=np.zeros(tmp_tau.size), rr=rr, theiler=theil)
            # compute recurrence flow
            tmp_flow[j] = floodRP(tmp_rp2)
        a_flow[i,:] = tmp_flow
        
        ## Peak detection
        a_cand = find_peaks(tmp_flow, prominence=pthresh)[0]
        
        # identify highest peak
        if a_cand.size>0:
            idx = a_cand[np.argmax(tmp_flow[a_cand])]
            tau = delays[idx]
        else:
            print('No peaks could be detected! The signal might be stochastic.')
            tau = np.nan
        
        l_tau.append(tau)
    # output
    return a_flow, l_tau




def RFMD(x, y, delays, rr, theiler=None):
    """
    Computes the recurrence flow measure of dependence for two (univariate) time series x and y
    for the specified delays. For the computation of recurrence plots, the vicinity threshold 
    needs to be specified by fixing a recurrence rate. A minimum temporal separation of
    time series values for flow computation can be introduced by specifying a Theiler window.
     
        
    Parameters
    ----------    
        x: 1D array (float)
            first time series 
        y : 1D array (float)
            second time series
        delays: 1D array (int)
            selection of delays/lags between x and y.
        rr: float
            fixed recurrence rate to fix vicinity threshold
        theiler: int
            Theiler window (ignore by setting to None)

    Returns
    -------
        a_rfmd : 1D array (float)
            recurrence flow measure of dependence for the given delays
    
    """
    N = delays.size
    a_rfmd = np.zeros(N)
    k = 0
    for tau in delays:
        if tau < 0:
            ts1, ts2 = x[:tau], y[-tau:]
        elif tau == 0:
            ts1, ts2 = x, y
        else:
            ts1, ts2 = x[tau:], y[:-tau]
        
        # FLOW
        tmp_arr = np.vstack([ts1, ts2]).T
        tmp_rp = RP(tmp_arr, delays=None, js=None, rr=rr, theiler=theiler)
        a_rfmd[k] = 1-floodRP(tmp_rp)
        k += 1
    
    return a_rfmd
    
    


def recflow_from_ts(ts, delays, js, rr, theiler=None):
    """
    Computes the recurrence flow from a given uni/multivariate time series.
    The recurrence plot is based on time delay coordinates specified by 'delays' and 'js'.
    For the computation of recurrence plots, the vicinity threshold 
    needs to be specified by fixing a recurrence rate. A minimum temporal separation of
    time series values can be introduced by specifying a Theiler window.
     
        
    Parameters
    ----------    
        ts: (NxM) array (float)
            (float) time series 
        delays: 1D array (int)
            selection of delays
        js: 1D array (int)
            selection of time series, see TDE()
        rr: float
            fixed recurrence rate to fix vicinity threshold
        theiler: int
            Theiler window (ignore by setting to None)

    Returns
    -------
        rflow : float
            recurrence flow
    """
    a_rp = RP(ts, delays, js, rr, theiler=None)
    rflow = floodRP(a_rp, return_flowmat=False)
    return rflow



def floodRP(rp, return_flowmat=False):
    """
    Floods a recurrence plot along its diagonals. No turning allowed.
    The fluid stops at a recurrence (black pixel = 1).
    Only the upper triangular matrix is flooded (symmetry!).
    Either returns the flow matrix or the recurrence flow.
     
        
    Parameters
    ----------    
        rp: 2D array (int)
            binary recurrence matrix
        return_flowmat: bool
            return the flow matrix instead of recurrence flow value

    Returns
    -------
        outp : float/2D array (int)
            either recurrence flow or flow matrix
    """
    N = rp.shape[0]
    a_flow = np.zeros((rp.shape))
    # loop over rows
    for n in range(N):
        tmp_diag = np.diag(rp, n)
        tmp_occ = np.where(tmp_diag==1)[0]
        if tmp_occ.size > 0:
            block = tmp_occ[0]
        else:
            block = tmp_diag.size
        # flood
        t_idx = _nth_diag_indices(a_flow, n)
        a_flow[t_idx[0][:block], t_idx[1][:block]] = np.ones(block)

    if return_flowmat:
        outp = a_flow
    else:
        outp = 2*np.sum(a_flow)/(rp.size-rp.sum())
    
    return outp



def TDE(ts, taus, js):    
    """
    Perform an embedding with delays `taus` and time series stored in `s`, 
    specified by their indices `js`
     
    Parameters
    ----------    
    s : `numpy.ndarray` (N, M)
        Input time series of length `N`. This can be a multivariate set, consisting of `M`time series, which are stored in the columns.
    taus : `list` or `numpy.ndarray`
        Denotes what delay times will be used for constructing the trajectory for which the continuity statistic to all time series in
        `s` will be computed.
    js : `list` or `numpy.ndarray`
        Denotes which of the timeseries contained in `s` will be used for constructing the trajectory using the delay values stored in
        `taus`. `js` can contain duplicate indices.
    Returns
    -------
    Y : `numpy.ndarray` (N', d)
        The trajectory from the embedding of length `N' = N-sum(taus)` of dimension `d = len(taus)`.
    
    Notes
    -----
    The generalized embedding works as follows:
    `taus, js` are `list`'s (or `numpy.ndarray`'s) of length `d`, which also coincides with the embedding dimension. For example, imagine 
    input trajectory :math:`s = [x, y, z]` where :math:`x, y, z` are timeseries (the columns of `s`).
    If `js = (0, 2, 1)` and `taus = (0, 2, 7)` the created delay vector at each step `t` will be
    .. math:: (x(t), z(t+2), y(t+7))
    
    Source
    -----
    Taken from PECUZAL Python package by K. Hauke Kraemer:
    https://github.com/hkraemer/PECUZAL_python
    
    """
    
    assert np.amax(js) <= np.ndim(ts)
    if np.ndim(ts) == 1:
        assert js[0] == 0 
    N = len(ts) - np.amax(taus)
    data = np.empty(shape=(N,len(taus)))
    for (i, tau) in enumerate(taus):
        if np.ndim(ts) == 1:
            data[:,i] = ts[tau:(N+tau)]
        else:
            data[:,i] = ts[tau:(N+tau), js[i]]
    return data



def RP(ts, delays, js, rr, theiler=None):
    """
    Generates a recurrence plot from uni/multivariate time series 'ts' based on the
    embedding given by 'delays' and 'js'. The vicinity threshold 
    needs to be specified by fixing a recurrence rate. A minimum temporal separation of
    time series values can be introduced by specifying a Theiler window.
     
        
    Parameters
    ----------    
        ts: (NxM) array (float)
            (float) time series 
        delays: 1D array (int)
            selection of delays
        js: 1D array (int)
            selection of time series, see TDE()
        rr: float
            fixed recurrence rate to fix vicinity threshold
        theiler: int
            Theiler window (ignore by setting to None)

    Returns
    -------
        a_rp : 2D array (int)
            recurrence matrix
            
    Dependencies
    -------
        Uses the 'pdist' and 'squareform' functions from the Scipy package.
    """
    if delays is not None:
        a_emb = TDE(ts, delays, js)
    else:
        a_emb = ts
    a_dist = scidist.squareform(scidist.pdist(a_emb, metric='euclidean'), force='tomatrix')
    a_rp = np.where(a_dist < np.quantile(a_dist, rr), 1, 0)
    # apply theiler window by setting all LOI-neighbouring diagonals in the RP up to the theiler window to zero
    if theiler is not None:
        tmp_theiler = np.arange(theiler+1)
        for i in tmp_theiler:
            a_rp[_nth_diag_indices(a_rp, tmp_theiler[i])] = 0
    return a_rp



def _nth_diag_indices(arr, n):
    """ Extracts n-th diagonal of a NxM matrix.

    Parameters
    ----------
    arr : {array-like, float}, shape = [N, M]
        1D array of x-values
    n : {int}, value
        index of diagonal
        
    Returns
    -------
    rows, cols :  1D arrays (int)
        indices of n-th diagonal             
    """
    rows, cols = np.diag_indices_from(arr)
    if n < 0:
        return rows[-n:], cols[:n]
    elif n > 0:
        return rows[:-n], cols[n:]
    else:
        return rows, cols



