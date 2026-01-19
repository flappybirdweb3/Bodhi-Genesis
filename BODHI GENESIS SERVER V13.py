#!/usr/bin/env python3
“””
[BODHI] BODHI GENESIS SERVER V13 - ADVANCED PHASE 2
V12 Complete + Kalman Filter + Monte Carlo + Polynomial Features

PHASE 1 (V12):

- Mamba V10 (73.3% accuracy) - 35% weight
- BiLSTM (74.2% accuracy) - 40% weight
- Transformer (66.9% accuracy) - 25% weight
- Meta-Labeler (LightGBM - trained on real trades)

PHASE 2 (V13 - NEW):

- Kalman Filter: Denoise RSI/ADX signals, reduce false signals
- Monte Carlo: Risk simulation, estimate win probability
- Polynomial Features: Feature engineering (RSI^2, RSI*ADX interactions)

TU HOP NHAT - NGUYEN TAC BAT DI BAT DICH:
THIEN (D1) = Troi  -> KHONG BAO GIO danh nguoc!
DIA (H4)   = Dat   -> Nen tang vung chac
NHAN (H1)  = Nguoi -> Momentum xac nhan
THOI (M15) = Timing -> Entry chinh xac

V6 RULES:

- RSI < 45 = BUY (pullback trong uptrend)
- RSI > 55 = SELL (pullback trong downtrend)
- ADX > 20 = Co trend
- 3 loss lien tiep = Cooldown 4h

FULL PIPELINE V13:
Signal -> Kalman Denoise -> V6 Check -> Polynomial Features -> Ensemble
-> Meta-Labeler -> Monte Carlo Risk -> PPO Risk -> Trade -> Karma

Usage:
uvicorn bodhi_server_v13_advanced:app –host 0.0.0.0 –port 9998 –reload
python bodhi_server_v13_advanced.py –port 9998
“””

import os
import sys
import json
import csv
import logging
import threading
import schedule
import time as time_module
from datetime import datetime, timedelta
from threading import Lock
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# Shadow Portfolios

from shadow_portfolio_lightweight import ShadowPortfolioManager

# FastAPI

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from functools import lru_cache

# PyTorch

try:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Categorical
TORCH_AVAILABLE = True
except ImportError:
TORCH_AVAILABLE = False
print(”[WARN] PyTorch not found - AI features disabled”)

# ======================================================================

# CONFIG

# ======================================================================

VERSION = “13.1 (Phase 2+3: Kalman + MC + Poly + Sentiment + Kelly + VolumeProfile + DBSCAN)”

# ======================================================================

# NUMBA JIT OPTIMIZATION (CPU-only, no GPU)

# ======================================================================

try:
import numba
from numba import jit, prange
NUMBA_AVAILABLE = True
# Logger message printed after logger initialized
except ImportError:
NUMBA_AVAILABLE = False
# Logger message printed after logger initialized

# ======================================================================
#
# XGBOOST IMPORT (for Retrain Engine)
#
# ======================================================================

try:
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
XGBOOST_AVAILABLE = True
except ImportError:
XGBOOST_AVAILABLE = False
if NUMBA_AVAILABLE:
@jit(nopython=True, cache=True)
def _monte_carlo_simulate_single(prices, returns_mean, returns_std, n_steps):
“”“Single Monte Carlo simulation (JIT compiled)”””
current_price = prices[-1]
max_price = current_price
min_price = current_price

```
    for _ in range(n_steps):
        ret = np.random.normal(returns_mean, returns_std)
        current_price *= (1 + ret)
        if current_price > max_price:
            max_price = current_price
        if current_price < min_price:
            min_price = current_price
    
    return current_price, max_price, min_price

@jit(nopython=True, parallel=True, cache=True)
def monte_carlo_fast(prices, returns_mean, returns_std, n_sims, n_steps, 
                     tp_pips, sl_pips, pip_value):
    """Fast Monte Carlo with parallel processing"""
    current_price = prices[-1]
    tp_price = current_price + (tp_pips * pip_value)
    sl_price = current_price - (sl_pips * pip_value)
    
    wins = 0
    total_profit = 0.0
    
    for i in prange(n_sims):
        final_price, max_price, min_price = _monte_carlo_simulate_single(
            prices, returns_mean, returns_std, n_steps
        )
        
        if max_price >= tp_price:
            wins += 1
            total_profit += tp_pips
        elif min_price <= sl_price:
            total_profit -= sl_pips
    
    win_probability = wins / n_sims
    avg_profit = total_profit / n_sims
    
    return wins, total_profit, avg_profit, win_probability

@jit(nopython=True, cache=True)
def kalman_filter_array_fast(measurements, Q=0.001, R=0.01):
    """Apply Kalman filter to array (JIT compiled)"""
    n = len(measurements)
    filtered = np.empty(n)
    
    x = measurements[0]
    P = 1.0
    
    for i in range(n):
        # Predict
        x_pred = x
        P_pred = P + Q
        
        # Update
        y = measurements[i] - x_pred
        S = P_pred + R
        K = P_pred / S
        x = x_pred + K * y
        P = (1 - K) * P_pred
        
        filtered[i] = x
    
    return filtered

@jit(nopython=True, cache=True)
def find_swing_highs_fast(prices, window=5):
    """Find swing highs (JIT compiled)"""
    swings = []
    n = len(prices)
    
    for i in range(window, n - window):
        is_swing = True
        for j in range(i - window, i + window + 1):
            if prices[j] > prices[i]:
                is_swing = False
                break
        if is_swing:
            swings.append(prices[i])
    
    return np.array(swings)

@jit(nopython=True, cache=True)
def find_swing_lows_fast(prices, window=5):
    """Find swing lows (JIT compiled)"""
    swings = []
    n = len(prices)
    
    for i in range(window, n - window):
        is_swing = True
        for j in range(i - window, i + window + 1):
            if prices[j] < prices[i]:
                is_swing = False
                break
        if is_swing:
            swings.append(prices[i])
    
    return np.array(swings)

pass  # logger.info("✅ Numba JIT functions compiled")
```

else:
# Dummy functions if Numba not available
def monte_carlo_fast(*args, **kwargs):
raise NotImplementedError(“Numba not installed”)
def kalman_filter_array_fast(*args, **kwargs):
raise NotImplementedError(“Numba not installed”)
def find_swing_highs_fast(*args, **kwargs):
raise NotImplementedError(“Numba not installed”)
def find_swing_lows_fast(*args, **kwargs):
raise NotImplementedError(“Numba not installed”)
DEFAULT_PORT = 9998

DATA_DIR = “./bodhi_data”
LOGS_DIR = “./bodhi_logs”
MODELS_DIR = “./bodhi_models”

# ═══════════════════════════════════════════════════════════════════════

# 12 PAIRS CONFIGURATION - Updated for 3 trades/day target

# ═══════════════════════════════════════════════════════════════════════

SYMBOLS = [
# EUR Cluster (London session)
‘EURUSD’, ‘EURGBP’, ‘EURJPY’,

```
# GBP Cluster (London session)
'GBPUSD', 'GBPJPY',

# USD Majors (Multi-session)
'USDJPY', 'USDCAD', 'AUDUSD',

# Commodities & Indices
'XAUUSD', 'XAGUSD', 'US30',

# Oceania
'NZDUSD'
```

]

SIGNAL_NAMES = {0: ‘SELL’, 1: ‘HOLD’, 2: ‘BUY’}

# Retrain config

RETRAIN_MIN_RECORDS = 1000
RETRAIN_DAY = 6  # Sunday
RETRAIN_HOUR = 6  # 6 AM

# ======================================================================

# V6 TỨ HỢP NHẤT CONFIG - NGUYÊN TẮC BẤT DI BẤT DỊCH

# ======================================================================

V6_CONFIG = {
# Entry conditions
‘rsi_buy_max’: 45,       # RSI < 45 = BUY pullback
‘rsi_sell_min’: 55,      # RSI > 55 = SELL pullback
‘adx_min’: 20,           # ADX > 20 = Có trend

```
# Risk management
'max_consecutive_losses': 3,  # 3 loss = cooldown
'cooldown_hours': 4,

# Karma thresholds
'karma_min': -20,        # Dưới -20 = Dừng trade
```

}

# ======================================================================

# SYMBOL-SPECIFIC SESSIONS (Server time UTC+0)

# Optimized for liquidity and spread

# ======================================================================

SYMBOL_SESSIONS = {
# EUR Cluster (London + NY overlap)
‘EURUSD’: {‘start’: 7, ‘end’: 20},   # London 7-16h + NY 13-20h overlap
‘EURGBP’: {‘start’: 7, ‘end’: 16},   # London only (both EU currencies)
‘EURJPY’: {‘start’: 7, ‘end’: 16},   # London session (avoid Tokyo for EUR)

```
# GBP Cluster (London + NY overlap)
'GBPUSD': {'start': 7, 'end': 20},   # London + NY overlap (best liquidity)
'GBPJPY': {'start': 7, 'end': 16},   # London only (high volatility)

# USD Majors (Multi-session coverage)
'USDJPY': {'start': 0, 'end': 16},   # Tokyo 0-9h + London 7-16h
'USDCAD': {'start': 13, 'end': 20},  # NY session (CAD most active)
'AUDUSD': {'start': 0, 'end': 9},    # Sydney + Tokyo (Asia-Pacific)

# Commodities (NY session - best liquidity)
'XAUUSD': {'start': 13, 'end': 20},  # NY session (US market hours)
'XAGUSD': {'start': 13, 'end': 20},  # NY session (follows gold)

# Indices (US equity hours)
'US30': {'start': 13, 'end': 20},    # NY session only

# Oceania (Early Asia)
'NZDUSD': {'start': 0, 'end': 9}     # Wellington + Sydney
```

}

# ======================================================================

# PIP MULTIPLIERS (for price to pips conversion)

# CRITICAL: Different pairs have different pip values!

# ======================================================================

PIP_MULTIPLIERS = {
# Standard FX pairs (5 decimals: 1.23456)
‘EURUSD’: 10000,    # 1 pip = 0.0001
‘GBPUSD’: 10000,
‘USDCAD’: 10000,
‘AUDUSD’: 10000,
‘NZDUSD’: 10000,
‘EURGBP’: 10000,

```
# JPY pairs (3 decimals: 123.456)
'USDJPY': 100,      # 1 pip = 0.01
'EURJPY': 100,
'GBPJPY': 100,

# Metals
'XAUUSD': 100,      # Gold (2 decimals: 1850.12)
'XAGUSD': 1000,     # Silver (3 decimals: 23.456)

# Indices
'US30': 100         # Dow (2 decimals: 35000.12)
```

}

# ======================================================================

# MAX SPREADS (reject trade if spread too wide, in pips)

# ======================================================================

MAX_SPREADS = {
# Tight spreads (Major pairs)
‘EURUSD’: 2.0,
‘GBPUSD’: 2.5,
‘USDJPY’: 2.0,
‘USDCAD’: 2.5,
‘AUDUSD’: 2.5,
‘NZDUSD’: 3.0,

```
# Medium spreads (Crosses)
'EURGBP': 3.0,
'EURJPY': 4.0,
'GBPJPY': 5.0,

# Wide spreads (Metals & Indices)
'XAUUSD': 5.0,
'XAGUSD': 8.0,
'US30': 5.0
```

}

# ======================================================================

# ENSEMBLE CONFIG

# ======================================================================

ENSEMBLE_CONFIG = {
# Model weights (based on accuracy)
‘mamba_weight’: 0.35,       # 73.3%
‘lstm_weight’: 0.40,        # 74.2% (highest)
‘transformer_weight’: 0.25, # 66.9%

```
# Decision thresholds
'ensemble_buy_threshold': 0.55,
'ensemble_sell_threshold': 0.45,

# Meta-Labeler thresholds (from real trade analysis)
'meta_strong_threshold': 0.70,    # High confidence
'meta_moderate_threshold': 0.60,  # Medium confidence
'meta_minimum_threshold': 0.50,   # Minimum to trade

# Consensus requirement
'require_consensus': True,  # 2/3 models must agree
'consensus_min_models': 2,
```

}

# ======================================================================

# KARMA LEVELS

# ======================================================================

KARMA_LEVELS = {
‘BUDDHA’: {‘min’: 200, ‘mult’: 2.0, ‘color’: ‘magenta’},
‘BODHISATTVA’: {‘min’: 100, ‘mult’: 1.5, ‘color’: ‘gold’},
‘ARHAT’: {‘min’: 50, ‘mult’: 1.25, ‘color’: ‘cyan’},
‘MONK’: {‘min’: 20, ‘mult’: 1.0, ‘color’: ‘green’},
‘NOVICE’: {‘min’: 0, ‘mult’: 1.0, ‘color’: ‘gray’},
‘SAMSARA’: {‘min’: -100, ‘mult’: 0.5, ‘color’: ‘red’},
}

# ======================================================================

# TREND ADAPTIVE SYSTEM CONFIG

# ======================================================================

TREND_CONFIG = {
‘STRONG’: {
‘adx_min’: 40,
‘tema_dist_min’: 0.5,
‘lot_mult’: 1.5,
‘sl_atr’: 2.0,
‘tp1_atr’: 3.0,
‘tp2_atr’: 5.0,
‘max_trades’: 5
},
‘MODERATE’: {
‘adx_min’: 25,
‘tema_dist_min’: 0.1,
‘lot_mult’: 1.0,
‘sl_atr’: 1.5,
‘tp1_atr’: 2.0,
‘tp2_atr’: 4.0,
‘max_trades’: 3
},
‘WEAK’: {
‘adx_min’: 0,
‘tema_dist_min’: 0,
‘lot_mult’: 0.5,
‘sl_atr’: 1.0,
‘tp1_atr’: 1.2,
‘tp2_atr’: 2.0,
‘max_trades’: 1
}
}

def get_trend_strength(adx_h4: float, rsi_h4: float, tema_dist_pct: float) -> str:
“”“Classify trend strength: STRONG / MODERATE / WEAK”””
if adx_h4 >= 40 and tema_dist_pct >= 0.5:
if rsi_h4 > 60 or rsi_h4 < 40:
return ‘STRONG’
if adx_h4 >= 25 and tema_dist_pct >= 0.1:
return ‘MODERATE’
return ‘WEAK’

def get_trend_params(trend_strength: str) -> dict:
“”“Get adaptive parameters for given trend strength”””
return TREND_CONFIG.get(trend_strength, TREND_CONFIG[‘MODERATE’])

# ======================================================================

# PHASE 2: KALMAN FILTER - Denoise Signals

# ======================================================================

class KalmanFilter:
“””
Kalman Filter for denoising RSI/ADX signals.
Reduces false signals caused by market noise.

```
State: [value, velocity]
Measurement: raw value
"""

def __init__(self, process_variance: float = 0.01, measurement_variance: float = 0.1):
    """
    Args:
        process_variance: Q - how much we expect the true value to change
        measurement_variance: R - how noisy our measurements are
    """
    self.Q = process_variance  # Process noise
    self.R = measurement_variance  # Measurement noise
    
    # State estimate [value, velocity]
    self.x = np.array([50.0, 0.0])  # Initial: RSI=50, no velocity
    
    # State covariance
    self.P = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
    
    # State transition matrix (predict next state)
    self.F = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
    
    # Measurement matrix (we only measure value, not velocity)
    self.H = np.array([[1.0, 0.0]])
    
    # Process noise covariance
    self.Q_matrix = np.array([[self.Q, 0.0],
                              [0.0, self.Q]])
    
    self.initialized = False

def reset(self, initial_value: float = 50.0):
    """Reset filter with new initial value"""
    self.x = np.array([initial_value, 0.0])
    self.P = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
    self.initialized = True

def update(self, measurement: float) -> float:
    """
    Process new measurement and return filtered value.
    
    Args:
        measurement: Raw RSI/ADX value
        
    Returns:
        Filtered (denoised) value
    """
    if not self.initialized:
        self.reset(measurement)
        return measurement
    
    # Predict step
    x_pred = self.F @ self.x
    P_pred = self.F @ self.P @ self.F.T + self.Q_matrix
    
    # Update step
    y = measurement - self.H @ x_pred  # Innovation
    S = self.H @ P_pred @ self.H.T + self.R  # Innovation covariance
    K = P_pred @ self.H.T / S  # Kalman gain
    
    self.x = x_pred + K.flatten() * y
    self.P = (np.eye(2) - np.outer(K, self.H)) @ P_pred
    
    # Clamp to valid range
    filtered_value = np.clip(self.x[0], 0, 100)
    
    return float(filtered_value)

def get_velocity(self) -> float:
    """Get current estimated velocity (rate of change)"""
    return float(self.x[1])

def filter_series(self, values: List[float]) -> List[float]:
    """Filter a series of values"""
    self.reset(values[0] if values else 50.0)
    return [self.update(v) for v in values]
```

class KalmanFilterBank:
“””
Bank of Kalman Filters for multiple indicators per symbol.
Maintains state across API calls.
“””

```
def __init__(self):
    self.filters = {}  # {symbol: {indicator: KalmanFilter}}
    self.lock = Lock()

def get_filter(self, symbol: str, indicator: str) -> KalmanFilter:
    """Get or create filter for symbol/indicator pair"""
    key = f"{symbol}_{indicator}"
    if key not in self.filters:
        # Different noise levels for different indicators
        if 'rsi' in indicator.lower():
            self.filters[key] = KalmanFilter(process_variance=0.05, measurement_variance=0.2)
        elif 'adx' in indicator.lower():
            self.filters[key] = KalmanFilter(process_variance=0.02, measurement_variance=0.15)
        else:
            self.filters[key] = KalmanFilter()
    return self.filters[key]

def filter_indicators(self, symbol: str, data: Dict[str, float]) -> Dict[str, float]:
    """
    Filter all indicators for a symbol.
    
    Args:
        symbol: Trading symbol
        data: Dict of indicator values {'rsi_m15': 45, 'adx_m15': 25, ...}
        
    Returns:
        Dict of filtered values with '_kalman' suffix
    """
    with self.lock:
        filtered = {}
        for key, value in data.items():
            if any(ind in key.lower() for ind in ['rsi', 'adx']):
                kf = self.get_filter(symbol, key)
                filtered[f"{key}_kalman"] = kf.update(float(value))
                filtered[f"{key}_velocity"] = kf.get_velocity()
        return filtered
```

# ======================================================================

# PHASE 2: MONTE CARLO - Risk Simulation

# ======================================================================

class MonteCarloSimulator:
“””
Monte Carlo simulation for trade risk estimation.
Simulates multiple scenarios to estimate win probability.
“””

```
def __init__(self, n_simulations: int = 1000):
    self.n_simulations = n_simulations
    
    # Historical volatility by symbol (can be updated from real data)
    self.volatility = {
        'EURUSD': 0.0008,  # ~8 pips typical M15 range
        'GBPUSD': 0.0012,  # ~12 pips
        'XAUUSD': 2.5,     # ~$2.5
        'US30': 25.0       # ~25 points
    }
    
    # Win rate by signal strength (from historical data)
    self.base_win_rates = {
        'STRONG': 0.65,
        'MODERATE': 0.55,
        'WEAK': 0.45
    }

def simulate_trade(self, 
                  symbol: str,
                  signal: int,  # 1=BUY, -1=SELL
                  entry_price: float,
                  sl_distance: float,
                  tp_distance: float,
                  signal_strength: str = 'MODERATE',
                  confidence: float = 0.5) -> Dict:
    """
    Simulate trade outcomes using Monte Carlo.
    
    Args:
        symbol: Trading symbol
        signal: 1 for BUY, -1 for SELL
        entry_price: Entry price
        sl_distance: Stop loss distance (in price)
        tp_distance: Take profit distance (in price)
        signal_strength: STRONG/MODERATE/WEAK
        confidence: Model confidence (0-1)
        
    Returns:
        Dict with win_probability, expected_rr, risk_score
    """
    if signal == 0:
        return {
            'win_probability': 0,
            'expected_rr': 0,
            'risk_score': 0,
            'simulations': 0,
            'recommendation': 'NO_TRADE'
        }
    
    vol = self.volatility.get(symbol, 0.001)
    base_wr = self.base_win_rates.get(signal_strength, 0.5)
    
    # Adjust win rate based on confidence
    adjusted_wr = base_wr * (0.7 + 0.3 * confidence)
    
    wins = 0
    total_pnl = 0
    
    for _ in range(self.n_simulations):
        # Simulate price path (random walk with drift based on signal)
        drift = signal * vol * 0.1 * confidence  # Small drift in signal direction
        
        # Simulate time steps until SL or TP hit
        price = entry_price
        hit_tp = False
        hit_sl = False
        
        for step in range(100):  # Max 100 steps
            # Random price change
            change = np.random.normal(drift, vol)
            price += change
            
            # Check TP/SL
            if signal == 1:  # BUY
                if price >= entry_price + tp_distance:
                    hit_tp = True
                    break
                elif price <= entry_price - sl_distance:
                    hit_sl = True
                    break
            else:  # SELL
                if price <= entry_price - tp_distance:
                    hit_tp = True
                    break
                elif price >= entry_price + sl_distance:
                    hit_sl = True
                    break
        
        if hit_tp:
            wins += 1
            total_pnl += tp_distance
        elif hit_sl:
            total_pnl -= sl_distance
        # else: no hit, treat as scratch (0 pnl)
    
    win_probability = wins / self.n_simulations
    expected_pnl = total_pnl / self.n_simulations
    risk_reward = tp_distance / sl_distance if sl_distance > 0 else 0
    
    # Risk score: 0-100, higher = riskier
    risk_score = 100 * (1 - win_probability) * (1 / (1 + risk_reward))
    
    # Recommendation
    if win_probability >= 0.6 and risk_reward >= 1.5:
        recommendation = 'STRONG_ENTRY'
    elif win_probability >= 0.5 and risk_reward >= 1.0:
        recommendation = 'NORMAL_ENTRY'
    elif win_probability >= 0.45:
        recommendation = 'WEAK_ENTRY'
    else:
        recommendation = 'AVOID'
    
    return {
        'win_probability': round(win_probability, 3),
        'expected_pnl_ratio': round(expected_pnl / sl_distance if sl_distance > 0 else 0, 3),
        'risk_reward': round(risk_reward, 2),
        'risk_score': round(risk_score, 1),
        'simulations': self.n_simulations,
        'recommendation': recommendation
    }

def update_volatility(self, symbol: str, volatility: float):
    """Update volatility estimate from real data"""
    self.volatility[symbol] = volatility
```

# ======================================================================

# PHASE 2: POLYNOMIAL FEATURES - Feature Engineering

# ======================================================================

class PolynomialFeatureGenerator:
“””
Generate polynomial and interaction features for better predictions.
Creates: RSI^2, ADX^2, RSI*ADX, normalized features, etc.
“””

```
def __init__(self, degree: int = 2, include_interactions: bool = True):
    self.degree = degree
    self.include_interactions = include_interactions

def generate(self, data: Dict[str, float]) -> Dict[str, float]:
    """
    Generate polynomial features from indicator data.
    
    Args:
        data: Dict of indicator values
        
    Returns:
        Dict with original + polynomial features
    """
    features = dict(data)  # Copy original
    
    # Extract base indicators
    rsi_m15 = data.get('rsi_m15', 50)
    rsi_h1 = data.get('rsi_h1', 50)
    rsi_h4 = data.get('rsi_h4', 50)
    adx_m15 = data.get('adx_m15', 20)
    adx_h4 = data.get('adx_h4', 20)
    
    # Normalize to 0-1 range
    rsi_m15_norm = rsi_m15 / 100
    rsi_h1_norm = rsi_h1 / 100
    rsi_h4_norm = rsi_h4 / 100
    adx_m15_norm = min(adx_m15 / 50, 1.0)
    adx_h4_norm = min(adx_h4 / 50, 1.0)
    
    # Polynomial features (degree 2)
    features['rsi_m15_sq'] = rsi_m15_norm ** 2
    features['rsi_h1_sq'] = rsi_h1_norm ** 2
    features['rsi_h4_sq'] = rsi_h4_norm ** 2
    features['adx_m15_sq'] = adx_m15_norm ** 2
    features['adx_h4_sq'] = adx_h4_norm ** 2
    
    # Interaction features
    if self.include_interactions:
        features['rsi_m15_x_adx_m15'] = rsi_m15_norm * adx_m15_norm
        features['rsi_h1_x_adx_h4'] = rsi_h1_norm * adx_h4_norm
        features['rsi_m15_x_rsi_h1'] = rsi_m15_norm * rsi_h1_norm
        features['rsi_h1_x_rsi_h4'] = rsi_h1_norm * rsi_h4_norm
        features['adx_m15_x_adx_h4'] = adx_m15_norm * adx_h4_norm
    
    # Derived features
    features['rsi_avg'] = (rsi_m15 + rsi_h1 + rsi_h4) / 3
    features['rsi_std'] = np.std([rsi_m15, rsi_h1, rsi_h4])
    features['rsi_range'] = max(rsi_m15, rsi_h1, rsi_h4) - min(rsi_m15, rsi_h1, rsi_h4)
    features['adx_avg'] = (adx_m15 + adx_h4) / 2
    
    # Distance from extremes (overbought/oversold)
    features['rsi_m15_dist_30'] = abs(rsi_m15 - 30) / 100  # Distance from oversold
    features['rsi_m15_dist_70'] = abs(rsi_m15 - 70) / 100  # Distance from overbought
    features['rsi_m15_dist_50'] = abs(rsi_m15 - 50) / 100  # Distance from neutral
    
    # Trend alignment score
    if data.get('main_trend', 0) == 1:  # Bullish
        features['trend_alignment'] = (100 - rsi_m15) / 100  # Lower RSI = better buy
    elif data.get('main_trend', 0) == -1:  # Bearish
        features['trend_alignment'] = rsi_m15 / 100  # Higher RSI = better sell
    else:
        features['trend_alignment'] = 0.5
    
    # Momentum features
    features['rsi_momentum'] = rsi_m15 - rsi_h1  # Short vs medium term
    features['rsi_acceleration'] = (rsi_m15 - rsi_h1) - (rsi_h1 - rsi_h4)  # Rate of change
    
    # ADX trend strength category
    if adx_m15 >= 40:
        features['adx_category'] = 1.0  # Strong trend
    elif adx_m15 >= 25:
        features['adx_category'] = 0.5  # Moderate trend
    else:
        features['adx_category'] = 0.0  # Weak/no trend
    
    return features

def get_feature_vector(self, data: Dict[str, float], 
                      feature_names: List[str] = None) -> np.ndarray:
    """
    Get feature vector for model input.
    
    Args:
        data: Raw indicator data
        feature_names: List of feature names to include (in order)
        
    Returns:
        numpy array of features
    """
    features = self.generate(data)
    
    if feature_names is None:
        feature_names = [
            'rsi_m15', 'rsi_h1', 'rsi_h4',
            'adx_m15', 'adx_h4',
            'rsi_m15_sq', 'rsi_h1_sq', 
            'rsi_m15_x_adx_m15', 'rsi_h1_x_adx_h4',
            'rsi_avg', 'rsi_std', 'rsi_range',
            'trend_alignment', 'rsi_momentum',
            'adx_category'
        ]
    
    return np.array([features.get(name, 0) for name in feature_names])
```

# ======================================================================

# PHASE 2: CONFIG

# ======================================================================

PHASE2_CONFIG = {
# Kalman Filter
‘kalman_enabled’: True,
‘kalman_process_variance’: 0.05,
‘kalman_measurement_variance’: 0.2,

```
# Monte Carlo - LOT ADJUSTMENT (không filter)
'monte_carlo_enabled': True,
'monte_carlo_simulations': 1000,
# Removed: monte_carlo_min_win_prob (không còn filter)

# Polynomial Features
'polynomial_enabled': True,
'polynomial_degree': 2,
'polynomial_interactions': True,
```

}

# Global Phase 2 instances

kalman_bank = KalmanFilterBank()
monte_carlo = MonteCarloSimulator(n_simulations=PHASE2_CONFIG[‘monte_carlo_simulations’])
poly_features = PolynomialFeatureGenerator(
degree=PHASE2_CONFIG[‘polynomial_degree’],
include_interactions=PHASE2_CONFIG[‘polynomial_interactions’]
)

# ======================================================================

# PHASE 3: SENTIMENT READER (NHAN - Man Engine)

# ======================================================================

class SentimentReader:
“””
Đọc sentiment từ JSON file (được tạo bởi News_Sentiment_Worker.py)
VADER-based, chạy CPU, không cần GPU.
“””

```
def __init__(self, sentiment_file: str = "./bodhi_data/market_sentiment.json"):
    self.sentiment_file = sentiment_file
    self.cache = {
        'score': 0.0,
        'status': 'NEUTRAL',
        'last_update': None,
        'news_count': 0
    }
    self.cache_time = None
    self.cache_ttl = 60  # Cache 60 seconds

def get_sentiment(self) -> Dict:
    """
    Đọc sentiment hiện tại.
    Returns: {'score': float, 'status': str, 'last_update': str}
    """
    now = datetime.now()
    
    # Check cache
    if self.cache_time and (now - self.cache_time).seconds < self.cache_ttl:
        return self.cache
    
    try:
        if os.path.exists(self.sentiment_file):
            with open(self.sentiment_file, 'r') as f:
                data = json.load(f)
                self.cache = {
                    'score': float(data.get('score', 0.0)),
                    'status': data.get('status', 'NEUTRAL'),
                    'last_update': data.get('last_update'),
                    'news_count': data.get('news_count', 0),
                    'top_news': data.get('top_news', [])[:3]
                }
                self.cache_time = now
                
                # Check if data is stale (> 1 hour old)
                if self.cache['last_update']:
                    try:
                        update_time = datetime.fromisoformat(self.cache['last_update'])
                        if (now - update_time).seconds > 3600:
                            self.cache['status'] = 'STALE'
                            pass  # logger.warning("[SENTIMENT] Data is stale (>1h old)")
                    except:
                        pass
    except Exception as e:
        pass  # logger.warning(f"[SENTIMENT] Error reading file: {e}")
    
    return self.cache

def get_sentiment_bias(self) -> float:
    """
    Trả về sentiment bias cho trading decision.
    Returns: -1.0 (very bearish) to +1.0 (very bullish)
    """
    sentiment = self.get_sentiment()
    return sentiment.get('score', 0.0)

def should_trade_with_sentiment(self, signal: int) -> Tuple[bool, float]:
    """
    Kiểm tra signal có phù hợp với sentiment không.
    
    Args:
        signal: 1 (BUY), -1 (SELL), 0 (HOLD)
        
    Returns:
        (should_trade, confidence_multiplier)
    """
    sentiment = self.get_sentiment()
    score = sentiment.get('score', 0.0)
    status = sentiment.get('status', 'NEUTRAL')
    
    # NEUTRAL sentiment = no adjustment
    if status == 'NEUTRAL' or status == 'STALE':
        return True, 1.0
    
    # BUY signal
    if signal == 1:
        if status == 'BULLISH':
            return True, 1.2  # Boost confidence
        elif status == 'BEARISH':
            return True, 0.7  # Reduce confidence (but still trade)
    
    # SELL signal
    elif signal == -1:
        if status == 'BEARISH':
            return True, 1.2  # Boost confidence
        elif status == 'BULLISH':
            return True, 0.7  # Reduce confidence
    
    return True, 1.0
```

# ======================================================================

# PHASE 3: KELLY CRITERION (Dynamic Lot Sizing)

# ======================================================================

class KellyCriterion:
“””
Kelly Criterion cho dynamic lot sizing.
f* = (p * b - q) / b

```
Trong đó:
- p = probability of winning (từ Meta-Labeler)
- q = probability of losing (1 - p)
- b = win/loss ratio (reward/risk)
"""

def __init__(self, max_kelly_fraction: float = 0.5, min_kelly_fraction: float = 0.1):
    """
    Args:
        max_kelly_fraction: Giới hạn tối đa Kelly (0.5 = Half Kelly)
        min_kelly_fraction: Giới hạn tối thiểu
    """
    self.max_fraction = max_kelly_fraction
    self.min_fraction = min_kelly_fraction

def calculate(self, win_probability: float, reward_risk_ratio: float) -> Dict:
    """
    Tính Kelly fraction.
    
    Args:
        win_probability: Xác suất thắng (0.0 - 1.0)
        reward_risk_ratio: Tỷ lệ reward/risk (TP/SL)
        
    Returns:
        Dict với kelly_fraction, lot_multiplier, recommendation
    """
    p = max(0.01, min(0.99, win_probability))  # Clamp to valid range
    q = 1 - p
    b = max(0.1, reward_risk_ratio)  # Avoid division by zero
    
    # Kelly Formula: f* = (p * b - q) / b
    kelly_raw = (p * b - q) / b
    
    # Clamp to safe range
    kelly_fraction = max(self.min_fraction, min(self.max_fraction, kelly_raw))
    
    # Negative Kelly = don't trade
    if kelly_raw <= 0:
        return {
            'kelly_raw': round(kelly_raw, 4),
            'kelly_fraction': 0,
            'lot_multiplier': 0,
            'recommendation': 'NO_TRADE',
            'reason': 'Negative edge - expected loss'
        }
    
    # Lot multiplier (1.0 = full position at Half Kelly)
    lot_multiplier = kelly_fraction / self.max_fraction
    
    # Recommendation
    if kelly_fraction >= 0.4:
        recommendation = 'STRONG'
    elif kelly_fraction >= 0.25:
        recommendation = 'NORMAL'
    elif kelly_fraction >= 0.15:
        recommendation = 'REDUCED'
    else:
        recommendation = 'MINIMAL'
    
    return {
        'kelly_raw': round(kelly_raw, 4),
        'kelly_fraction': round(kelly_fraction, 4),
        'lot_multiplier': round(lot_multiplier, 3),
        'recommendation': recommendation,
        'win_prob': round(p, 3),
        'rr_ratio': round(b, 2)
    }
```

# ======================================================================

# PHASE 3: VOLUME PROFILE (DIA - Earth Engine)

# ======================================================================

class VolumeProfile:
“””
Volume Profile Analysis - Tìm POC và Value Area.
Chạy hoàn toàn trên CPU với numpy.
“””

```
def __init__(self, num_bins: int = 50, value_area_pct: float = 0.70):
    """
    Args:
        num_bins: Số mức giá để phân tích
        value_area_pct: % volume cho Value Area (default 70%)
    """
    self.num_bins = num_bins
    self.value_area_pct = value_area_pct

def calculate(self, highs: List[float], lows: List[float], 
              volumes: List[float], closes: List[float] = None) -> Dict:
    """
    Tính Volume Profile.
    
    Args:
        highs: List giá High
        lows: List giá Low
        volumes: List volume
        closes: List giá Close (optional)
        
    Returns:
        Dict với POC, VAH, VAL, profile data
    """
    if not highs or not lows or not volumes:
        return self._empty_result()
    
    highs = np.array(highs)
    lows = np.array(lows)
    volumes = np.array(volumes)
    
    # Price range
    price_min = np.min(lows)
    price_max = np.max(highs)
    
    if price_max <= price_min:
        return self._empty_result()
    
    # Create price bins
    bin_edges = np.linspace(price_min, price_max, self.num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_volumes = np.zeros(self.num_bins)
    
    # Distribute volume across price bins
    for i in range(len(highs)):
        h, l, v = highs[i], lows[i], volumes[i]
        if v <= 0:
            continue
            
        # Find bins that this candle touches
        low_bin = np.searchsorted(bin_edges, l, side='right') - 1
        high_bin = np.searchsorted(bin_edges, h, side='left')
        
        low_bin = max(0, min(low_bin, self.num_bins - 1))
        high_bin = max(0, min(high_bin, self.num_bins - 1))
        
        # Distribute volume evenly across touched bins
        num_touched = high_bin - low_bin + 1
        vol_per_bin = v / max(1, num_touched)
        
        for b in range(low_bin, high_bin + 1):
            bin_volumes[b] += vol_per_bin
    
    # POC (Point of Control) - highest volume price
    poc_idx = np.argmax(bin_volumes)
    poc_price = bin_centers[poc_idx]
    
    # Value Area (70% of volume)
    total_volume = np.sum(bin_volumes)
    target_volume = total_volume * self.value_area_pct
    
    # Expand from POC until we capture target volume
    vah_idx = poc_idx
    val_idx = poc_idx
    current_volume = bin_volumes[poc_idx]
    
    while current_volume < target_volume:
        # Check which direction to expand
        upper_vol = bin_volumes[vah_idx + 1] if vah_idx + 1 < self.num_bins else 0
        lower_vol = bin_volumes[val_idx - 1] if val_idx - 1 >= 0 else 0
        
        if upper_vol >= lower_vol and vah_idx + 1 < self.num_bins:
            vah_idx += 1
            current_volume += upper_vol
        elif val_idx - 1 >= 0:
            val_idx -= 1
            current_volume += lower_vol
        else:
            break
    
    vah_price = bin_centers[min(vah_idx, self.num_bins - 1)]
    val_price = bin_centers[max(val_idx, 0)]
    
    # Current price position relative to VP
    current_price = closes[-1] if closes else (highs[-1] + lows[-1]) / 2
    
    if current_price > vah_price:
        position = 'ABOVE_VA'
        bias = -0.3  # Price above value = potential sell
    elif current_price < val_price:
        position = 'BELOW_VA'
        bias = 0.3   # Price below value = potential buy
    else:
        position = 'INSIDE_VA'
        bias = 0.0   # Inside value area = neutral
    
    return {
        'poc': round(poc_price, 5),
        'vah': round(vah_price, 5),  # Value Area High
        'val': round(val_price, 5),  # Value Area Low
        'position': position,
        'bias': bias,
        'current_price': round(current_price, 5),
        'total_volume': round(total_volume, 2),
        'bins': self.num_bins
    }

def _empty_result(self) -> Dict:
    return {
        'poc': 0, 'vah': 0, 'val': 0,
        'position': 'UNKNOWN', 'bias': 0,
        'current_price': 0, 'total_volume': 0, 'bins': 0
    }
```

# ======================================================================

# PHASE 3: DBSCAN S/R ZONES (DIA - Earth Engine)

# ======================================================================

class DBSCANSupportResistance:
“””
DBSCAN Clustering để tự động tìm vùng Support/Resistance.
Chạy hoàn toàn trên CPU với scikit-learn.
“””

```
def __init__(self, eps_pct: float = 0.002, min_samples: int = 3):
    """
    Args:
        eps_pct: Khoảng cách % để gom cluster (0.002 = 0.2%)
        min_samples: Số điểm tối thiểu để tạo cluster
    """
    self.eps_pct = eps_pct
    self.min_samples = min_samples

def find_zones(self, highs: List[float], lows: List[float], 
               closes: List[float]) -> Dict:
    """
    Tìm vùng S/R từ swing highs/lows.
    
    Args:
        highs: List giá High
        lows: List giá Low
        closes: List giá Close
        
    Returns:
        Dict với support_zones, resistance_zones
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        pass  # logger.warning("[DBSCAN] sklearn not installed")
        return self._empty_result()
    
    if len(highs) < 10:
        return self._empty_result()
    
    highs = np.array(highs)
    lows = np.array(lows)
    closes = np.array(closes)
    
    # Find swing highs and lows
    swing_highs = self._find_swing_points(highs, is_high=True)
    swing_lows = self._find_swing_points(lows, is_high=False)
    
    if len(swing_highs) < 2 and len(swing_lows) < 2:
        return self._empty_result()
    
    # Calculate eps based on price range
    price_range = np.max(highs) - np.min(lows)
    eps = price_range * self.eps_pct
    
    # Cluster swing highs (Resistance)
    resistance_zones = []
    if len(swing_highs) >= self.min_samples:
        X_highs = swing_highs.reshape(-1, 1)
        db_highs = DBSCAN(eps=eps, min_samples=self.min_samples).fit(X_highs)
        
        for label in set(db_highs.labels_):
            if label == -1:  # Noise
                continue
            cluster_prices = swing_highs[db_highs.labels_ == label]
            zone_price = np.mean(cluster_prices)
            zone_strength = len(cluster_prices)
            resistance_zones.append({
                'price': round(zone_price, 5),
                'strength': zone_strength,
                'type': 'RESISTANCE'
            })
    
    # Cluster swing lows (Support)
    support_zones = []
    if len(swing_lows) >= self.min_samples:
        X_lows = swing_lows.reshape(-1, 1)
        db_lows = DBSCAN(eps=eps, min_samples=self.min_samples).fit(X_lows)
        
        for label in set(db_lows.labels_):
            if label == -1:
                continue
            cluster_prices = swing_lows[db_lows.labels_ == label]
            zone_price = np.mean(cluster_prices)
            zone_strength = len(cluster_prices)
            support_zones.append({
                'price': round(zone_price, 5),
                'strength': zone_strength,
                'type': 'SUPPORT'
            })
    
    # Sort by strength
    resistance_zones.sort(key=lambda x: x['strength'], reverse=True)
    support_zones.sort(key=lambda x: x['strength'], reverse=True)
    
    # Current price analysis
    current_price = closes[-1]
    nearest_support = self._find_nearest(support_zones, current_price, 'below')
    nearest_resistance = self._find_nearest(resistance_zones, current_price, 'above')
    
    return {
        'resistance_zones': resistance_zones[:5],  # Top 5
        'support_zones': support_zones[:5],
        'nearest_support': nearest_support,
        'nearest_resistance': nearest_resistance,
        'current_price': round(current_price, 5),
        'zone_count': len(resistance_zones) + len(support_zones)
    }

def _find_swing_points(self, prices: np.ndarray, is_high: bool, 
                       window: int = 5) -> np.ndarray:
    """Tìm swing highs hoặc lows"""
    swings = []
    for i in range(window, len(prices) - window):
        if is_high:
            if prices[i] == np.max(prices[i-window:i+window+1]):
                swings.append(prices[i])
        else:
            if prices[i] == np.min(prices[i-window:i+window+1]):
                swings.append(prices[i])
    return np.array(swings)

def _find_nearest(self, zones: List[Dict], price: float, 
                  direction: str) -> Optional[Dict]:
    """Tìm zone gần nhất"""
    if not zones:
        return None
    
    if direction == 'above':
        above_zones = [z for z in zones if z['price'] > price]
        if above_zones:
            return min(above_zones, key=lambda x: x['price'] - price)
    else:
        below_zones = [z for z in zones if z['price'] < price]
        if below_zones:
            return max(below_zones, key=lambda x: x['price'])
    return None

def _empty_result(self) -> Dict:
    return {
        'resistance_zones': [],
        'support_zones': [],
        'nearest_support': None,
        'nearest_resistance': None,
        'current_price': 0,
        'zone_count': 0
    }
```

# ======================================================================

# PHASE 3: CONFIG

# ======================================================================

PHASE3_CONFIG = {
# Sentiment (NHAN)
‘sentiment_enabled’: True,
‘sentiment_file’: ‘./bodhi_data/market_sentiment.json’,

```
# Kelly Criterion
'kelly_enabled': True,
'kelly_max_fraction': 0.5,  # Half Kelly (conservative)
'kelly_min_fraction': 0.1,

# Volume Profile (DIA)
'volume_profile_enabled': True,
'volume_profile_bins': 50,
'value_area_pct': 0.70,

# DBSCAN S/R (DIA)
'dbscan_enabled': True,
'dbscan_eps_pct': 0.002,  # 0.2% price distance
'dbscan_min_samples': 3,
```

}

# Global Phase 3 instances

sentiment_reader = SentimentReader(PHASE3_CONFIG[‘sentiment_file’])
kelly_criterion = KellyCriterion(
max_kelly_fraction=PHASE3_CONFIG[‘kelly_max_fraction’],
min_kelly_fraction=PHASE3_CONFIG[‘kelly_min_fraction’]
)
volume_profile = VolumeProfile(
num_bins=PHASE3_CONFIG[‘volume_profile_bins’],
value_area_pct=PHASE3_CONFIG[‘value_area_pct’]
)
dbscan_sr = DBSCANSupportResistance(
eps_pct=PHASE3_CONFIG[‘dbscan_eps_pct’],
min_samples=PHASE3_CONFIG[‘dbscan_min_samples’]
)

# ======================================================================

# SYMBOL MAPPING (Legacy - For backward compatibility)

# ======================================================================

# 

# NOTE: normalize_symbol() now uses WILDCARD MATCHING as primary method:

# EURUSD* → EURUSD (handles any suffix: EURUSDm, EURUSDpro, EURUSD.raw, etc.)

# GBPUSD* → GBPUSD

# XAUUSD* → XAUUSD

# … etc for all 12 pairs

# 

# This SYMBOL_MAP is kept for:

# 1. Backward compatibility

# 2. Special broker naming (e.g., GOLD → XAUUSD, DJ30 → US30)

# 3. Explicit mappings for edge cases

# 

# ======================================================================

SYMBOL_MAP = {
# ═══════════════════════════════════════════════════════════════
# 12 PAIRS SYMBOL MAPPING - Broker variants to Bodhi standard
# ═══════════════════════════════════════════════════════════════

```
# EUR Cluster
'EURUSD': 'EURUSD', 'EURUSD.A': 'EURUSD', 'EURUSDX': 'EURUSD',
'EURUSD_': 'EURUSD', 'EURUSD.': 'EURUSD', 'EURUSDC': 'EURUSD',
'EURUSD.RAW': 'EURUSD', 'EURUSDM': 'EURUSD',

'EURGBP': 'EURGBP', 'EURGBP.A': 'EURGBP', 'EURGBPX': 'EURGBP',
'EURGBP_': 'EURGBP', 'EURGBP.': 'EURGBP', 'EURGBPC': 'EURGBP',
'EURGBP.RAW': 'EURGBP', 'EURGBPM': 'EURGBP',

'EURJPY': 'EURJPY', 'EURJPY.A': 'EURJPY', 'EURJPYX': 'EURJPY',
'EURJPY_': 'EURJPY', 'EURJPY.': 'EURJPY', 'EURJPYC': 'EURJPY',
'EURJPY.RAW': 'EURJPY', 'EURJPYM': 'EURJPY',

# GBP Cluster
'GBPUSD': 'GBPUSD', 'GBPUSD.A': 'GBPUSD', 'GBPUSDX': 'GBPUSD',
'GBPUSD_': 'GBPUSD', 'GBPUSD.': 'GBPUSD', 'GBPUSDC': 'GBPUSD',
'GBPUSD.RAW': 'GBPUSD', 'GBPUSDM': 'GBPUSD',

'GBPJPY': 'GBPJPY', 'GBPJPY.A': 'GBPJPY', 'GBPJPYX': 'GBPJPY',
'GBPJPY_': 'GBPJPY', 'GBPJPY.': 'GBPJPY', 'GBPJPYC': 'GBPJPY',
'GBPJPY.RAW': 'GBPJPY', 'GBPJPYM': 'GBPJPY',

# USD Majors
'USDJPY': 'USDJPY', 'USDJPY.A': 'USDJPY', 'USDJPYX': 'USDJPY',
'USDJPY_': 'USDJPY', 'USDJPY.': 'USDJPY', 'USDJPYC': 'USDJPY',
'USDJPY.RAW': 'USDJPY', 'USDJPYM': 'USDJPY',

'USDCAD': 'USDCAD', 'USDCAD.A': 'USDCAD', 'USDCADX': 'USDCAD',
'USDCAD_': 'USDCAD', 'USDCAD.': 'USDCAD', 'USDCADC': 'USDCAD',
'USDCAD.RAW': 'USDCAD', 'USDCADM': 'USDCAD',

'AUDUSD': 'AUDUSD', 'AUDUSD.A': 'AUDUSD', 'AUDUSDX': 'AUDUSD',
'AUDUSD_': 'AUDUSD', 'AUDUSD.': 'AUDUSD', 'AUDUSDC': 'AUDUSD',
'AUDUSD.RAW': 'AUDUSD', 'AUDUSDM': 'AUDUSD',

# Commodities
'XAUUSD': 'XAUUSD', 'XAUUSD.A': 'XAUUSD', 'XAUUSDX': 'XAUUSD',
'XAUUSD_': 'XAUUSD', 'XAUUSD.': 'XAUUSD', 'XAUUSDC': 'XAUUSD',
'XAUUSD.RAW': 'XAUUSD', 'XAUUSDM': 'XAUUSD',
'GOLD': 'XAUUSD', 'GOLDC': 'XAUUSD', 'GOLD.A': 'XAUUSD',

'XAGUSD': 'XAGUSD', 'XAGUSD.A': 'XAGUSD', 'XAGUSDX': 'XAGUSD',
'XAGUSD_': 'XAGUSD', 'XAGUSD.': 'XAGUSD', 'XAGUSDC': 'XAGUSD',
'XAGUSD.RAW': 'XAGUSD', 'XAGUSDM': 'XAGUSD',
'SILVER': 'XAGUSD', 'SILVERC': 'XAGUSD', 'SILVER.A': 'XAGUSD',

# Indices
'US30': 'US30', 'US30.A': 'US30', 'US30.CASH': 'US30', 'US30CASH': 'US30',
'US30C': 'US30', 'DJ30': 'US30', 'DJ30.C': 'US30', 'DJI30': 'US30',
'DJI30X': 'US30', 'DJI30_': 'US30', 'USA30': 'US30', 'USA30.IDX': 'US30',
'DOWJONES': 'US30', 'DOW': 'US30', 'DOW30': 'US30',

# Oceania
'NZDUSD': 'NZDUSD', 'NZDUSD.A': 'NZDUSD', 'NZDUSDX': 'NZDUSD',
'NZDUSD_': 'NZDUSD', 'NZDUSD.': 'NZDUSD', 'NZDUSDC': 'NZDUSD',
'NZDUSD.RAW': 'NZDUSD', 'NZDUSDM': 'NZDUSD',
```

}

BASE_PAIR_PREFIXES = (
    'XAUUSD', 'XAGUSD', 'US30',
    'EURUSD', 'EURGBP', 'EURJPY',
    'GBPUSD', 'GBPJPY',
    'USDJPY', 'USDCAD', 'AUDUSD',
    'NZDUSD',
)

ALT_PREFIXES = (
    ('GOLD', 'XAUUSD'),
    ('SILVER', 'XAGUSD'),
    ('DJ', 'US30'),
    ('DOW', 'US30'),
    ('USA30', 'US30'),
)

@lru_cache(maxsize=256)
def normalize_symbol(raw_symbol: str) -> str:
“””
Normalize broker-specific symbol to Bodhi standard (12 pairs support)

```
Handles wildcard matching:
- EURUSD* → EURUSD (e.g., EURUSDm, EURUSDpro, EURUSD.raw, etc.)
- GBPUSD* → GBPUSD
- XAUUSD* → XAUUSD
... etc for all 12 pairs
"""
if not raw_symbol:
    return 'EURUSD'

# Clean and uppercase
symbol = raw_symbol.upper().strip()

# Remove common broker prefixes/suffixes first
# Some brokers use: #EURUSD, FX:EURUSD, etc.
symbol = symbol.replace('#', '').replace('FX:', '').replace('FOREX:', '')

# ═══════════════════════════════════════════════════════════════
# LEGACY: Direct lookup in SYMBOL_MAP (backward compatibility)
# ═══════════════════════════════════════════════════════════════
normalized = SYMBOL_MAP.get(symbol)
if normalized is not None:
    return normalized

# ═══════════════════════════════════════════════════════════════
# WILDCARD MATCHING - Check if symbol STARTS WITH known pairs
# Handles: EURUSD*, GBPUSD*, XAUUSD*, etc.
# ═══════════════════════════════════════════════════════════════

# Define all 12 base pairs in priority order
# (configured once at module scope for speed)
# Check if symbol starts with any base pair
for base_pair in BASE_PAIR_PREFIXES:
    if symbol.startswith(base_pair):
        # Wildcard match! EURUSD* → EURUSD
        return base_pair

# ═══════════════════════════════════════════════════════════════
# ALTERNATIVE NAMES (Gold, Silver, Dow, etc.)
# ═══════════════════════════════════════════════════════════════
for prefix, canonical in ALT_PREFIXES:
    if symbol.startswith(prefix):
        return canonical

# ═══════════════════════════════════════════════════════════════
# FUZZY MATCHING (Last resort - for unusual formats)
# ═══════════════════════════════════════════════════════════════

# EUR Cluster
if 'EUR' in symbol and 'USD' in symbol: return 'EURUSD'
if 'EUR' in symbol and 'GBP' in symbol: return 'EURGBP'
if 'EUR' in symbol and 'JPY' in symbol: return 'EURJPY'

# GBP Cluster
if 'GBP' in symbol and 'USD' in symbol: return 'GBPUSD'
if 'GBP' in symbol and 'JPY' in symbol: return 'GBPJPY'

# USD Majors
if 'USD' in symbol and 'JPY' in symbol: return 'USDJPY'
if 'USD' in symbol and 'CAD' in symbol: return 'USDCAD'
if 'AUD' in symbol and 'USD' in symbol: return 'AUDUSD'

# Commodities
if 'XAU' in symbol or 'GOLD' in symbol: return 'XAUUSD'
if 'XAG' in symbol or 'SILVER' in symbol: return 'XAGUSD'

# Indices
if 'US30' in symbol or 'DJ' in symbol or 'DOW' in symbol: return 'US30'

# Oceania
if 'NZD' in symbol and 'USD' in symbol: return 'NZDUSD'

# Unknown symbol - log warning and default
logger.warning(f"Unknown symbol: {raw_symbol} - defaulting to EURUSD")
return 'EURUSD'
```

# ======================================================================
#
# PIPS NORMALIZATION
#
# ======================================================================

@lru_cache(maxsize=256)
def get_base_symbol(symbol: str) -> str:
“”“Remove suffix like .cash, .a, .raw, etc.”””
return symbol.replace(’.cash’, ‘’).replace(’.a’, ‘’).replace(’.raw’, ‘’).replace(’_’, ‘’).replace(’.’, ‘’).upper()

def calculate_pips(symbol: str, open_price: float, close_price: float,
profit_money: float, lot: float, trade_type: str = “UNKNOWN”) -> float:
“”“Calculate profit pips from price or estimate from profit money.”””
base_symbol = get_base_symbol(symbol)

```
# METHOD 1: Calculate from prices
if open_price > 0 and close_price > 0:
    price_diff = close_price - open_price
    if trade_type.upper() == "SELL":
        price_diff = open_price - close_price
    
    if 'US30' in base_symbol or 'DJI' in base_symbol or 'DOW' in base_symbol:
        return round(price_diff, 1)
    elif 'XAU' in base_symbol or 'GOLD' in base_symbol:
        return round(price_diff * 10, 1)
    elif 'JPY' in base_symbol:
        return round(price_diff * 100, 1)
    else:
        return round(price_diff * 10000, 1)

# METHOD 2: Estimate from profit money
if profit_money == 0:
    return 0.0
if lot <= 0:
    lot = 0.01

pip_value_per_lot = 10.0
if 'US30' in base_symbol or 'DJI' in base_symbol:
    pip_value_per_lot = 1.0
elif 'XAU' in base_symbol or 'GOLD' in base_symbol:
    pip_value_per_lot = 10.0
elif 'JPY' in base_symbol:
    pip_value_per_lot = 7.0

pip_value = pip_value_per_lot * lot
if pip_value > 0:
    return round(profit_money / pip_value, 1)
return 0.0
```

def normalize_pips(symbol: str, ea_pips: float, profit_money: float, lot: float,
open_price: float = 0, close_price: float = 0, trade_type: str = “UNKNOWN”) -> float:
“”“Normalize pips from EA - use price if available, validate otherwise.”””
if open_price > 0 and close_price > 0:
return calculate_pips(symbol, open_price, close_price, profit_money, lot, trade_type)

```
if profit_money == 0:
    return 0.0

expected_pips = calculate_pips(symbol, 0, 0, profit_money, lot)

if expected_pips != 0:
    ratio = abs(ea_pips / expected_pips) if expected_pips != 0 else float('inf')
    if 0.2 <= ratio <= 5.0:
        return ea_pips

return expected_pips
```

# ======================================================================

# SETUP DIRECTORIES & LOGGING

# ======================================================================

for d in [DATA_DIR, LOGS_DIR, MODELS_DIR]:
os.makedirs(d, exist_ok=True)

# Configure logging with UTF-8 support for Windows

log_formatter = logging.Formatter(’%(asctime)s [%(levelname)s] %(message)s’)

# File handler with UTF-8

file_handler = logging.FileHandler(os.path.join(LOGS_DIR, ‘server_v12.log’), encoding=‘utf-8’)
file_handler.setFormatter(log_formatter)

# Console handler with UTF-8 (for Windows compatibility)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
try:
# Try to set UTF-8 encoding for console on Windows
import sys
if sys.platform == ‘win32’:
sys.stdout.reconfigure(encoding=‘utf-8’, errors=‘replace’)
sys.stderr.reconfigure(encoding=‘utf-8’, errors=‘replace’)
except:
pass

logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger(**name**)

# ======================================================================

# LOG OPTIMIZATION STATUS

# ======================================================================

# Log Numba status

if NUMBA_AVAILABLE:
logger.info(“✅ Numba JIT optimization enabled (CPU)”)
logger.info(“✅ Numba JIT functions compiled”)
else:
logger.warning(“⚠️  Numba not available. Install: pip install numba”)

# Log XGBoost status

if XGBOOST_AVAILABLE:
logger.info(“✅ XGBoost available for retraining”)
logger.info(“✅ Retrain engine classes loaded”)
else:
logger.warning(“⚠️  XGBoost not available. Install: pip install xgboost scikit-learn”)

# ======================================================================

# LOG OPTIMIZATION STATUS

# ======================================================================

# Log Numba status

if NUMBA_AVAILABLE:
logger.info(“✅ Numba JIT optimization enabled (CPU)”)
else:
logger.warning(“⚠️  Numba not available. Install: pip install numba”)

# Log XGBoost status

if XGBOOST_AVAILABLE:
logger.info(“✅ XGBoost available for retraining”)
else:
logger.warning(“⚠️  XGBoost not available. Install: pip install xgboost scikit-learn”)

# ======================================================================

# MODEL DEFINITIONS

# ======================================================================

if TORCH_AVAILABLE:

```
# -----------------------------------------------------------------
# MAMBA MODEL (V10 - 73.3%)
# -----------------------------------------------------------------

class SimplifiedMambaBlock(nn.Module):
    """Simplified Mamba block V10 - MUST MATCH TRAINING SCRIPT"""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        d_inner = dim * expand
        
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, dim, bias=False)
        
        self.A = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))
        
    def forward(self, x):
        b, l, d = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :l]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        x_dbl = self.x_proj(x_conv)
        delta, B = x_dbl.split([self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        
        y = x_conv * self.D + x_conv
        y = y * F.silu(z)
        return self.out_proj(y)

class BodhiMambaV10(nn.Module):
    """Bodhi Genesis V10 - M15 Entry Model 73.3%"""
    def __init__(self, input_dim=18, hidden_dim=192, num_layers=4, num_classes=3, d_state=16):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.blocks = nn.ModuleList([
            SimplifiedMambaBlock(hidden_dim, d_state=d_state)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)

# -----------------------------------------------------------------
# BiLSTM MODEL (74.2%)
# -----------------------------------------------------------------

class BodhiBiLSTM(nn.Module):
    """BiLSTM Model - 74.2% accuracy"""
    def __init__(self, input_dim=18, hidden_dim=192, num_layers=3, num_classes=3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        # BiLSTM: output = hidden_dim * 2 = 384
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        # Head: LayerNorm(384) -> Linear(384,192) -> ReLU -> Dropout -> Linear(192,3)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        return self.head(x)

# -----------------------------------------------------------------
# TRANSFORMER MODEL (66.9%)
# -----------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class BodhiTransformer(nn.Module):
    """Transformer Model - 66.9% accuracy"""
    def __init__(self, input_dim=18, hidden_dim=192, num_layers=4, num_heads=3, num_classes=3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Head: LayerNorm(192) -> Linear(192,96) -> ReLU -> Dropout -> Linear(96,3)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)

# -----------------------------------------------------------------
# PPO RISK VALIDATOR
# -----------------------------------------------------------------

class PPORiskValidator(nn.Module):
    """PPO-based Risk Validator - Trung Đạo"""
    def __init__(self, n_features=12, n_actions=3, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)
    
    def get_action(self, state, deterministic=True):
        action_probs, value = self.forward(state)
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
        return action, action_probs, value
```

# ======================================================================

# DATA LOGGER

# ======================================================================

class DataLogger:
“”“Log signals and trades to CSV for retraining”””

```
def __init__(self, data_dir=DATA_DIR):
    self.data_dir = data_dir
    self.trades_file = os.path.join(data_dir, 'trades_v12.csv')
    self.signals_file = os.path.join(data_dir, 'signals_v12.csv')
    self.lock = Lock()
    self._init_files()

def _init_files(self):
    """Initialize CSV files with headers"""
    if not os.path.exists(self.trades_file):
        with open(self.trades_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'symbol', 'type', 'lot', 'open_price', 'close_price',
                'profit_pips', 'profit_money', 'karma_before', 'karma_after',
                'is_clean', 'magic', 'duration_minutes',
                'ensemble_signal', 'meta_probability', 'had_consensus'
            ])
    
    if not os.path.exists(self.signals_file):
        with open(self.signals_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'symbol', 'final_signal', 'confidence',
                'v6_signal', 'ensemble_signal', 'meta_probability',
                'mamba_signal', 'lstm_signal', 'transformer_signal',
                'has_consensus', 'signal_strength', 'approved',
                'rsi_m15', 'rsi_h1', 'rsi_h4', 'adx_m15', 'main_trend'
            ])

def log_trade(self, data: dict):
    """Log trade to CSV"""
    with self.lock:
        try:
            with open(self.trades_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    data.get('symbol', ''),
                    data.get('type', ''),
                    data.get('lot', 0),
                    data.get('open_price', 0),
                    data.get('close_price', 0),
                    data.get('profit_pips', 0),
                    data.get('profit_money', 0),
                    data.get('karma_before', 0),
                    data.get('karma_after', 0),
                    data.get('is_clean', True),
                    data.get('magic', 0),
                    data.get('duration_minutes', 0),
                    data.get('ensemble_signal', 0.5),
                    data.get('meta_probability', 0.5),
                    data.get('had_consensus', False)
                ])
        except Exception as e:
            logger.error(f"Error logging trade: {e}")

def log_signal(self, data: dict):
    """Log signal to CSV"""
    with self.lock:
        try:
            with open(self.signals_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    data.get('symbol', ''),
                    data.get('final_signal', 0),
                    data.get('confidence', 0),
                    data.get('v6_signal', 0),
                    data.get('ensemble_signal', 0.5),
                    data.get('meta_probability', 0.5),
                    data.get('mamba_signal', 0.5),
                    data.get('lstm_signal', 0.5),
                    data.get('transformer_signal', 0.5),
                    data.get('has_consensus', False),
                    data.get('signal_strength', 'NONE'),
                    data.get('approved', True),
                    data.get('rsi_m15', 50),
                    data.get('rsi_h1', 50),
                    data.get('rsi_h4', 50),
                    data.get('adx_m15', 20),
                    data.get('main_trend', 0)
                ])
        except Exception as e:
            logger.error(f"Error logging signal: {e}")

def get_trade_count(self) -> int:
    """Get number of logged trades"""
    try:
        if os.path.exists(self.trades_file):
            with open(self.trades_file, 'r') as f:
                return sum(1 for _ in f) - 1  # Exclude header
    except:
        pass
    return 0

def get_trades_df(self) -> pd.DataFrame:
    """Get trades as DataFrame"""
    try:
        if os.path.exists(self.trades_file):
            return pd.read_csv(self.trades_file)
    except:
        pass
    return pd.DataFrame()
```

# ======================================================================

# RETRAIN ENGINE

# ======================================================================

if XGBOOST_AVAILABLE:
class RetrainEngine:
“”“Auto-retrain Meta-Labeler from collected trade data”””

```
    def __init__(self, data_logger, models_dir='models', min_records=1000, 
                 min_accuracy=0.65, improvement_threshold=0.02):
        self.data_logger = data_logger
        self.models_dir = models_dir
        self.min_records = min_records
        self.min_accuracy = min_accuracy
        self.improvement_threshold = improvement_threshold
        
        self.lock = Lock()
        self.last_retrain = None
        self.retrain_count = 0
        
        self.meta_model_path = os.path.join(models_dir, 'meta_labeler_real.pkl')
        self.backup_dir = os.path.join(models_dir, 'backups')
        os.makedirs(self.backup_dir, exist_ok=True)
        
        logger.info(f"[RETRAIN] Initialized (min={min_records}, acc={min_accuracy:.0%})")
    
    def get_status(self):
        trade_count = self.data_logger.get_trade_count()
        return {
            'last_retrain': self.last_retrain.isoformat() if self.last_retrain else None,
            'retrain_count': self.retrain_count,
            'trade_records': trade_count,
            'min_records_required': self.min_records,
            'ready_to_retrain': trade_count >= self.min_records,
            'model_exists': os.path.exists(self.meta_model_path)
        }
    
    def should_retrain(self):
        """Check if should retrain (Sunday 6-7 AM)"""
        trade_count = self.data_logger.get_trade_count()
        if trade_count < self.min_records:
            return False
        
        now = datetime.now()
        if now.weekday() == 6 and 6 <= now.hour < 7:
            if self.last_retrain and self.last_retrain.date() == now.date():
                return False
            return True
        
        return False
    
    def retrain(self):
        """Execute complete retrain workflow"""
        with self.lock:
            result = {
                'success': False,
                'message': '',
                'old_accuracy': None,
                'new_accuracy': None,
                'deployed': False
            }
            
            try:
                trade_count = self.data_logger.get_trade_count()
                if trade_count < self.min_records:
                    result['message'] = f"Not enough data: {trade_count}/{self.min_records}"
                    return result
                
                logger.info(f"[RETRAIN] Starting with {trade_count} records...")
                
                # Load trades
                df = pd.read_csv(self.data_logger.trades_file)
                df = df.dropna(subset=['profit_pips', 'ensemble_signal', 'meta_probability'])
                
                # Prepare features
                y = (df['profit_pips'] > 0).astype(int).values
                X = np.column_stack([
                    df['ensemble_signal'].fillna(0.5).values,
                    df['meta_probability'].fillna(0.5).values,
                    df['had_consensus'].fillna(0).astype(int).values,
                    df['is_clean'].fillna(1).astype(int).values,
                    df['karma_before'].fillna(50).values / 100.0
                ])
                
                # Split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Evaluate old model
                old_accuracy = 0.5
                if os.path.exists(self.meta_model_path):
                    old_model = joblib.load(self.meta_model_path)
                    y_pred = old_model.predict(X_val)
                    old_accuracy = accuracy_score(y_val, y_pred)
                
                result['old_accuracy'] = old_accuracy
                logger.info(f"[RETRAIN] Old accuracy: {old_accuracy:.2%}")
                
                # Train new model
                params = {
                    'objective': 'binary:logistic',
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'random_state': 42,
                    'n_jobs': -1
                }
                new_model = xgb.XGBClassifier(**params)
                new_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
                # Evaluate new model
                y_pred = new_model.predict(X_val)
                new_accuracy = accuracy_score(y_val, y_pred)
                result['new_accuracy'] = new_accuracy
                logger.info(f"[RETRAIN] New accuracy: {new_accuracy:.2%}")
                
                # Check if should deploy
                improvement = new_accuracy - old_accuracy
                should_deploy = (
                    new_accuracy >= self.min_accuracy and
                    improvement >= self.improvement_threshold
                )
                
                if should_deploy:
                    # Backup old model
                    if os.path.exists(self.meta_model_path):
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        backup_path = os.path.join(
                            self.backup_dir,
                            f'meta_labeler_backup_{timestamp}.pkl'
                        )
                        shutil.copy2(self.meta_model_path, backup_path)
                        logger.info(f"[RETRAIN] Backed up to: {backup_path}")
                    
                    # Deploy new model
                    joblib.dump(new_model, self.meta_model_path)
                    logger.info(f"[RETRAIN] Deployed new model")
                    
                    result['success'] = True
                    result['deployed'] = True
                    result['message'] = f"✅ Retrain successful! {old_accuracy:.2%} → {new_accuracy:.2%}"
                    
                    self.last_retrain = datetime.now()
                    self.retrain_count += 1
                else:
                    result['message'] = f"❌ Not deployed. Acc: {new_accuracy:.2%}, Need +{self.improvement_threshold:.0%}"
                
                logger.info(f"[RETRAIN] {result['message']}")
                return result
                
            except Exception as e:
                result['message'] = f"❌ Error: {str(e)}"
                logger.error(f"[RETRAIN] {result['message']}", exc_info=True)
                return result
    
    def manual_retrain(self):
        """Manually trigger retrain"""
        logger.info("[RETRAIN] Manual retrain triggered")
        return self.retrain()

class RetrainScheduler:
    """Background scheduler for auto-retrain"""
    
    def __init__(self, retrain_engine, check_interval=3600):
        self.retrain_engine = retrain_engine
        self.check_interval = check_interval
        self.running = False
        self.thread = None
    
    def start(self):
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"[SCHEDULER] Started (check every {self.check_interval}s)")
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("[SCHEDULER] Stopped")
    
    def _run(self):
        while self.running:
            try:
                if self.retrain_engine.should_retrain():
                    logger.info("[SCHEDULER] Triggering scheduled retrain...")
                    result = self.retrain_engine.retrain()
                    logger.info(f"[SCHEDULER] {result['message']}")
                
                time_module.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"[SCHEDULER] Error: {e}")
                time_module.sleep(self.check_interval)

logger.info("✅ Retrain engine classes loaded")
```

else:
class RetrainEngine:
def **init**(self, *args, **kwargs):
pass
def get_status(self):
return {‘error’: ‘XGBoost not installed’}
def manual_retrain(self):
return {‘success’: False, ‘message’: ‘XGBoost not installed’}

```
class RetrainScheduler:
    def __init__(self, *args, **kwargs):
        pass
    def start(self):
        pass
    def stop(self):
        pass
```

# ======================================================================

# KARMA ENGINE

# ======================================================================

class KarmaEngine:
“”“Nhân Quả Báo Ứng - Karma System”””

```
def __init__(self, data_dir=DATA_DIR):
    self.data_dir = data_dir
    self.karma_file = os.path.join(data_dir, 'karma_v12.json')
    self.karma = {s: {
        'points': 0, 'trades': 0, 'wins': 0, 'losses': 0,
        'consecutive_losses': 0, 'last_loss_time': None,
        'total_pips': 0, 'best_streak': 0, 'current_streak': 0
    } for s in SYMBOLS}
    self.lock = Lock()
    self._load()

def _load(self):
    """Load karma from file"""
    try:
        if os.path.exists(self.karma_file):
            with open(self.karma_file, 'r') as f:
                saved = json.load(f)
                for s in SYMBOLS:
                    if s in saved:
                        self.karma[s].update(saved[s])
            logger.info(f"[KARMA] Loaded karma from {self.karma_file}")
    except Exception as e:
        logger.warning(f"[KARMA] Load error: {e}")

def _save(self):
    """Save karma to file"""
    try:
        with open(self.karma_file, 'w') as f:
            json.dump(self.karma, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"[KARMA] Save error: {e}")

def get_karma(self, symbol: str) -> dict:
    s = normalize_symbol(symbol)
    k = self.karma.get(s, {'points': 0})
    level, mult = self.get_level(k.get('points', 0))
    return {**k, 'level': level, 'lot_multiplier': mult}

def get_all_karma(self) -> dict:
    return {s: self.get_karma(s) for s in SYMBOLS}

def get_level(self, points: int) -> tuple:
    for name, cfg in KARMA_LEVELS.items():
        if points >= cfg['min']:
            return name, cfg['mult']
    return 'SAMSARA', 0.5

def process_trade(self, data: dict) -> dict:
    """Process trade and update karma"""
    with self.lock:
        s = normalize_symbol(data.get('symbol', 'EURUSD'))
        k = self.karma[s]
        
        profit_pips = data.get('profit_pips', 0)
        is_clean = data.get('is_clean', True)
        
        karma_before = k['points']
        k['trades'] += 1
        k['total_pips'] += profit_pips
        
        if profit_pips > 0:
            k['wins'] += 1
            k['consecutive_losses'] = 0
            k['current_streak'] += 1
            k['best_streak'] = max(k['best_streak'], k['current_streak'])
            
            # Karma gain
            gain = min(profit_pips / 10, 5)
            if is_clean:
                gain *= 1.5
            k['points'] += gain
        else:
            k['losses'] += 1
            k['consecutive_losses'] += 1
            k['current_streak'] = 0
            k['last_loss_time'] = datetime.now().isoformat()
            
            # Karma loss
            loss = min(abs(profit_pips) / 5, 10)
            if not is_clean:
                loss *= 1.5
            k['points'] -= loss
        
        level, mult = self.get_level(k['points'])
        
        self._save()
        
        return {
            'symbol': s,
            'karma_before': karma_before,
            'karma_after': k['points'],
            'change': k['points'] - karma_before,
            'level': level,
            'lot_multiplier': mult,
            'consecutive_losses': k['consecutive_losses'],
            'win_rate': k['wins'] / k['trades'] * 100 if k['trades'] > 0 else 0
        }

def should_cooldown(self, symbol: str) -> tuple:
    """Check if symbol should be in cooldown"""
    s = normalize_symbol(symbol)
    k = self.karma.get(s, {})
    
    if k.get('consecutive_losses', 0) >= V6_CONFIG['max_consecutive_losses']:
        last_loss = k.get('last_loss_time')
        if last_loss:
            try:
                last_loss_dt = datetime.fromisoformat(last_loss)
                cooldown_until = last_loss_dt + timedelta(hours=V6_CONFIG['cooldown_hours'])
                if datetime.now() < cooldown_until:
                    return True, f"Cooldown until {cooldown_until.strftime('%H:%M')}"
            except:
                pass
        k['consecutive_losses'] = 0
        self._save()
    
    return False, ""
```

# ======================================================================

# PPO RISK ENGINE

# ======================================================================

class PPORiskEngine:
“”“PPO Risk Engine - Validates signals with Trung Đạo”””

```
def __init__(self, models_dir=MODELS_DIR):
    self.models_dir = models_dir
    self.model = None
    self.model_loaded = False
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
    
    self.config = {
        'max_trades_per_day': 10,
        'max_drawdown_atr': 1.5,
        'min_adx_trend': 20,
        'min_confidence': 60,
        'karma_threshold': -10,
    }
    
    self._load_model()

def _load_model(self):
    if not TORCH_AVAILABLE:
        return
    
    model_path = os.path.join(self.models_dir, 'ppo_best.pt')
    if os.path.exists(model_path):
        try:
            self.model = PPORiskValidator().to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['policy_state_dict'])
            self.model.eval()
            self.model_loaded = True
            logger.info(f"[PPO] Model loaded: {model_path}")
        except Exception as e:
            logger.warning(f"[PPO] Model load error: {e}")
    else:
        logger.info("[PPO] Model not found - using rule-based risk check")

def validate_signal(self, signal_data: dict) -> dict:
    """Validate signal with PPO/rules"""
    signal = signal_data.get('signal', 0)
    confidence = signal_data.get('confidence', 50)
    karma = signal_data.get('karma', 0)
    trades_today = signal_data.get('trades_today', 0)
    consecutive_losses = signal_data.get('consecutive_losses', 0)
    adx = signal_data.get('adx_m15', 25)
    
    # HOLD = no risk needed
    if signal == 0:
        return {
            'approved': True,
            'action': 'APPROVE',
            'lot_multiplier': 0,
            'reason': 'HOLD signal'
        }
    
    # Check karma
    if karma < self.config['karma_threshold']:
        return {
            'approved': False,
            'action': 'REJECT',
            'lot_multiplier': 0,
            'reason': f'Karma too low: {karma}'
        }
    
    # Check overtrading
    if trades_today >= self.config['max_trades_per_day']:
        return {
            'approved': False,
            'action': 'REJECT',
            'lot_multiplier': 0,
            'reason': f'Max trades reached: {trades_today}'
        }
    
    # Check consecutive losses (V6: 3 loss = cooldown)
    if consecutive_losses >= V6_CONFIG['max_consecutive_losses']:
        return {
            'approved': False,
            'action': 'REJECT',
            'lot_multiplier': 0,
            'reason': f'Consecutive losses: {consecutive_losses}'
        }
    
    # Check ADX (V6: ADX > 20)
    if adx < V6_CONFIG['adx_min']:
        return {
            'approved': False,
            'action': 'REJECT',
            'lot_multiplier': 0,
            'reason': f'Weak trend ADX: {adx:.0f}'
        }
    
    # Check confidence
    if confidence < self.config['min_confidence']:
        return {
            'approved': True,
            'action': 'REDUCE',
            'lot_multiplier': 0.5,
            'reason': f'Low confidence: {confidence:.0f}%'
        }
    
    # All checks passed
    return {
        'approved': True,
        'action': 'APPROVE',
        'lot_multiplier': 1.0,
        'reason': 'Risk OK'
    }
```

# ======================================================================

# ENSEMBLE ENGINE

# ======================================================================

class EnsembleEngine:
“”“Time-MoE Ensemble: Mamba + LSTM + Transformer với Meta-Labeler”””

```
def __init__(self, models_dir: str = MODELS_DIR):
    self.models_dir = models_dir
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
    
    # Models
    self.mamba_model = None
    self.lstm_model = None
    self.transformer_model = None
    self.meta_labeler = None
    
    # Status
    self.mamba_loaded = False
    self.lstm_loaded = False
    self.transformer_loaded = False
    self.meta_loaded = False
    
    self._load_models()

def _load_models(self):
    """Load all ensemble models"""
    # Load Mamba
    for path in [os.path.join(self.models_dir, 'bodhi_v9_m15_final.pt'), 'bodhi_v9_m15_final.pt']:
        if os.path.exists(path):
            self._load_mamba(path)
            break
    
    # Load LSTM
    for path in [os.path.join(self.models_dir, 'bodhi_lstm_ensemble.pt'), 'bodhi_lstm_ensemble.pt']:
        if os.path.exists(path):
            self._load_lstm(path)
            break
    
    # Load Transformer
    for path in [os.path.join(self.models_dir, 'bodhi_transformer_ensemble.pt'), 'bodhi_transformer_ensemble.pt']:
        if os.path.exists(path):
            self._load_transformer(path)
            break
    
    # Load Meta-Labeler
    for path in [os.path.join(self.models_dir, 'meta_labeler_real.joblib'), 'meta_labeler_real.joblib']:
        if os.path.exists(path):
            self._load_meta_labeler(path)
            break
    
    loaded_count = sum([self.mamba_loaded, self.lstm_loaded, self.transformer_loaded])
    logger.info(f"[ENSEMBLE] Loaded {loaded_count}/3 models: "
               f"Mamba={self.mamba_loaded}(73.3%), "
               f"LSTM={self.lstm_loaded}(74.2%), "
               f"Trans={self.transformer_loaded}(66.9%), "
               f"Meta={self.meta_loaded}")

def _load_mamba(self, path: str):
    if not TORCH_AVAILABLE or self.device is None:
        return
    try:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg = ckpt.get('config', {})
        self.mamba_model = BodhiMambaV10(
            cfg.get('input_dim', 18), cfg.get('hidden_dim', 192),
            cfg.get('num_layers', 4), 3, cfg.get('d_state', 16)
        ).to(self.device)
        self.mamba_model.load_state_dict(ckpt['model_state_dict'])
        self.mamba_model.eval()
        self.mamba_loaded = True
        logger.info(f"[ENSEMBLE] [OK] Mamba loaded from {path}")
    except Exception as e:
        logger.error(f"[ENSEMBLE] [X] Mamba error: {e}")

def _load_lstm(self, path: str):
    if not TORCH_AVAILABLE or self.device is None:
        return
    try:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg = ckpt.get('config', {})
        self.lstm_model = BodhiBiLSTM(
            cfg.get('input_dim', 18), cfg.get('hidden_dim', 192),
            cfg.get('num_layers', 3), 3
        ).to(self.device)
        self.lstm_model.load_state_dict(ckpt['model_state_dict'])
        self.lstm_model.eval()
        self.lstm_loaded = True
        logger.info(f"[ENSEMBLE] [OK] LSTM loaded from {path}")
    except Exception as e:
        logger.error(f"[ENSEMBLE] [X] LSTM error: {e}")

def _load_transformer(self, path: str):
    if not TORCH_AVAILABLE or self.device is None:
        return
    try:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg = ckpt.get('config', {})
        self.transformer_model = BodhiTransformer(
            cfg.get('input_dim', 18), cfg.get('hidden_dim', 192),
            cfg.get('num_layers', 4), cfg.get('num_heads', 3), 3
        ).to(self.device)
        self.transformer_model.load_state_dict(ckpt['model_state_dict'])
        self.transformer_model.eval()
        self.transformer_loaded = True
        logger.info(f"[ENSEMBLE] [OK] Transformer loaded from {path}")
    except Exception as e:
        logger.error(f"[ENSEMBLE] [X] Transformer error: {e}")

def _load_meta_labeler(self, path: str):
    try:
        self.meta_labeler = joblib.load(path)
        self.meta_loaded = True
        logger.info(f"[ENSEMBLE] [OK] Meta-Labeler loaded from {path}")
    except Exception as e:
        logger.error(f"[ENSEMBLE] [X] Meta-Labeler error: {e}")

def _predict_model(self, model, features: np.ndarray) -> Tuple[float, float]:
    """Generic model prediction"""
    if model is None:
        return 0.5, 50.0
    try:
        if features.ndim == 2:
            features = features.reshape(1, *features.shape)
        X = torch.FloatTensor(features).to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(model(X), dim=1).cpu().numpy()[0]
            # probs: [SELL, HOLD, BUY]
            signal = (probs[2] - probs[0] + 1) / 2  # 0-1 scale
            confidence = float(max(probs) * 100)
        return signal, confidence
    except Exception as e:
        logger.error(f"Model prediction error: {e}")
        return 0.5, 50.0

def get_ensemble_prediction(self, features: np.ndarray) -> Dict:
    """Get weighted ensemble prediction from all models"""
    mamba_sig, mamba_conf = self._predict_model(self.mamba_model, features)
    lstm_sig, lstm_conf = self._predict_model(self.lstm_model, features)
    trans_sig, trans_conf = self._predict_model(self.transformer_model, features)
    
    # Weighted ensemble (only use loaded models)
    w = ENSEMBLE_CONFIG
    total_w, weighted_sig, weighted_conf = 0, 0, 0
    
    if self.mamba_loaded:
        weighted_sig += w['mamba_weight'] * mamba_sig
        weighted_conf += w['mamba_weight'] * mamba_conf
        total_w += w['mamba_weight']
    if self.lstm_loaded:
        weighted_sig += w['lstm_weight'] * lstm_sig
        weighted_conf += w['lstm_weight'] * lstm_conf
        total_w += w['lstm_weight']
    if self.transformer_loaded:
        weighted_sig += w['transformer_weight'] * trans_sig
        weighted_conf += w['transformer_weight'] * trans_conf
        total_w += w['transformer_weight']
    
    ens_sig = weighted_sig / total_w if total_w > 0 else 0.5
    ens_conf = weighted_conf / total_w if total_w > 0 else 50
    
    # Check consensus
    buy_threshold = w['ensemble_buy_threshold']
    sell_threshold = w['ensemble_sell_threshold']
    
    buy_votes = sum([
        self.mamba_loaded and mamba_sig > buy_threshold,
        self.lstm_loaded and lstm_sig > buy_threshold,
        self.transformer_loaded and trans_sig > buy_threshold
    ])
    sell_votes = sum([
        self.mamba_loaded and mamba_sig < sell_threshold,
        self.lstm_loaded and lstm_sig < sell_threshold,
        self.transformer_loaded and trans_sig < sell_threshold
    ])
    
    loaded_models = sum([self.mamba_loaded, self.lstm_loaded, self.transformer_loaded])
    consensus_threshold = min(2, loaded_models)
    has_consensus = buy_votes >= consensus_threshold or sell_votes >= consensus_threshold
    
    # Determine action
    if ens_sig > buy_threshold and (not w['require_consensus'] or buy_votes >= consensus_threshold):
        action, action_code = 'BUY', 1
    elif ens_sig < sell_threshold and (not w['require_consensus'] or sell_votes >= consensus_threshold):
        action, action_code = 'SELL', -1
    else:
        action, action_code = 'HOLD', 0
    
    return {
        'ensemble_signal': ens_sig,
        'ensemble_confidence': ens_conf,
        'action': action,
        'action_code': action_code,
        'mamba': {'signal': mamba_sig, 'confidence': mamba_conf, 'loaded': self.mamba_loaded},
        'lstm': {'signal': lstm_sig, 'confidence': lstm_conf, 'loaded': self.lstm_loaded},
        'transformer': {'signal': trans_sig, 'confidence': trans_conf, 'loaded': self.transformer_loaded},
        'has_consensus': has_consensus,
        'buy_votes': buy_votes,
        'sell_votes': sell_votes,
        'loaded_models': loaded_models
    }

def get_meta_probability(self, market_data: Dict, ensemble_result: Dict) -> float:
    """Get Meta-Labeler probability of profitable trade"""
    if not self.meta_loaded:
        return self._rule_based_meta(market_data, ensemble_result)
    
    try:
        features = self._build_meta_features(market_data, ensemble_result)
        proba = self.meta_labeler.predict_proba(features)[0][1]
        return float(proba)
    except Exception as e:
        logger.error(f"[META] Prediction error: {e}")
        return self._rule_based_meta(market_data, ensemble_result)

def _build_meta_features(self, data: Dict, ensemble: Dict) -> pd.DataFrame:
    """Build feature vector for Meta-Labeler"""
    hour = datetime.now().hour
    symbol = data.get('symbol', 'EURUSD')
    
    def rsi_dir(r):
        return 1 if r < 45 else (-1 if r > 55 else 0)
    
    rsi_m5 = data.get('rsi_m5', 50)
    rsi_m15 = data.get('rsi_m15', 50)
    rsi_h1 = data.get('rsi_h1', 50)
    rsi_h4 = data.get('rsi_h4', 50)
    rsi_vals = [rsi_m5, rsi_m15, rsi_h1, rsi_h4]
    
    session = SYMBOL_SESSIONS.get(symbol, {'start': 9, 'end': 20})
    in_session = session['start'] <= hour <= session['end']
    
    features = {
        'ai_signal_encoded': ensemble.get('action_code', 0),
        'ai_confidence': ensemble.get('ensemble_confidence', 50) / 100,
        'followed_ai': 1,
        'rsi_m5': rsi_m5, 'rsi_m15': rsi_m15, 'rsi_h1': rsi_h1, 'rsi_h4': rsi_h4,
        'rsi_m5_direction': rsi_dir(rsi_m5),
        'rsi_m15_direction': rsi_dir(rsi_m15),
        'rsi_h1_direction': rsi_dir(rsi_h1),
        'rsi_h4_direction': rsi_dir(rsi_h4),
        'rsi_avg': np.mean(rsi_vals),
        'rsi_std': np.std(rsi_vals),
        'rsi_range': max(rsi_vals) - min(rsi_vals),
        'adx_m5': data.get('adx_m5', 20),
        'adx_h4': data.get('adx_h4', 20),
        'adx_avg': (data.get('adx_m5', 20) + data.get('adx_h4', 20)) / 2,
        'adx_trending': int(data.get('adx_h4', 20) > 20),
        'tu_hop_nhat': int(ensemble.get('has_consensus', False)),
        'tf_agreement': max(ensemble.get('buy_votes', 0), ensemble.get('sell_votes', 0)),
        'karma_before': data.get('karma', 0),
        'karma_after': 0, 'karma_change': 0,
        'karma_positive': int(data.get('karma', 0) > 0),
        'atr': data.get('atr', 0.001),
        'drawdown_pips': 0, 'had_sl': 1, 'had_tp': 1, 'has_risk_management': 1,
        'is_fomo': 0,
        'is_revenge': int(data.get('consecutive_losses', 0) >= 2),
        'is_clean': 1, 'bad_behavior': 0,
        'hour': hour,
        'day_of_week': datetime.now().weekday(),
        'trades_today': data.get('trades_today', 0),
        'is_trading_session': int(in_session),
        'is_asian': int(0 <= hour <= 8),
        'is_london': int(8 <= hour <= 16),
        'is_newyork': int(13 <= hour <= 21),
        'is_overlap': int(13 <= hour <= 16),
        'lot': 0.01, 'duration_mins': 0,
        'symbol_EURUSD': int(symbol == 'EURUSD'),
        'symbol_GBPUSD': int(symbol == 'GBPUSD'),
        'symbol_US30': int(symbol == 'US30'),
        'symbol_XAUUSD': int(symbol == 'XAUUSD'),
    }
    
    df = pd.DataFrame([features])
    
    try:
        expected_cols = self.meta_labeler.feature_names_in_
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]
    except:
        pass
    
    return df

def _rule_based_meta(self, data: Dict, ensemble: Dict) -> float:
    """Fallback rule-based meta prediction"""
    prob = 0.5
    
    if ensemble.get('has_consensus', False):
        prob += 0.15
    
    hour = datetime.now().hour
    symbol = data.get('symbol', 'EURUSD')
    session = SYMBOL_SESSIONS.get(symbol, {'start': 9, 'end': 20})
    if session['start'] <= hour <= session['end']:
        prob += 0.10
    
    signal_strength = abs(ensemble.get('ensemble_signal', 0.5) - 0.5) * 2
    prob += signal_strength * 0.1
    
    if data.get('adx_h4', 20) > 25:
        prob += 0.05
    
    if data.get('consecutive_losses', 0) >= 2:
        prob -= 0.20
    
    return max(0, min(1, prob))
```

# ======================================================================

# RETRAIN ENGINE

# ======================================================================

# ======================================================================

# V6 TỨ HỢP NHẤT LOGIC

# ======================================================================

def check_tu_hop_nhat(data: dict) -> tuple:
“””
V6 TỨ HỢP NHẤT - TẤT CẢ PHẢI ĐỒNG THUẬN MỚI VÀO LỆNH

```
THIÊN (D1) = Trời  -> main_trend
ĐỊA (H4)   = Đất   -> rsi_h4, adx_h4
NHÂN (H1)  = Người -> rsi_h1
THỜI (M15) = Timing -> rsi_m15, adx_m15
"""
main_trend = data.get('main_trend', 0)

rsi_d1 = data.get('rsi_d1', 50)
rsi_h4 = data.get('rsi_h4', 50)
rsi_h1 = data.get('rsi_h1', 50)
rsi_m15 = data.get('rsi_m15', 50)

adx_h4 = data.get('adx_h4', 20)
adx_h1 = data.get('adx_h1', 20)
adx_m15 = data.get('adx_m15', 20)

# Check ADX trend strength
has_trend = adx_m15 >= V6_CONFIG['adx_min']

# BUY CONDITIONS
thien_buy = main_trend >= 0  # D1: KHÔNG đánh ngược THIÊN!
dia_buy = rsi_h4 > 40 and rsi_h4 < 70
nhan_buy = rsi_h1 > 45
thoi_buy = rsi_m15 < V6_CONFIG['rsi_buy_max']  # RSI < 45

buy_signal = thien_buy and dia_buy and nhan_buy and thoi_buy and has_trend

# SELL CONDITIONS
thien_sell = main_trend <= 0
dia_sell = rsi_h4 < 60 and rsi_h4 > 30
nhan_sell = rsi_h1 < 55
thoi_sell = rsi_m15 > V6_CONFIG['rsi_sell_min']  # RSI > 55

sell_signal = thien_sell and dia_sell and nhan_sell and thoi_sell and has_trend

if buy_signal:
    trend_str = "BULL" if main_trend == 1 else "NEUT"
    reason = f"TỨ HỢP NHẤT BUY | THIÊN:{trend_str} ĐỊA:H4={rsi_h4:.0f} NHÂN:H1={rsi_h1:.0f} THỜI:M15={rsi_m15:.0f}"
    return 1, reason

if sell_signal:
    trend_str = "BEAR" if main_trend == -1 else "NEUT"
    reason = f"TỨ HỢP NHẤT SELL | THIÊN:{trend_str} ĐỊA:H4={rsi_h4:.0f} NHÂN:H1={rsi_h1:.0f} THỜI:M15={rsi_m15:.0f}"
    return -1, reason

# HOLD
reasons = []
if not has_trend:
    reasons.append(f"ADX={adx_m15:.0f}<{V6_CONFIG['adx_min']}")
if main_trend == -1 and rsi_m15 < 45:
    reasons.append("THIÊN BEARISH blocks BUY")
if main_trend == 1 and rsi_m15 > 55:
    reasons.append("THIÊN BULLISH blocks SELL")

reason = f"HOLD | {', '.join(reasons) if reasons else 'No setup'}"
return 0, reason
```

# ======================================================================

# PYDANTIC MODELS

# ======================================================================

class SignalRequest(BaseModel):
symbol: str = “EURUSD”
main_trend: int = 0
rsi_d1: float = 50
rsi_h4: float = 50
rsi_h1: float = 50
rsi_m15: float = 50
rsi_m5: float = 50
adx_h4: float = 20
adx_h1: float = 20
adx_m15: float = 20
adx_m5: float = 20
atr: float = 0.001
current_price: float = 0
tema_d1: float = 0
karma: int = 0
trades_today: int = 0
consecutive_losses: int = 0
sila_streak: int = 0

class TradeRequest(BaseModel):
symbol: str = “EURUSD”
type: str = “BUY”
lot: float = 0.01
profit_pips: float = 0
profit_money: float = 0
open_price: float = 0
close_price: float = 0
is_clean: bool = True
magic: int = 0
duration_minutes: int = 0

# ======================================================================

# LIFESPAN (replaces deprecated on_event)

# ======================================================================

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
“”“Lifespan context manager - startup and shutdown”””
global ensemble_engine
# Startup
ensemble_engine = EnsembleEngine(MODELS_DIR)
logger.info(f”[STARTUP] Bodhi Genesis V12 ready on port {DEFAULT_PORT}”)
yield
# Shutdown
logger.info(”[SHUTDOWN] Server stopping…”)

# ======================================================================

# FASTAPI APP

# ======================================================================

app = FastAPI(
title=”[BODHI] Bodhi Genesis V12 - Complete Ensemble”,
description=“Mamba (73.3%) + LSTM (74.2%) + Transformer (66.9%) + Meta-Labeler + PPO Risk”,
version=VERSION,
lifespan=lifespan
)

app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],
allow_credentials=True,
allow_methods=[”*”],
allow_headers=[”*”],
)

# Global instances

data_logger = DataLogger()
karma_engine = KarmaEngine()
retrain_engine = RetrainEngine(data_logger)
ppo_engine = PPORiskEngine() if TORCH_AVAILABLE else None
ensemble_engine = None

# Shadow Portfolio Manager (NEW)

try:
shadow_manager = ShadowPortfolioManager()
logger.info(f”[SHADOW] Shadow Portfolios initialized (CSV mode)”)
logger.info(f”[SHADOW] Active portfolios: {list(shadow_manager.portfolios.keys())}”)
except Exception as e:
logger.error(f”[SHADOW] Initialization failed: {e}”)
shadow_manager = None

# Cache last signal per symbol

last_signals = {}

# ======================================================================

# ENDPOINTS

# ======================================================================

@app.get(”/”)
async def root():
return {
“name”: “[BODHI] Bodhi Genesis V12 - Complete Ensemble”,
“version”: VERSION,
“philosophy”: “TỨ HỢP NHẤT - Thien Dia Nhan Hop Nhat”,
“pipeline”: “V6 -> Ensemble -> Meta-Labeler -> PPO -> Trade -> Karma”,
“entry_tf”: “M15”,
“models”: {
“mamba”: {“loaded”: ensemble_engine.mamba_loaded if ensemble_engine else False, “accuracy”: “73.3%”, “weight”: “35%”},
“lstm”: {“loaded”: ensemble_engine.lstm_loaded if ensemble_engine else False, “accuracy”: “74.2%”, “weight”: “40%”},
“transformer”: {“loaded”: ensemble_engine.transformer_loaded if ensemble_engine else False, “accuracy”: “66.9%”, “weight”: “25%”},
“meta_labeler”: {“loaded”: ensemble_engine.meta_loaded if ensemble_engine else False},
“ppo_risk”: {“loaded”: ppo_engine.model_loaded if ppo_engine else False}
},
“sessions”: {s: f”{v[‘start’]}:00-{v[‘end’]}:00” for s, v in SYMBOL_SESSIONS.items()},
“trade_records”: data_logger.get_trade_count(),
“status”: “online”
}

@app.get(”/health”)
@app.get(”/api/health”)
async def health():
models_loaded = sum([
ensemble_engine.mamba_loaded if ensemble_engine else False,
ensemble_engine.lstm_loaded if ensemble_engine else False,
ensemble_engine.transformer_loaded if ensemble_engine else False
])
return {
“status”: “ok”,
“version”: VERSION,
“models_loaded”: f”{models_loaded}/3”,
“meta_loaded”: ensemble_engine.meta_loaded if ensemble_engine else False,
“ppo_loaded”: ppo_engine.model_loaded if ppo_engine else False,
“timestamp”: datetime.now().isoformat()
}

@app.api_route(”/api/heartbeat”, methods=[“GET”, “POST”])
@app.api_route(”/heartbeat”, methods=[“GET”, “POST”])
async def heartbeat(request: Request):
data = {}
if request.method == “POST”:
try:
data = await request.json()
except:
pass

```
raw_symbol = data.get('symbol', 'EURUSD')
symbol = normalize_symbol(raw_symbol)

models_loaded = sum([
    ensemble_engine.mamba_loaded if ensemble_engine else False,
    ensemble_engine.lstm_loaded if ensemble_engine else False,
    ensemble_engine.transformer_loaded if ensemble_engine else False
])

cached = last_signals.get(symbol, {})

return {
    "status": "ok",
    "server": "ONLINE",
    "version": VERSION,
    "entry_tf": "M15",
    "models_loaded": f"{models_loaded}/3",
    "meta_loaded": ensemble_engine.meta_loaded if ensemble_engine else False,
    "ppo_loaded": ppo_engine.model_loaded if ppo_engine else False,
    "ai_model": cached.get("signal", "READY"),
    "last_signal": cached.get("signal", "READY"),
    "ai_confidence": cached.get("confidence", 73.3),
    "meta_probability": cached.get("meta_probability", 0.5),
    "trade_records": data_logger.get_trade_count(),
    "v6_rules": "RSI<45=BUY, RSI>55=SELL, ADX>20",
    "timestamp": datetime.now().isoformat()
}
```

@app.api_route(”/api/signal”, methods=[“GET”, “POST”])
@app.api_route(”/signal”, methods=[“GET”, “POST”])
async def get_signal(request: Request):
“””
V13 SIGNAL - Full Pipeline with Phase 2 Advanced Features

```
Pipeline:
  1. Check cooldown (V6: 3 loss = cooldown)
  2. Check session (symbol-specific)
  3. [NEW] Kalman Filter - Denoise RSI/ADX signals
  4. [NEW] Polynomial Features - Feature engineering
  5. V6 TU HOP NHAT logic (uses filtered values)
  6. Ensemble prediction (Mamba + LSTM + Transformer)
  7. Meta-Labeler filtering
  8. [NEW] Monte Carlo Risk Simulation
  9. PPO Risk validation
  10. Return final decision
"""
data = {}
if request.method == "POST":
    try:
        data = await request.json()
    except:
        pass

raw_symbol = data.get('symbol', 'EURUSD')
symbol = normalize_symbol(raw_symbol)

# Extract data - BACKWARD COMPATIBLE
rsi_m5 = float(data.get('rsi_m5', 50))
rsi_m15 = float(data.get('rsi_m15', 50))
rsi_h1 = float(data.get('rsi_h1', 50))
rsi_h4 = float(data.get('rsi_h4', 50))
rsi_d1 = float(data.get('rsi_d1', rsi_h4))

adx_m5 = float(data.get('adx_m5', 20))
adx_m15 = float(data.get('adx_m15', adx_m5))
adx_h4 = float(data.get('adx_h4', 20))
adx_h1 = float(data.get('adx_h1', adx_h4))
atr = float(data.get('atr', 0.001))

main_trend = int(data.get('main_trend', 0))
tema_d1 = float(data.get('tema_d1', 0))
current_price = float(data.get('current_price', 0))

trades_today = int(data.get('trades_today', 0))
consecutive_losses = int(data.get('consecutive_losses', 0))
karma_from_ea = int(data.get('karma', 0))

# ===================================================================
# STEP 1: COOLDOWN CHECK
# ===================================================================
in_cooldown, cooldown_reason = karma_engine.should_cooldown(symbol)
if in_cooldown:
    logger.warning(f"[COOLDOWN] {symbol} | {cooldown_reason}")
    return {
        "signal": 0, "signal_name": "HOLD", "confidence": 0,
        "reason": cooldown_reason, "approved": False,
        "meta_probability": 0, "signal_strength": "COOLDOWN",
        "symbol": symbol, "timestamp": datetime.now().isoformat()
    }

# ===================================================================
# STEP 2: SESSION CHECK
# ===================================================================
current_hour = datetime.now().hour
session = SYMBOL_SESSIONS.get(symbol, {'start': 9, 'end': 20})

if not (session['start'] <= current_hour < session['end']):
    return {
        "signal": 0, "signal_name": "HOLD", "confidence": 0,
        "reason": f"Outside session ({session['start']}:00-{session['end']}:00, now={current_hour}:00)",
        "approved": False, "meta_probability": 0, "signal_strength": "OUTSIDE_SESSION",
        "symbol": symbol, "timestamp": datetime.now().isoformat()
    }

# ===================================================================
# STEP 3: PHASE 2 - KALMAN FILTER (Denoise signals)
# ===================================================================
kalman_data = {}
if PHASE2_CONFIG['kalman_enabled']:
    raw_indicators = {
        'rsi_m15': rsi_m15, 'rsi_h1': rsi_h1, 'rsi_h4': rsi_h4,
        'adx_m15': adx_m15, 'adx_h4': adx_h4
    }
    kalman_data = kalman_bank.filter_indicators(symbol, raw_indicators)
    
    # Use filtered values for V6 check
    rsi_m15_filtered = kalman_data.get('rsi_m15_kalman', rsi_m15)
    rsi_h1_filtered = kalman_data.get('rsi_h1_kalman', rsi_h1)
    rsi_h4_filtered = kalman_data.get('rsi_h4_kalman', rsi_h4)
    adx_m15_filtered = kalman_data.get('adx_m15_kalman', adx_m15)
    
    logger.info(f"[KALMAN] {symbol} | RSI M15: {rsi_m15:.1f} -> {rsi_m15_filtered:.1f} | "
               f"ADX: {adx_m15:.1f} -> {adx_m15_filtered:.1f}")
else:
    rsi_m15_filtered = rsi_m15
    rsi_h1_filtered = rsi_h1
    rsi_h4_filtered = rsi_h4
    adx_m15_filtered = adx_m15

# ===================================================================
# STEP 4: PHASE 2 - POLYNOMIAL FEATURES
# ===================================================================
poly_data = {}
if PHASE2_CONFIG['polynomial_enabled']:
    poly_input = {
        'rsi_m15': rsi_m15_filtered, 'rsi_h1': rsi_h1_filtered, 
        'rsi_h4': rsi_h4_filtered, 'adx_m15': adx_m15_filtered, 
        'adx_h4': adx_h4, 'main_trend': main_trend
    }
    poly_data = poly_features.generate(poly_input)

# ===================================================================
# STEP 5: TREND STRENGTH
# ===================================================================
tema_dist_pct = 0
if tema_d1 > 0 and current_price > 0:
    tema_dist_pct = abs((current_price - tema_d1) / tema_d1 * 100)

trend_strength = get_trend_strength(adx_h4, rsi_h4_filtered, tema_dist_pct)
trend_params = get_trend_params(trend_strength)

# ===================================================================
# STEP 6: V6 TU HOP NHAT (uses filtered values)
# ===================================================================
v6_data = {
    'main_trend': main_trend,
    'rsi_d1': rsi_d1, 'rsi_h4': rsi_h4_filtered, 
    'rsi_h1': rsi_h1_filtered, 'rsi_m15': rsi_m15_filtered,
    'adx_h4': adx_h4, 'adx_h1': adx_h1, 'adx_m15': adx_m15_filtered,
}
signal_v6, reason_v6 = check_tu_hop_nhat(v6_data)
logger.info(f"[V6] {symbol} | {reason_v6}")

# ===================================================================
# STEP 7: ENSEMBLE PREDICTION
# ===================================================================
# ENHANCED: Using 5 polynomial features instead of 3
# Positions 2-3: Now using rsi_std and rsi_range (were placeholders 0.5)
features_single = [
    rsi_m15_filtered / 100, 
    adx_m15_filtered / 100, 
    poly_data.get('rsi_std', 0) / 100,      # Position 2: RSI volatility (was 0.5)
    poly_data.get('rsi_range', 0) / 100,    # Position 3: RSI spread (was 0.5)
    rsi_h1_filtered / 100, 
    1 if main_trend >= 0 else -1, 
    adx_h1 / 100,
    rsi_h4_filtered / 100, 
    1 if main_trend >= 0 else -1, 
    adx_h4 / 100,
    rsi_d1 / 100, 
    main_trend,
    current_hour / 24, 
    datetime.now().weekday() / 6, 
    datetime.now().minute / 60,
    poly_data.get('trend_alignment', 0.5),
    poly_data.get('rsi_momentum', 0) / 100,
    poly_data.get('adx_category', 0.5)
][:18]
features = np.array([features_single for _ in range(20)])

ensemble_result = ensemble_engine.get_ensemble_prediction(features)

logger.info(f"[ENSEMBLE] {symbol} | Mamba={ensemble_result['mamba']['signal']:.3f} "
           f"LSTM={ensemble_result['lstm']['signal']:.3f} "
           f"Trans={ensemble_result['transformer']['signal']:.3f} "
           f"-> {ensemble_result['action']} (consensus={ensemble_result['has_consensus']})")

# ===================================================================
# STEP 8: META-LABELER
# ===================================================================
market_data = {
    'symbol': symbol,
    'rsi_m5': rsi_m5, 'rsi_m15': rsi_m15_filtered, 'rsi_h1': rsi_h1_filtered, 'rsi_h4': rsi_h4_filtered,
    'adx_m5': adx_m5, 'adx_h4': adx_h4, 'atr': atr,
    'karma': karma_from_ea, 'trades_today': trades_today,
    'consecutive_losses': consecutive_losses
}
meta_probability = ensemble_engine.get_meta_probability(market_data, ensemble_result)
logger.info(f"[META] {symbol} | Probability={meta_probability:.2%}")

# ===================================================================
# STEP 9: PHASE 2 - MONTE CARLO RISK SIMULATION
# ===================================================================
monte_carlo_result = {}
if PHASE2_CONFIG['monte_carlo_enabled'] and signal_v6 != 0 and current_price > 0:
    # Calculate SL/TP distances based on ATR
    sl_distance = atr * trend_params['sl_atr']
    tp_distance = atr * trend_params['tp1_atr']
    
    monte_carlo_result = monte_carlo.simulate_trade(
        symbol=symbol,
        signal=signal_v6,
        entry_price=current_price,
        sl_distance=sl_distance,
        tp_distance=tp_distance,
        signal_strength=trend_strength,
        confidence=ensemble_result['ensemble_confidence'] / 100
    )
    
    logger.info(f"[MONTE_CARLO] {symbol} | Win Prob={monte_carlo_result['win_probability']:.1%} "
               f"RR={monte_carlo_result['risk_reward']:.2f} "
               f"Rec={monte_carlo_result['recommendation']}")

# ===================================================================
# STEP 10: FINAL DECISION
# ===================================================================
final_signal = signal_v6
final_confidence = ensemble_result['ensemble_confidence']
approved = True
lot_multiplier = 1.0

# Determine signal strength from Meta-Labeler
meta_cfg = ENSEMBLE_CONFIG
if meta_probability >= meta_cfg['meta_strong_threshold']:
    signal_strength = "STRONG"
    lot_multiplier = 1.0
elif meta_probability >= meta_cfg['meta_moderate_threshold']:
    signal_strength = "MODERATE"
    lot_multiplier = 0.75
elif meta_probability >= meta_cfg['meta_minimum_threshold']:
    signal_strength = "WEAK"
    lot_multiplier = 0.5
else:
    signal_strength = "FILTERED"
    if final_signal != 0:
        final_signal = 0
        approved = False
        lot_multiplier = 0

# PHASE 2: Monte Carlo LOT ADJUSTMENT (không filter, chỉ adjust lot)
mc_adjustment = 1.0
if monte_carlo_result and PHASE2_CONFIG['monte_carlo_enabled'] and final_signal != 0:
    mc_win_prob = monte_carlo_result.get('win_probability', 0.5)
    mc_recommendation = monte_carlo_result.get('recommendation', 'NORMAL_ENTRY')
    mc_risk_score = monte_carlo_result.get('risk_score', 50)
    
    # LOT ADJUSTMENT based on Monte Carlo (KHÔNG FILTER)
    if mc_recommendation == 'STRONG_ENTRY':
        mc_adjustment = 1.25  # Tăng 25% lot
        logger.info(f"[MC_ADJUST] {symbol} | STRONG_ENTRY -> lot x1.25")
    elif mc_recommendation == 'NORMAL_ENTRY':
        mc_adjustment = 1.0   # Giữ nguyên
    elif mc_recommendation == 'WEAK_ENTRY':
        mc_adjustment = 0.75  # Giảm 25% lot
        logger.info(f"[MC_ADJUST] {symbol} | WEAK_ENTRY -> lot x0.75")
    elif mc_recommendation == 'AVOID':
        mc_adjustment = 0.5   # Giảm 50% lot (vẫn trade, chỉ nhỏ hơn)
        logger.info(f"[MC_ADJUST] {symbol} | AVOID -> lot x0.5 (still trading)")
    
    # Fine-tune based on win probability
    if mc_win_prob >= 0.65:
        mc_adjustment *= 1.1  # High confidence boost
    elif mc_win_prob < 0.40:
        mc_adjustment *= 0.8  # Low confidence reduce
    
    lot_multiplier *= mc_adjustment
    
    logger.info(f"[MC_RESULT] {symbol} | Win={mc_win_prob:.1%} RR={monte_carlo_result.get('risk_reward', 0):.2f} "
               f"Rec={mc_recommendation} -> lot_adj={mc_adjustment:.2f}")

# Reduce confidence if no consensus
if not ensemble_result['has_consensus'] and final_signal != 0:
    final_confidence *= 0.8
    lot_multiplier *= 0.75

# Reduce confidence if V6 vs Ensemble disagree
if signal_v6 != 0 and ensemble_result['action_code'] != signal_v6:
    final_confidence *= 0.7

# ===================================================================
# STEP 10B: PHASE 3 - SENTIMENT CHECK (NHAN)
# ===================================================================
sentiment_data = {}
sentiment_adjustment = 1.0
if PHASE3_CONFIG['sentiment_enabled'] and final_signal != 0:
    sentiment_data = sentiment_reader.get_sentiment()
    should_trade, sent_mult = sentiment_reader.should_trade_with_sentiment(final_signal)
    sentiment_adjustment = sent_mult
    lot_multiplier *= sentiment_adjustment
    
    logger.info(f"[SENTIMENT] {symbol} | Score={sentiment_data.get('score', 0):.2f} "
               f"Status={sentiment_data.get('status', 'N/A')} -> lot_adj={sentiment_adjustment:.2f}")

# ===================================================================
# STEP 10C: PHASE 3 - KELLY CRITERION (Dynamic Lot Sizing)
# ===================================================================
kelly_result = {}
kelly_adjustment = 1.0
if PHASE3_CONFIG['kelly_enabled'] and final_signal != 0:
    # Get win probability from Meta-Labeler or Monte Carlo
    win_prob = meta_probability
    if monte_carlo_result:
        # Use Monte Carlo win prob if available (more accurate)
        win_prob = monte_carlo_result.get('win_probability', meta_probability)
    
    # Get Risk/Reward ratio
    rr_ratio = trend_params['tp1_atr'] / trend_params['sl_atr'] if trend_params['sl_atr'] > 0 else 1.5
    
    kelly_result = kelly_criterion.calculate(win_prob, rr_ratio)
    kelly_adjustment = kelly_result.get('lot_multiplier', 1.0)
    
    # If Kelly says NO_TRADE, reduce lot significantly but don't filter
    if kelly_result.get('recommendation') == 'NO_TRADE':
        kelly_adjustment = 0.25  # Still trade but minimal
        logger.warning(f"[KELLY] {symbol} | Negative edge! Reducing to 25% lot")
    
    lot_multiplier *= kelly_adjustment
    
    logger.info(f"[KELLY] {symbol} | WinProb={win_prob:.1%} RR={rr_ratio:.2f} "
               f"Kelly={kelly_result.get('kelly_fraction', 0):.3f} -> lot_adj={kelly_adjustment:.2f}")

# ===================================================================
# STEP 11: PPO RISK VALIDATION
# ===================================================================
karma_data = karma_engine.get_karma(symbol)
karma_points = karma_data.get('points', 0)
effective_karma = karma_from_ea if karma_from_ea != 0 else karma_points

risk_result = ppo_engine.validate_signal({
    'signal': final_signal,
    'confidence': final_confidence,
    'symbol': symbol,
    'karma': effective_karma,
    'trades_today': trades_today,
    'consecutive_losses': consecutive_losses,
    'adx_m15': adx_m15
}) if ppo_engine else {'approved': True, 'action': 'APPROVE', 'lot_multiplier': 1.0, 'reason': 'PPO disabled'}

if not risk_result['approved']:
    final_signal = 0
    lot_multiplier = 0
    approved = False
else:
    lot_multiplier *= risk_result['lot_multiplier']

# Apply trend adaptive lot multiplier
lot_multiplier *= trend_params['lot_mult']
lot_multiplier = min(lot_multiplier, 2.0)  # Cap at 2x

signal_name = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}[final_signal]

# Cache for heartbeat
last_signals[symbol] = {
    "signal": signal_name,
    "confidence": round(final_confidence, 1),
    "meta_probability": round(meta_probability, 3)
}

# Log signal
data_logger.log_signal({
    'symbol': symbol,
    'final_signal': final_signal,
    'confidence': final_confidence,
    'v6_signal': signal_v6,
    'ensemble_signal': ensemble_result['ensemble_signal'],
    'meta_probability': meta_probability,
    'mamba_signal': ensemble_result['mamba']['signal'],
    'lstm_signal': ensemble_result['lstm']['signal'],
    'transformer_signal': ensemble_result['transformer']['signal'],
    'has_consensus': ensemble_result['has_consensus'],
    'signal_strength': signal_strength,
    'approved': approved,
    'rsi_m15': rsi_m15, 'rsi_h1': rsi_h1, 'rsi_h4': rsi_h4,
    'adx_m15': adx_m15, 'main_trend': main_trend
})

logger.info(f"[FINAL] {symbol} | {signal_name} | Conf={final_confidence:.1f}% | "
           f"Lot={lot_multiplier:.2f}x | Meta={meta_probability:.2%} | {signal_strength}")

# ===================================================================
# BUILD MAIN RESPONSE (V13)
# ===================================================================

main_response = {
    # Mamba signal (backward compatible)
    "mamba_signal": ensemble_result['mamba']['signal'],
    "mamba_confidence": round(ensemble_result['mamba']['confidence'], 1),
    
    # PPO validation
    "ppo_action": risk_result['action'],
    "ppo_reason": risk_result['reason'],
    
    # Final decision
    "signal": final_signal,
    "signal_name": signal_name,
    "confidence": round(final_confidence, 1),
    "lot_multiplier": round(lot_multiplier, 2),
    "approved": approved,
    
    # Meta-Labeler
    "meta_probability": round(meta_probability, 3),
    "signal_strength": signal_strength,
    
    # Ensemble details
    "ensemble_signal": round(ensemble_result['ensemble_signal'], 3),
    "has_consensus": ensemble_result['has_consensus'],
    "lstm_signal": round(ensemble_result['lstm']['signal'], 3),
    "transformer_signal": round(ensemble_result['transformer']['signal'], 3),
    
    # V6
    "v6_signal": signal_v6,
    "reason": reason_v6,
    
    # Trend Adaptive
    "trend_strength": trend_strength,
    "trend_lot_mult": trend_params['lot_mult'],
    "trend_max_trades": trend_params['max_trades'],
    "trend_sl_atr": trend_params['sl_atr'],
    "trend_tp1_atr": trend_params['tp1_atr'],
    
    # PHASE 2: Kalman Filter
    "kalman_enabled": PHASE2_CONFIG['kalman_enabled'],
    "rsi_m15_kalman": kalman_data.get('rsi_m15_kalman', rsi_m15),
    "rsi_m15_velocity": kalman_data.get('rsi_m15_velocity', 0),
    "adx_m15_kalman": kalman_data.get('adx_m15_kalman', adx_m15),
    
    # PHASE 2: Monte Carlo (LOT ADJUSTMENT ONLY - no filtering)
    "monte_carlo_enabled": PHASE2_CONFIG['monte_carlo_enabled'],
    "mc_win_probability": monte_carlo_result.get('win_probability', 0) if monte_carlo_result else 0,
    "mc_risk_reward": monte_carlo_result.get('risk_reward', 0) if monte_carlo_result else 0,
    "mc_recommendation": monte_carlo_result.get('recommendation', 'N/A') if monte_carlo_result else 'N/A',
    "mc_risk_score": monte_carlo_result.get('risk_score', 0) if monte_carlo_result else 0,
    "mc_lot_adjustment": round(mc_adjustment, 2),
    "mc_mode": "LOT_ADJUSTMENT_ONLY",
    
    # PHASE 2: Polynomial Features
    "polynomial_enabled": PHASE2_CONFIG['polynomial_enabled'],
    "poly_trend_alignment": poly_data.get('trend_alignment', 0.5) if poly_data else 0.5,
    "poly_rsi_momentum": poly_data.get('rsi_momentum', 0) if poly_data else 0,
    "poly_adx_category": poly_data.get('adx_category', 0.5) if poly_data else 0.5,
    
    # PHASE 3: Sentiment (NHAN)
    "sentiment_enabled": PHASE3_CONFIG['sentiment_enabled'],
    "sentiment_score": sentiment_data.get('score', 0) if sentiment_data else 0,
    "sentiment_status": sentiment_data.get('status', 'N/A') if sentiment_data else 'N/A',
    "sentiment_adjustment": round(sentiment_adjustment, 2),
    
    # PHASE 3: Kelly Criterion
    "kelly_enabled": PHASE3_CONFIG['kelly_enabled'],
    "kelly_fraction": kelly_result.get('kelly_fraction', 0) if kelly_result else 0,
    "kelly_recommendation": kelly_result.get('recommendation', 'N/A') if kelly_result else 'N/A',
    "kelly_adjustment": round(kelly_adjustment, 2),
    
    # Context
    "symbol": symbol,
    "karma": karma_points,
    "entry_tf": "M15",
    "philosophy": "TU HOP NHAT + Phase2 (Kalman/MC/Poly) + Phase3 (Sentiment/Kelly) + SHADOWS",
    "version": VERSION,
    "timestamp": datetime.now().isoformat()
}

# ===================================================================
# NEW: SHADOW PORTFOLIOS PROCESSING
# ===================================================================

if shadow_manager:
    try:
        # Generate signal_id for traffic routing
        import hashlib
        signal_id = hashlib.md5(f"{symbol}{datetime.now().isoformat()}".encode()).hexdigest()
        
        # Extract OHLC data
        close = float(data.get('close', data.get('current_price', 0)))
        open_price = float(data.get('open', close))
        high = float(data.get('high', close))
        low = float(data.get('low', close))
        
        # Prepare signal data for shadows
        signal_data = {
            'symbol': symbol,
            'close': close,
            'open': open_price,
            'high': high,
            'low': low,
            'rsi_m15': rsi_m15,
            'rsi_h1': rsi_h1,
            'rsi_h4': rsi_h4,
            'adx_h4': adx_h4,
            'atr': atr,
            'karma': karma_points,
            'trend_strength': trend_strength,
            'market_regime': 'NEUTRAL'  # Can be enhanced later
        }
        
        # Ensemble result format for shadows: [SELL_prob, HOLD_prob, BUY_prob]
        # V13 ensemble_result has: predictions[0]=SELL, predictions[1]=HOLD, predictions[2]=BUY
        ensemble_probs = [
            ensemble_result['predictions'][0],  # SELL
            ensemble_result['predictions'][1],  # HOLD
            ensemble_result['predictions'][2]   # BUY
        ]
        
        # Process each portfolio
        portfolio_results = {}
        
        for portfolio_name, portfolio_info in shadow_manager.portfolios.items():
            
            # Check traffic %
            if not shadow_manager.should_process_signal(portfolio_name, signal_id):
                continue
            
            # Get strategy decision
            decision = shadow_manager.get_strategy_decision(
                portfolio_name=portfolio_name,
                signal_data=signal_data,
                ensemble_result=ensemble_probs,
                meta_prob=meta_probability
            )
            
            logger.info(
                f"[SHADOW] {portfolio_name}: {decision['signal']} "
                f"(conf={decision['confidence']:.2f}) - {decision['reason']}"
            )
            
            # If LIVE portfolio
            if portfolio_info['is_live']:
                portfolio_results[portfolio_name] = {
                    'type': 'LIVE',
                    'signal': decision['signal'],
                    'confidence': decision['confidence'],
                    'matches_main': decision['signal'] == signal_name
                }
            
            # If SHADOW portfolio → Log to CSV
            else:
                shadow_manager.log_shadow_trade(
                    portfolio_name=portfolio_name,
                    signal_data=signal_data,
                    decision=decision
                )
                
                portfolio_results[portfolio_name] = {
                    'type': 'SHADOW',
                    'signal': decision['signal'],
                    'confidence': decision['confidence'],
                    'simulated': True
                }
        
        # Add shadow info to response
        main_response['shadow_portfolios'] = {
            'enabled': True,
            'processed': list(portfolio_results.keys()),
            'results': portfolio_results
        }
        
        logger.info(f"[SHADOW] Processed {len(portfolio_results)} portfolios")
        
    except Exception as e:
        logger.error(f"[SHADOW] Processing error: {e}", exc_info=True)
        main_response['shadow_portfolios'] = {
            'enabled': True,
            'error': str(e)
        }
else:
    main_response['shadow_portfolios'] = {
        'enabled': False,
        'reason': 'Shadow manager not initialized'
    }

return main_response
```

@app.post(”/api/trade”)
@app.post(”/trade”)
async def log_trade(request: Request):
“”“Log trade result and update karma”””
data = await request.json()
if not data:
raise HTTPException(status_code=400, detail=“No data”)

```
symbol = normalize_symbol(data.get('symbol', 'EURUSD'))

# Normalize pips
original_pips = data.get('profit_pips', 0)
normalized_pips = normalize_pips(
    symbol=symbol,
    ea_pips=original_pips,
    profit_money=data.get('profit_money', 0),
    lot=data.get('lot', 0.01),
    open_price=data.get('open_price', 0),
    close_price=data.get('close_price', 0),
    trade_type=data.get('type', 'UNKNOWN')
)

if abs(normalized_pips - original_pips) > 1:
    logger.info(f"[PIPS] Normalized {symbol}: {original_pips:.1f} -> {normalized_pips:.1f}")

data['profit_pips'] = normalized_pips
data['symbol'] = symbol

# Update karma
karma_result = karma_engine.process_trade(data)

# Log trade
data['karma_before'] = karma_result['karma_before']
data['karma_after'] = karma_result['karma_after']
data_logger.log_trade(data)

# Tokenomics integration (optional)
token_result = None
try:
    import httpx
    async with httpx.AsyncClient() as client:
        token_data = {
            'address': data.get('address', f"trader_{data.get('magic', 0)}"),
            'profit_pips': normalized_pips,
            'is_clean': data.get('is_clean', True),
        }
        resp = await client.post("http://localhost:8888/trade_karma", json=token_data, timeout=5)
        if resp.status_code == 200:
            token_result = resp.json()
except:
    pass

logger.info(f"[TRADE] {symbol} | {normalized_pips:.1f} pips | Karma: {karma_result['karma_after']:.0f}")

return {
    "logged": True,
    "symbol": symbol,
    "profit_pips": normalized_pips,
    "karma": karma_result,
    "karma_after": karma_result['karma_after'],
    "level": karma_result['level'],
    "token": token_result,
    "trade_count": data_logger.get_trade_count(),
    "pips_normalized": normalized_pips != original_pips,
    "timestamp": datetime.now().isoformat()
}
```

@app.get(”/api/karma/{symbol}”)
@app.get(”/karma/{symbol}”)
async def get_karma(symbol: str):
s = normalize_symbol(symbol)
return {“symbol”: s, “karma”: karma_engine.get_karma(s)}

@app.get(”/api/karma”)
@app.get(”/karma”)
async def get_all_karma():
return {“karma”: karma_engine.get_all_karma()}

@app.get(”/api/retrain/status”)
@app.get(”/retrain/status”)
async def retrain_status():
return retrain_engine.get_status()

@app.post(”/api/retrain/trigger”)
@app.post(”/retrain/trigger”)
async def trigger_retrain():
success = retrain_engine.retrain()
return {
“triggered”: True,
“success”: success,
“status”: retrain_engine.get_status()
}

@app.post(”/api/risk”)
async def check_risk(request: Request):
“”“Standalone risk check endpoint”””
data = await request.json() if request.method == “POST” else {}
result = ppo_engine.validate_signal(data) if ppo_engine else {‘approved’: True}
return {**result, “timestamp”: datetime.now().isoformat()}

@app.post(”/api/pipeline”)
async def full_pipeline(request: Request):
“”“Full pipeline endpoint - one call does everything”””
data = await request.json()
symbol = normalize_symbol(data.get(‘symbol’, ‘EURUSD’))

```
# Get signal
signal_result = await get_signal(request)

# Get karma
karma = karma_engine.get_karma(symbol)

return {
    "pipeline": "V6 -> Ensemble -> Meta-Labeler -> PPO -> Karma",
    "ensemble": {
        "mamba": signal_result.get('mamba_signal', 0.5),
        "lstm": signal_result.get('lstm_signal', 0.5),
        "transformer": signal_result.get('transformer_signal', 0.5),
        "consensus": signal_result.get('has_consensus', False)
    },
    "meta_labeler": {
        "probability": signal_result.get('meta_probability', 0.5),
        "strength": signal_result.get('signal_strength', 'NONE')
    },
    "ppo": {
        "action": signal_result.get('ppo_action', 'APPROVE'),
        "reason": signal_result.get('ppo_reason', ''),
        "approved": signal_result.get('approved', True)
    },
    "final": {
        "signal": signal_result.get('signal', 0),
        "signal_name": signal_result.get('signal_name', 'HOLD'),
        "lot_multiplier": signal_result.get('lot_multiplier', 1.0)
    },
    "karma": karma,
    "timestamp": datetime.now().isoformat()
}
```

@app.get(”/api/phase2”)
@app.get(”/phase2”)
async def phase2_status():
“”“Get Phase 2 advanced features status”””
return {
“phase”: 2,
“name”: “Advanced Features”,
“status”: “ACTIVE”,
“features”: {
“kalman_filter”: {
“enabled”: PHASE2_CONFIG[‘kalman_enabled’],
“description”: “Denoise RSI/ADX signals using Kalman filtering”,
“config”: {
“process_variance”: PHASE2_CONFIG[‘kalman_process_variance’],
“measurement_variance”: PHASE2_CONFIG[‘kalman_measurement_variance’]
},
“benefit”: “Reduces false signals from market noise”
},
“monte_carlo”: {
“enabled”: PHASE2_CONFIG[‘monte_carlo_enabled’],
“description”: “Risk simulation for DYNAMIC LOT SIZING (no filtering)”,
“config”: {
“simulations”: PHASE2_CONFIG[‘monte_carlo_simulations’],
“mode”: “LOT_ADJUSTMENT_ONLY”
},
“lot_adjustments”: {
“STRONG_ENTRY”: “lot x 1.25 (high win prob)”,
“NORMAL_ENTRY”: “lot x 1.0 (default)”,
“WEAK_ENTRY”: “lot x 0.75 (lower confidence)”,
“AVOID”: “lot x 0.5 (still trades, smaller size)”
},
“benefit”: “Dynamic lot sizing based on risk simulation - NO SIGNAL FILTERING”
},
“polynomial_features”: {
“enabled”: PHASE2_CONFIG[‘polynomial_enabled’],
“description”: “Feature engineering: RSI^2, RSI*ADX interactions”,
“config”: {
“degree”: PHASE2_CONFIG[‘polynomial_degree’],
“interactions”: PHASE2_CONFIG[‘polynomial_interactions’]
},
“benefit”: “Captures non-linear relationships in data”
}
},
“pipeline”: “Signal -> Kalman -> V6 -> Poly -> Ensemble -> Meta(FILTER) -> MC(LOT) -> PPO -> Trade”,
“filter_points”: {
“meta_labeler”: “ONLY filter point - prob < 50% = FILTERED”,
“monte_carlo”: “NO filter - only adjusts lot size”
},
“timestamp”: datetime.now().isoformat()
}

@app.post(”/api/phase2/config”)
async def update_phase2_config(request: Request):
“”“Update Phase 2 configuration”””
data = await request.json()

```
# Update Kalman
if 'kalman_enabled' in data:
    PHASE2_CONFIG['kalman_enabled'] = bool(data['kalman_enabled'])

# Update Monte Carlo (lot adjustment only, no filter)
if 'monte_carlo_enabled' in data:
    PHASE2_CONFIG['monte_carlo_enabled'] = bool(data['monte_carlo_enabled'])
if 'monte_carlo_simulations' in data:
    PHASE2_CONFIG['monte_carlo_simulations'] = int(data['monte_carlo_simulations'])
    monte_carlo.n_simulations = PHASE2_CONFIG['monte_carlo_simulations']

# Update Polynomial
if 'polynomial_enabled' in data:
    PHASE2_CONFIG['polynomial_enabled'] = bool(data['polynomial_enabled'])

logger.info(f"[PHASE2] Config updated: {PHASE2_CONFIG}")

return {
    "updated": True,
    "config": PHASE2_CONFIG,
    "timestamp": datetime.now().isoformat()
}
```

@app.get(”/dashboard”)
@app.get(”/api/dashboard”)
async def dashboard():
return {
“name”: “[BODHI] Bodhi Genesis V13 - Phase 2 Advanced”,
“version”: VERSION,
“philosophy”: “TU HOP NHAT + Kalman + Monte Carlo + Polynomial”,
“pipeline”: “Kalman -> V6 -> Poly -> Ensemble -> Meta -> MC -> PPO -> Trade -> Karma”,
“entry_tf”: “M15”,
“v6_rules”: {
“buy”: f”RSI < {V6_CONFIG[‘rsi_buy_max’]}”,
“sell”: f”RSI > {V6_CONFIG[‘rsi_sell_min’]}”,
“trend”: f”ADX > {V6_CONFIG[‘adx_min’]}”,
“sessions”: {sym: f”{s[‘start’]}:00-{s[‘end’]}:00” for sym, s in SYMBOL_SESSIONS.items()},
“cooldown”: f”{V6_CONFIG[‘max_consecutive_losses’]} losses = {V6_CONFIG[‘cooldown_hours’]}h cooldown”
},
“phase1_ensemble”: {
“mamba”: {“loaded”: ensemble_engine.mamba_loaded if ensemble_engine else False, “accuracy”: “73.3%”, “weight”: “35%”},
“lstm”: {“loaded”: ensemble_engine.lstm_loaded if ensemble_engine else False, “accuracy”: “74.2%”, “weight”: “40%”},
“transformer”: {“loaded”: ensemble_engine.transformer_loaded if ensemble_engine else False, “accuracy”: “66.9%”, “weight”: “25%”},
“meta_labeler”: {“loaded”: ensemble_engine.meta_loaded if ensemble_engine else False},
},
“phase2_advanced”: {
“kalman_filter”: PHASE2_CONFIG[‘kalman_enabled’],
“monte_carlo”: PHASE2_CONFIG[‘monte_carlo_enabled’],
“polynomial_features”: PHASE2_CONFIG[‘polynomial_enabled’],
},
“meta_thresholds”: {
“strong”: f”{ENSEMBLE_CONFIG[‘meta_strong_threshold’]:.0%} -> lot 1.0x”,
“moderate”: f”{ENSEMBLE_CONFIG[‘meta_moderate_threshold’]:.0%} -> lot 0.75x”,
“minimum”: f”{ENSEMBLE_CONFIG[‘meta_minimum_threshold’]:.0%} -> lot 0.5x”
},
“ppo_loaded”: ppo_engine.model_loaded if ppo_engine else False,
“trade_records”: data_logger.get_trade_count(),
“karma”: karma_engine.get_all_karma(),
“retrain”: retrain_engine.get_status(),
“timestamp”: datetime.now().isoformat()
}

# ======================================================================

# SHADOW PORTFOLIOS ENDPOINTS (NEW)

# ======================================================================

@app.get(”/api/shadow/status”)
async def get_shadow_status():
“”“Get shadow portfolios status”””

```
if not shadow_manager:
    return {"success": False, "error": "Shadow manager not initialized"}

portfolios = shadow_manager.portfolios

live = {name: info for name, info in portfolios.items() if info['is_live']}
shadows = {name: info for name, info in portfolios.items() if not info['is_live']}

# Get sentiment
sentiment = shadow_manager._get_market_sentiment()

return {
    'success': True,
    'live_portfolio': live,
    'shadow_portfolios': shadows,
    'total_shadows': len(shadows),
    'market_sentiment': sentiment,
    'timestamp': datetime.now().isoformat()
}
```

@app.get(”/api/shadow/performance”)
async def get_shadow_performance(days: int = 7):
“”“Compare performance of all portfolios”””

```
if not shadow_manager:
    return {"success": False, "error": "Shadow manager not initialized"}

try:
    comparison = shadow_manager.compare_portfolios(days=days)
    recommendation = shadow_manager.get_promotion_recommendation(comparison)
    
    return {
        'success': True,
        'period_days': days,
        'portfolios': comparison,
        'recommendation': recommendation,
        'timestamp': datetime.now().isoformat()
    }

except Exception as e:
    logger.error(f"[SHADOW] Performance error: {e}")
    return {"success": False, "error": str(e)}
```

@app.post(”/api/shadow/promote/{shadow_name}”)
async def promote_shadow(shadow_name: str):
“”“Promote a shadow portfolio to live”””

```
if not shadow_manager:
    return {"success": False, "error": "Shadow manager not initialized"}

try:
    result = shadow_manager.promote_shadow_to_live(shadow_name)
    
    logger.warning(
        f"[PROMOTION] '{result['old_live']}' → '{result['new_live']}' "
        f"(v{result['new_version']})"
    )
    
    return result

except ValueError as e:
    return {"success": False, "error": str(e)}
except Exception as e:
    logger.error(f"[SHADOW] Promotion error: {e}")
    return {"success": False, "error": str(e)}
```

@app.patch(”/api/shadow/traffic/{portfolio_name}”)
async def update_shadow_traffic(portfolio_name: str, traffic_pct: float):
“”“Update traffic percentage for a shadow”””

```
if not shadow_manager:
    return {"success": False, "error": "Shadow manager not initialized"}

if not (0 <= traffic_pct <= 100):
    return {"success": False, "error": "traffic_pct must be 0-100"}

try:
    result = shadow_manager.update_traffic(portfolio_name, traffic_pct)
    
    logger.info(
        f"[SHADOW] Traffic updated: {portfolio_name} "
        f"{result['old_traffic']}% → {result['new_traffic']}%"
    )
    
    return result

except ValueError as e:
    return {"success": False, "error": str(e)}
except Exception as e:
    logger.error(f"[SHADOW] Traffic error: {e}")
    return {"success": False, "error": str(e)}
```

@app.get(”/api/sentiment”)
async def get_market_sentiment():
“”“Get current market sentiment from VADER worker”””

```
if not shadow_manager:
    return {"success": False, "error": "Shadow manager not initialized"}

sentiment = shadow_manager._get_market_sentiment()

return {
    'success': True,
    'sentiment': sentiment,
    'timestamp': datetime.now().isoformat()
}
```

# ======================================================================

# SCHEDULER - Weekly retrain

# ======================================================================

def scheduled_retrain():
“”“Scheduled retrain job”””
logger.info(”[SCHED] Scheduled retrain check…”)
retrain_engine.retrain()

def run_scheduler():
“”“Run scheduler in background”””
schedule.every().sunday.at(“06:00”).do(scheduled_retrain)
while True:
schedule.run_pending()
time_module.sleep(60)

# ======================================================================

# MAIN

# ======================================================================

def main():
global ensemble_engine, ppo_engine

```
import argparse
parser = argparse.ArgumentParser(description='Bodhi Genesis V13 - Phase 2 Advanced')
parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='Server port')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host')
args = parser.parse_args()

print("=" * 70)
print("[BODHI] BODHI GENESIS SERVER V13 - PHASE 2 ADVANCED")
print("   Ensemble + Kalman Filter + Monte Carlo + Polynomial Features")
print("=" * 70)
print("""
```

PHASE 1 - ENSEMBLE MODELS:
+—————–+–––––+––––+
| Model           | Accuracy | Weight |
+—————–+–––––+––––+
| Mamba V10       | 73.3%    | 35%    |
| BiLSTM          | 74.2%    | 40%    |
| Transformer     | 66.9%    | 25%    |
+—————–+–––––+––––+

PHASE 2 - ADVANCED FEATURES:
+———————+–––––––––––––––––+
| Feature             | Description                      |
+———————+–––––––––––––––––+
| Kalman Filter       | Denoise RSI/ADX, reduce noise    |
| Monte Carlo         | Risk simulation, win probability |
| Polynomial Features | RSI^2, RSI*ADX interactions      |
+———————+–––––––––––––––––+

FULL PIPELINE V13:
Signal -> Kalman -> V6 -> Poly Features -> Ensemble
-> Meta-Labeler -> Monte Carlo -> PPO -> Trade -> Karma

BACKWARD COMPATIBLE with EA V4.13+
“””)

```
# Initialize engines
ensemble_engine = EnsembleEngine(MODELS_DIR)
ppo_engine = PPORiskEngine(MODELS_DIR)

print(f"\n[*] Entry TF: M15")
print(f"[*] Server: http://localhost:{args.port}")
print(f"[*] Version: {VERSION}")

print(f"\n[PHASE 1] Ensemble Models:")
print(f"    Mamba:       {'[OK] Loaded (73.3%)' if ensemble_engine.mamba_loaded else '[X] Not found'}")
print(f"    LSTM:        {'[OK] Loaded (74.2%)' if ensemble_engine.lstm_loaded else '[X] Not found'}")
print(f"    Transformer: {'[OK] Loaded (66.9%)' if ensemble_engine.transformer_loaded else '[X] Not found'}")
print(f"    Meta-Labeler: {'[OK] Loaded' if ensemble_engine.meta_loaded else '[X] Not found (pip install lightgbm)'}")
print(f"    PPO Risk:    {'[OK] Loaded' if ppo_engine.model_loaded else '[!] Rule-based'}")

print(f"\n[PHASE 2] Advanced Features:")
print(f"    Kalman Filter:  {'[OK] Enabled' if PHASE2_CONFIG['kalman_enabled'] else '[X] Disabled'}")
print(f"    Monte Carlo:    {'[OK] Enabled (' + str(PHASE2_CONFIG['monte_carlo_simulations']) + ' sims)' if PHASE2_CONFIG['monte_carlo_enabled'] else '[X] Disabled'}")
print(f"    Polynomial:     {'[OK] Enabled (degree=' + str(PHASE2_CONFIG['polynomial_degree']) + ')' if PHASE2_CONFIG['polynomial_enabled'] else '[X] Disabled'}")

print(f"\n[*] Trade records: {data_logger.get_trade_count()}")
print(f"[*] Retrain: Sunday 6AM (min {RETRAIN_MIN_RECORDS} records)")

print(f"\n[*] Endpoints:")
print(f"   GET  /health          - Health check")
print(f"   POST /api/signal      - Get AI signal (Full Pipeline V13)")
print(f"   POST /api/trade       - Log trade result")
print(f"   GET  /api/karma       - Get karma")
print(f"   GET  /api/phase2      - Phase 2 status")
print(f"   POST /api/pipeline    - Full pipeline response")
print(f"   GET  /dashboard       - Full dashboard")

# Start scheduler thread
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()
print(f"\n[*] Scheduler started (Sunday 6AM retrain)")

print(f"\n[*] Starting FastAPI + Uvicorn...")
print("=" * 70)

uvicorn.run(app, host=args.host, port=args.port)
```

if **name** == ‘**main**’:
main()
