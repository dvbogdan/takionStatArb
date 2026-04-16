import json
import os
import os.path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime
from numba import jit
from numpy.typing import ArrayLike
from yfinance.domain import industry

from basesettings import T_DAYS_A_YEAR, DATA_LEN
from common_math import calc_daily_oc_gaps, calc_daily_oc_growth
from localsettings import MD_ROOT_STOOQ, MD_DAILY
from ymd import MDProvider


def load_polygon_daily():
    pass


# @jit(nopython=True)
def merge_dates(dates0: List[str], dates1: List[str],
                O: List[float], H: List[float], L: List[float], C: List[float],
                V: List[float]) -> Optional[Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]]:
    n = len(dates0)
    n1 = len(dates1)
    filled = [False] * n
    O2 = np.zeros(n)
    H2 = np.zeros(n)
    L2 = np.zeros(n)
    C2 = np.zeros(n)
    V2 = np.zeros(n)
    # first available
    availIdx = -1
    j = 0
    for i in range(n):
        while j < n1 and dates1[j] < dates0[i]:
            j += 1
        if j == n1:
            break
        if dates1[j] == dates0[i]:
            availIdx = i
            break
    if availIdx == -1:
        return None
    # filling
    last_C = 0
    if availIdx > 0:
        if j == 0:
            raise RuntimeError('invalid cursor value: j')
        last_C = C[j-1]
    i: int = 0
    while i < availIdx:
        O2[i] = last_C
        H2[i] = last_C
        L2[i] = last_C
        C2[i] = last_C
        i += 1
    # copying values
    j = availIdx
    while i < n:
        while j < n1 and dates1[j] < dates0[i]:
            j += 1
        if j < n1 and dates1[j] == dates0[i]:
            V2[i] = V[j]
            O2[i] = O[j]
            H2[i] = H[j]
            L2[i] = L[j]
            C2[i] = C[j]
            last_C = C[j]
            # if j < 40:
            #     print('copy', j, i)
            # j += 1
        else:
            O2[i] = last_C
            H2[i] = last_C
            L2[i] = last_C
            C2[i] = last_C
        i += 1
    # test
    if True:
        # print(C)
        # print(C2)
        for i in range(n):
            if O2[i] <= 0 or H2[i] <= 0 or L2[i] <= 0 or C2[i] <= 0:
                raise RuntimeError(f'invalid value: O, H, L, and C: {O2[i]}, {H2[i]}, {L2[i]}, {C2[i]}', )
        for j in range(n1):
            if dates1[j] < dates0[0]:
                continue
            ok = False
            for i1 in range(n):
                if dates0[i1] == dates1[j]:
                    ok = True
                    i = i1
                    break
            if not ok:
                print('date', dates1[j], 'is not found in dates0')
                return None
            if C[j] != C2[i]:
                print('wrong values', j, i)
                return None
    return O2, H2, L2, C2, V2



def load_symbol_data_daily(symbol, r_date, dates, O, H, L, C, V, *, dates0) -> bool:
    result_fn = f"{symbol}_{r_date}.pq"
    result_path = os.path.join(MD_DAILY, result_fn)
    # print('dates0', dates0[:3], dates0[-3:], len(dates0))
    # print('dates1', dates[:3], dates[-3:], len(dates), df1.shape[0])
    # print(len(O), len(C))
    # print(O[:5], C[:5])
    ok = True
    for j in range(len(dates)):
        if O[j] <= 0 or H[j] <= 0 or L[j] <= 0 or C[j] <= 0:
            ok = False
            break
    if not ok:
        return False
    res = merge_dates(dates0, dates, O, H, L, C, V)
    if res is None:
        print('skip', symbol, 'wrong dates')
        # exit()
        return False
    O2, H2, L2, C2, V2 = res
    gaps = calc_daily_oc_gaps(O2, C2)
    growth = calc_daily_oc_growth(O2, C2)
    # print(gaps)
    data2 = [[dates0[j], O2[j], H2[j], L2[j], C2[j], V2[j], gaps[j], growth[j]] for j in range(len(dates0))]
    df2 = pd.DataFrame(data2, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Gap', 'Growth'])
    # print(df2)
    df2.to_parquet(result_path)
    return True

def load_stooq_dates():
    rpath = 'daily/us/nasdaq etfs'
    path = str(os.path.join(MD_ROOT_STOOQ, rpath))
    df1 = pd.read_csv(os.path.join(path, 'qqq.us.txt'))
    df1 = df1.iloc[-DATA_LEN:]
    dates = list(df1['<DATE>'])
    return dates

def load_stooq_symbol_group_f(path):
    dates0 = load_stooq_dates()
    r_date = dates0[-1]
    # print(path)
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files.sort()
    for fn in files:
        symbol = fn.split('.')[0]
        # print()
        print(symbol)
        os.makedirs(MD_DAILY, exist_ok=True)
        df2 = None
        result_fn = f"{symbol}_{r_date}.pq"
        result_path = os.path.join(MD_DAILY, result_fn)
        if os.path.exists(result_path):
            # df2 = pd.read_parquet(result_path)
            continue
        path1 = os.path.join(path, fn)
        # print('df1 path', path1)
        if os.stat(path1).st_size == 0:
            continue
        df1 = pd.read_csv(path1)
        df1size = df1.shape[0]
        if df1.iloc[0]['<DATE>'] > dates0[0]:
            continue
        # print(df1.columns)
        df1['date1'] = pd.to_datetime(df1['<DATE>'], format='%Y%m%d')
        # df1['ts1'] = df1['date1']
        df1['ts1'] = df1['date1'].apply(lambda x: int((x - datetime(1970, 1, 1)).total_seconds()))
        dataLen = min(int(1.2 * DATA_LEN), df1.shape[0])
        df1 = df1.iloc[-dataLen:]
        # print(df1.iloc[-1])
        dates = list(df1['<DATE>'])
        O = [float(x) for x in list(df1['<OPEN>'])]
        H = [float(x) for x in list(df1['<HIGH>'])]
        L = [float(x) for x in list(df1['<LOW>'])]
        C = [float(x) for x in list(df1['<CLOSE>'])]
        V = [float(x) for x in list(df1['<VOL>'])]
        load_symbol_data_daily(symbol, r_date, dates, O, H, L, C, V, dates0=dates0)

def load_stooq_symbol_group(rpath):
    path = str(os.path.join(MD_ROOT_STOOQ, rpath))
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if len(files) > 0:
        load_stooq_symbol_group_f(path)
    else:
        dirList = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        dirList.sort()
        for dirName in dirList:
            print('-' * 80)
            print(dirName)
            dirPath = os.path.join(path, dirName)
            load_stooq_symbol_group_f(dirPath)

def load_stooq_daily():
    load_stooq_symbol_group('daily/us/nasdaq etfs')
    load_stooq_symbol_group('daily/us/nasdaq stocks')
    load_stooq_symbol_group('daily/us/nyse etfs')
    load_stooq_symbol_group('daily/us/nyse stocks')

def get_symbol_list_for_rdate(r_date: str):
    # result_fn = f"{symbol}_{r_date}.pq"
    # result_path = os.path.join(MD_DAILY, result_fn)

    symbols = []
    files = [f for f in os.listdir(MD_DAILY) if os.path.isfile(os.path.join(MD_DAILY, f))]
    for fn in files:
        a = fn.split('.')
        if len(a) < 2:
            continue
        if a[1] != 'pq':
            continue
        a2 = a[0].split('_')
        if len(a2) < 2:
            continue
        if a2[1] != r_date:
            continue
        symbols.append(a2[0])
    # print(symbols)
    symbols.sort()
    return symbols

def read_symbol_info(symbol: str, r_date: str) -> None:
    symbolInfoPath = os.path.join(MD_DAILY, f'{symbol}_meta.json')
    # if symbol == 'spy':
    #     print(symbolInfoPath)
    if os.path.exists(symbolInfoPath):
        with open(symbolInfoPath, 'r') as f:
            m = json.load(f)
            if 'short_pct' in m and 'quote_type' in m:
                return
    print(symbol)
    sector, industry, shortPct, quoteType = MDProvider.get_symbol_meta(symbol)
    m = {
        'symbol': symbol,
        'quote_type': quoteType,
        'sector': sector,
        'industry': industry,
        'short_pct': shortPct,
    }
    with open(symbolInfoPath, 'w') as f:
        json.dump(m, f)

def read_symbols_info(r_date: str):
    symbols = get_symbol_list_for_rdate(r_date)
    for symbol in symbols:
        read_symbol_info(symbol, r_date)


