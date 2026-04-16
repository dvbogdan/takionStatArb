import json
import os
from copy import deepcopy

from numba import jit
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from typing import Dict, Any, List, Optional, Set

from basesettings import N_ACTIVE_SYMBOLS, GROUP_Q_LIST, MIN_GROUP_SIZE, MAX_GROUP_SIZE, MAX_N_APPR_GROUPS, \
    N_YEARS_DAILY_S, EQ_WHITE, N_MAX_ETF, GROUP_LIMIT, SYMBOL_BLACKLIST, ALL_GROUPS_LIMIT_T
from common_math import get_n_items_q, get_daily_sharpe_annualized
from localsettings import MD_DAILY, MD_ROOT, GSA_ROOT
from mdaccess import read_symbol_info
from reporting import COLOR_GREEN, COLOR_0, COLOR_RED, COLOR_DARKGREEN, COLOR_DARKBLUE, COLOR_BLUE, COLOR_DARKRED, \
    COLOR_DARKGRAY
from s_sadaily import SADailyStrategy


@jit(nopython=True)
def calc_daily_oc_gaps(o, c) -> ArrayLike:
    n = len(o)
    gaps = np.zeros(n)
    for j in range(1, n):
        gaps[j] = o[j] - c[j - 1]
    return gaps

def get_main_sector_of_a_group(g_symbols: List[str], symbolData: Dict[str, Any],
                               *, k : Optional[str] = None) -> str:
    if k is None:
        k = 'sector'
    sectorW = {}
    for sym in g_symbols:
        s = symbolData[sym]
        if 'sector' in s:
            sector = s[k]
            if sector not in sectorW:
                sectorW[sector] = 0
            sectorW[sector] += 1
    sectorWList = [(sector, sectorW[sector]) for sector in sectorW.keys()]
    sectorWList.sort(key=lambda r: -r[1])
    if len(sectorW) == 0:
        return 'Generic'
    sector0 = sectorWList[0][0]
    if sector0 == 'Generic':
        if len(sectorWList) <= 1:
            return 'Generic'
        if len(sectorWList) == 2:
            # print(sectorWList[1][0])
            return sectorWList[1][0]
        else:
            # print(sectorWList[1][0] if sectorWList[1][1] > sum([sectorWList[j][1] for j in range(2, len(sectorWList))]) else 'Generic')
            return sectorWList[1][0] if sectorWList[1][1] > sum([sectorWList[j][1] for j in range(2, len(sectorWList))]) else 'Generic'
    else:
        # print(sectorWList[0][0])
        return sectorWList[0][0]

def find_symbol_in_list(symbol, symbolListEq) -> Optional[Any]:
    for r in symbolListEq:
        if r[0] == symbol:
            return r
    return None

def calc_etf_fraction(g_symbols: List[str], symbolData: Dict[str, Any]) -> float:
    S0, S1 = 0, 0
    for sym in g_symbols:
        s = symbolData[sym]
        if s['quote_type'] == 'ETF':
            S1 += 1
        S0 += 1
    return S1 / S0 if S0 > 0 else 0

def select_symbols_daily(r_date: str, s_var: str):
    activeSymbolsPreset, activeSymbolsPresetM = None, None
    symbolSelectionPath = os.path.join(MD_ROOT, 'symbol_selection.json')
    if os.path.exists(symbolSelectionPath):
        print('Reading preset active symbols...')
        with open(symbolSelectionPath, 'r') as f:
            symbolSelInfo = json.load(f)
            activeSymbolsPreset = [r[0] for r in symbolSelInfo['active_symbols']]
    else:
        print('Analyzing symbol database...')
    if activeSymbolsPreset is not None:
        # print(activeSymbolsPreset)
        activeSymbolsPresetM = {x for x in activeSymbolsPreset}

    # testing
    # activeSymbolsPreset, activeSymbolsPresetM = None, None

    dailyDataRoot = MD_DAILY
    files = [f for f in os.listdir(dailyDataRoot) if os.path.isfile(os.path.join(dailyDataRoot, f))]
    files.sort()
    symbolData = {}
    symbolListETF: List[Any] = []
    symbolListEq: List[Any] = []
    symbolListOther: List[Any] = []
    nAllSymbols = 0
    for file in files:
        a1 = file.split(".")
        if len(a1) < 2:
            continue
        if a1[1] != 'pq':
            continue
        a2 = a1[0].split("_")
        if a2[1] != r_date:
            continue
        symbol = a2[0]
        if symbol in SYMBOL_BLACKLIST:
            continue
        nAllSymbols += 1
        if activeSymbolsPresetM is not None:
            if symbol not in activeSymbolsPresetM:
                continue

        read_symbol_info(symbol, r_date)

        result_fn = f"{symbol}_{r_date}.pq"
        symbolMeta_fn = f"{symbol}_meta.json"
        path1 = os.path.join(dailyDataRoot, result_fn)
        df1 = pd.read_parquet(path1)
        # print(df1.columns)
        # df1['Gaps'] = calc_daily_oc_gaps(df1['Open'], df1['Close'])
        m: Dict[str, Any] = {}
        metaPath = os.path.join(MD_DAILY, symbolMeta_fn)
        if os.path.exists(metaPath) or symbol in []: # 'spy'
            with open(metaPath, 'r') as f:
                m = json.load(f)
                if 'sector' in m:
                    if m['sector'] is None:
                        m['sector'] = 'Generic'
                else:
                    raise RuntimeError(f"No sector found in {symbolMeta_fn}")
                if 'short_pct' in m:
                    if m['short_pct'] is None:
                        m['short_pct'] = 0.5
                if 'quote_type' not in m or 'short_pct' not in m:
                    print(f'Incomplete infor for symbol: {symbol}')
                    continue
                m['type_sector'] = '{}:{}'.format(m['quote_type'], m['sector'])
                m['target_sector'] = m['type_sector']
        symbolData[symbol] = m
        m['dates'] = list(df1['Date'])
        m['V'] = np.sum(df1['Volume'] * df1['Close'])
        m['gaps'] = df1['Gap']
        m['growth'] = df1['Growth']
        m['O'] = df1['Open']
        m['H'] = df1['High']
        m['L'] = df1['Low']
        m['C'] = df1['Close']
        W = m['V']
        if m['quote_type'] == 'ETF':
            symbolListETF.append((symbol, W))
        elif m['quote_type'] == 'EQUITY':
            symbolListEq.append((symbol, W))
        else:
            symbolListOther.append((symbol, W))
    print(f"Universe: {nAllSymbols} symbols")

    print("ETF", symbolListETF[:10])
    print("Equity", symbolListEq[:10])
    # symbolList = symbolListETF + symbolListEq

    # print(len(symbolList))
    symbolListETF.sort(key=lambda r: -r[1])
    symbolListEq.sort(key=lambda r: -r[1])
    # print([r[0] for r in symbolListEq])
    print('Equity:', len(symbolListEq), 'ETF:', len(symbolListETF), 'Other:', len(symbolListOther),
          'Totally:', len(symbolListEq) + len(symbolListETF) + len(symbolListOther))

    if True:
        symbolSetETF = set([r[0] for r in symbolListETF])
        symbolSetEq = set([r[0] for r in symbolListEq])

    print(f"Symbols selected: {len(symbolListETF)}, {len(symbolListEq)}")

    activeSymbolList = symbolListETF[:N_MAX_ETF] + symbolListEq[:(N_ACTIVE_SYMBOLS - N_MAX_ETF)]
    allSymbolSet = set([r[0] for r in symbolListETF]).union([r[0] for r in symbolListEq])
    activeSymbolSet = set([r[0] for r in activeSymbolList])
    addedFromWL = []
    for symbol in EQ_WHITE:
        if symbol not in activeSymbolSet:
            if symbol not in allSymbolSet:
                print(f'Unknown symbol: {symbol}')
                continue
            addedFromWL.append(symbol)
            r = find_symbol_in_list(symbol, symbolListETF)
            if r is not None:
                activeSymbolList.append(r)
            r = find_symbol_in_list(symbol, symbolListEq)
            if r is not None:
                activeSymbolList.append(r)
    if len(addedFromWL) > 0:
        text1 = ' '.join(addedFromWL)
        print(f'Adding from white list: {text1}')
    activeSymbolList.sort(key=lambda r: -r[1])
    print(f"Active symbols: {len(activeSymbolList)}")
    # exit()

    if activeSymbolsPresetM is None:
        with open(symbolSelectionPath, 'w') as f:
            symbolSelInfo = {'active_symbols': activeSymbolList}
            json.dump(symbolSelInfo, f, indent=4)

    # for symbolI in activeSymbolList:
    #     symbol = symbolI[0]
    #     read_symbol_info(symbol, r_date)

    print("----")
    print(f'{COLOR_GREEN}Symbol study{COLOR_0}')
    A = {}
    groupList : Dict[str, List[Any]] = {}
    allGroupList : List[Any] = []
    nSymbols = len(activeSymbolList)
    for i in range(nSymbols):
        sym1 = activeSymbolList[i][0]
        g1 = symbolData[sym1]['gaps']
        g1n = np.linalg.norm(g1)
        symbolData[sym1]['GapNorm'] = g1n
        symbolData[sym1]['A'] = g1n / np.sqrt(len(g1))
        # print(float(symbolData[sym1]['A']))

    symbolStat = {}

    print(f'{COLOR_DARKGREEN}Symbol          GS      Pnl   Sharpe        W{COLOR_0}')
    print([r[0] for r in activeSymbolList][130:180])
    for i in range(nSymbols): # nSymbols
        sym1 = activeSymbolList[i][0]
        # if sym1 != 'cop':
        #     continue
        # print(i, sym1)
        # symbolStat[sym1] = {'Pnl': 0, 'Activity': 0}
        dates = symbolData[sym1]['dates']
        g1 = symbolData[sym1]['gaps']
        g1n = symbolData[sym1]['GapNorm']
        candidates = []
        approved_groups = []
        for j in range(nSymbols):
            if j == i:
                continue
            sym2 = activeSymbolList[j][0]
            g2 = symbolData[sym2]['gaps']
            g2n = symbolData[sym2]['GapNorm']
            corr = g1.dot(g2) / g1n / g2n
            A[(i, j)] = corr
            candidates.append((sym2, corr))
        candidates.sort(key=lambda r: -r[1])
        prevNc = -1
        for q in GROUP_Q_LIST:
            nc = get_n_items_q([r[1] for r in candidates], [q])[0]
            if nc == prevNc:
                continue
            # print(q, nc)
            if MIN_GROUP_SIZE <= nc <= MAX_GROUP_SIZE:
                approved_groups.append((q, nc))
                prevNc = nc
                if len(approved_groups) >= MAX_N_APPR_GROUPS:
                    break
        for groupInfo in approved_groups:
            nc = groupInfo[1]
            g_symbols = [sym1] + [c[0] for c in candidates[:nc]]
            # print(g_symbols)
            # print("sim: ", nc, g_symbols)

            # Analysis

            sector = get_main_sector_of_a_group(g_symbols, symbolData) # , k='target_sector'

            # Simulation

            strategy = SADailyStrategy(sym1, g_symbols, dates, symbolData)
            dpnl, dvolume, dpnlBySymbol = strategy.simulate()
            Pnl = np.sum(dpnl) / N_YEARS_DAILY_S
            PnlK1Limit = 1.0
            PnlK1 = Pnl / PnlK1Limit if Pnl < PnlK1Limit else 1
            sharpe = get_daily_sharpe_annualized(dpnl)
            W = 0
            if Pnl > 0:
                W = PnlK1 * Pnl * (
                        (4 * np.tanh(sharpe / 4)) # ** 1.5
                )
            # print('Pnl', Pnl, ':', PnlK1,
            #       'Sharpe', sharpe, ':', 4 * np.tanh(sharpe / 4),
            #       '    W', W)
            # print('Pnl', Pnl, '    Volume', np.sum(dvolume), dpnl.shape[0],
            #       '    Sharpe', sharpe, '    W', W)
            extra : Dict[str, Any] = {}
            etfPnl, eqPnl = 0, 0
            for j in range(len(g_symbols)):
                sym2 = g_symbols[j]
                Pnl2 = np.sum(dpnlBySymbol[j]) / N_YEARS_DAILY_S
                if Pnl2 != 0:
                    if sym2 not in symbolStat:
                        symbolStat[sym2] = {}
                        symbolStat[sym2]['Pnl'] = 0.0
                        symbolStat[sym2]['Activity'] = 0
                    symbolStat[sym2]['Pnl'] += Pnl2
                    symbolStat[sym2]['Activity'] += 1
                    if symbolData[sym2]['quote_type'] == 'ETF':
                        etfPnl += Pnl2
                    else:
                        eqPnl += Pnl2
            extra['etfPnl'] = etfPnl
            extra['eqPnl'] = eqPnl
            if sector not in groupList:
                groupList[sector] = []
            groupList[sector].append([sym1, W, g_symbols, dpnl, extra])
            allGroupList.append([sym1, W, g_symbols, dpnl, extra])
            centralSymbolType = symbolData[sym1]['quote_type']
            etfFraction = calc_etf_fraction(g_symbols, symbolData)
            print(f"{sym1:<14s}  {nc:2d}  {Pnl:7.2f}  {sharpe:7.2f}  {W:7.2f}        " +
                  f"{COLOR_DARKRED}{sector:<20s}  {centralSymbolType:<8s}{COLOR_0}    {COLOR_DARKGRAY}ETF: {int(round(100 * etfFraction)):3d} %{COLOR_0}")
        # exit()

    selGroupList : Dict[str, List[Any]] = {}
    for sector in groupList:
        groupList[sector].sort(key=lambda r: -r[1])
        l0, l = groupList[sector], []
        if len(l0) > 0:
            nSkipped = 1
            for i in range(len(l0)):
                ok = True
                for j in range(len(l)):
                    g_symbols1 = l0[i][2]
                    g_symbols2 = l[j][2]
                    # print(g_symbols1, g_symbols2)
                    n1, n2 = len(g_symbols1), len(g_symbols2)
                    n3 = len(set(g_symbols1).intersection(set(g_symbols2)))
                    if n3 >= 0.8 * n1 and n3 >= 0.8 * n2:
                        ok = False
                        break
                if not ok:
                    nSkipped += 1
                else:
                    l.append(l0[i])
            groupList[sector] = l
            if nSkipped > 0:
                print(f'Skipped {nSkipped} of {sector}')
            groupLimit = GROUP_LIMIT[sector] if sector in GROUP_LIMIT else 2
            selGroupList[sector] = groupList[sector][:groupLimit]

    if s_var == 'S': # by sector
        ...
    elif s_var == 'T': # tower style ALL_GROUPS_LIMIT_T
        allGroupList.sort(key=lambda r: -r[1])
        l0, l = allGroupList[:2 * ALL_GROUPS_LIMIT_T], []
        # if len(l0) > 0: # always
        nSkipped = 1
        for i in range(len(l0)):
            ok = True
            for j in range(len(l)):
                g_symbols1 = l0[i][2]
                g_symbols2 = l[j][2]
                # print(g_symbols1, g_symbols2)
                n1, n2 = len(g_symbols1), len(g_symbols2)
                n3 = len(set(g_symbols1).intersection(set(g_symbols2)))
                if n3 >= 0.8 * n1 and n3 >= 0.8 * n2:
                    ok = False
                    break
            if not ok:
                nSkipped += 1
            else:
                l.append(l0[i])
        if nSkipped > 0:
            print(f'Skipped {nSkipped} in the general list')
        allGroupList = l[:ALL_GROUPS_LIMIT_T]
        originalSelGroupList = deepcopy(selGroupList)
        selGroupList = {}
        symbolSetBM = set()
        for r1 in allGroupList:
            g_symbols1 = r1[2]
            sector1 = get_main_sector_of_a_group(g_symbols1, symbolData)
            if sector1 not in selGroupList:
                selGroupList[sector1] = []
            selGroupList[sector1].append(r1)
            if sector1 == 'Basic Materials':
                symbolSetBM.add(r1[0])
        if 'Basic Materials' in originalSelGroupList:
            if 'Basic Materials' not in selGroupList:
                selGroupList['Basic Materials'] = []
            m1 = selGroupList['Basic Materials']
            m2 = originalSelGroupList['Basic Materials']
            print('m1', len(m1), 'm2', len(m2))
            nGroupsInSector1 = len(m1)
            MaxBM = 9
            if nGroupsInSector1 < MaxBM:
                for r2 in m2:
                    if r2[0] not in symbolSetBM:
                        symbolSetBM.add(r2[0])
                        m1.append(r2)
                        nGroupsInSector1 += 1
                        if nGroupsInSector1 >= MaxBM:
                            break
                m1.sort(key=lambda r: -r[1])
    else:
        raise RuntimeError(f'Unknown variable {s_var}')

    # for reducedSector in ['Consumer Cyclical', 'Technology']

    sectors = list(sorted(selGroupList.keys()))

    print("----")
    report = ''
    report += f"{COLOR_GREEN}Selected groups{COLOR_0}\n"
    s_path0 = os.path.join(GSA_ROOT, r_date)
    s_path = os.path.join(s_path0, s_var)
    os.makedirs(s_path, exist_ok=True)
    s_meta_path = os.path.join(s_path, 'meta.json')
    s_meta = {
        'groups': [],
    }
    symbolAppearance = {}
    nGroups = 0
    selGroupSymbols : Set[str] = set()
    for sector in sectors:
        report += f'{COLOR_RED}{sector}{COLOR_0}' + "\n"
        nGroups += len(selGroupList[sector])
        gPnl = 0
        for groupInfo in selGroupList[sector]:
            sym1, W, g_symbols, dpnl, extra = groupInfo
            nc = len(g_symbols)
            # print(g_symbols)
            # print(nc)
            Pnl = np.sum(dpnl) / N_YEARS_DAILY_S
            sharpe = get_daily_sharpe_annualized(dpnl)
            gPnl += Pnl
            selGroupSymbols = selGroupSymbols.union(set(g_symbols))
            etfFraction = calc_etf_fraction(g_symbols, symbolData)
            etfFractionText = ''
            etfFractionText = f'{COLOR_DARKGRAY}ETF: {int(round(100 * etfFraction)):3d} %{COLOR_0}' if etfFraction > 0 else ' ' * 10

            etfPnl1, eqPnl1 = extra['etfPnl'], extra['eqPnl']
            pnlDetailedText = f'{COLOR_DARKGRAY}Pnl:{etfPnl1:8.2f} {eqPnl1:8.2f} {COLOR_0}'
            report += f"{sym1:<14s}  {nc:2d}  {Pnl:7.2f} {sharpe:7.2f} {W:7.2f}        {etfFractionText}     {pnlDetailedText}\n"

            elements = []
            for sym2 in g_symbols:
                element = {
                    'symbol': sym2,
                    'A': symbolData[sym2]['A'],
                }
                elements.append(element)
                if sym2 not in symbolAppearance:
                    symbolAppearance[sym2] = 0
                symbolAppearance[sym2] += 1
            groupMeta = {
                'central_symbol': sym1,
                'symbols': g_symbols,
                'nc': len(g_symbols),
                'sector': sector,
                'etf_fraction': etfFraction,
                'pnl_est': Pnl,
                'W': W,
            }
            s_meta['groups'].append(groupMeta)

            g_key = f"{sym1}_{len(g_symbols)}"
            g_path = os.path.join(s_path, g_key)
            groupInfo1 = {
                'central_symbol': sym1,
                'symbols': g_symbols,
                'elements': elements,
                'sector': sector,
                'etf_fraction': etfFraction,
                'pnl_est': Pnl,
                'W': W,
            }
            with open(g_path, 'w') as f:
                f.write(json.dumps(groupInfo1, indent=4))
        if len(selGroupList[sector]) > 0:
            gPnl /= len(selGroupList[sector])
            report += f"                    {COLOR_BLUE}{gPnl:7.2f}{COLOR_0}\n"

    with open(s_meta_path, 'w') as f:
        f.write(json.dumps(s_meta, indent=4))

    report += f'{COLOR_BLUE}--' + "\n"
    report += f'N groups:  {nGroups}' + "\n"
    report += f'N symbols: {len(selGroupSymbols)}' + "\n"
    report += f'{COLOR_0}' + "\n"

    symbolAppearanceL = [(sym, symbolAppearance[sym]) for sym in symbolAppearance]
    symbolAppearanceL.sort(key=lambda r: -r[1])
    report += "----\n"
    report += f"{COLOR_GREEN}Symbol appearance{COLOR_0}\n"
    for r in symbolAppearanceL[:20]:
        sym = r[0]
        sector = symbolData[sym]['sector']
        report += f"{sym:<14s}  {symbolAppearance[sym]:3d}        {COLOR_DARKRED}{sector:<16s}{COLOR_0}\n"
    if len(symbolAppearance) > 20:
        report += "...\n"

    symbolStatL = []
    for sym in symbolStat:
        stat = symbolStat[sym]
        symbolStatL.append((sym, stat['Activity'], stat['Pnl']))
    symbolStatL.sort(key=lambda r: -r[1])

    report += "----\n"
    report += f"{COLOR_GREEN}Symbol activity{COLOR_0}\n"
    for r in symbolStatL[:24]:
        sym = r[0]
        sector = symbolData[sym]['sector']
        report += f"{sym:<14s}  {r[1]:3d}  {r[2]:8.1f}        {COLOR_DARKRED}{sector:<16s}{COLOR_0}\n"
    if len(symbolStatL) > 24:
        report += "...\n"
    print(report)
    with open(os.path.join(s_path, 'strategy_analysis.txt'), 'w') as f:
        f.write(report)
