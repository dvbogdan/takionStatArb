import threading
from copy import deepcopy

import dateutil
import json
import os
import time
from datetime import datetime
from threading import Thread, Lock

import matplotlib.pyplot as plt
import numpy as np
# from requests.packages import target

from reporting import COLOR_DARKGREEN, COLOR_0, COLOR_DARKBLUE, COLOR_DARKGRAY, COLOR_BLUE, COLOR_GREEN, COLOR_DARKRED, \
    COLOR_DARKYELLOW
from time_util import estdatetime, time_est_s
from ymd import MDProvider
from typing import Dict, Any, Tuple, List, Optional, Set
from numba import jit
from numpy.typing import ArrayLike
from basesettings import N_YEARS_DAILY_S
from common_math import calc_daily_oc_gaps, calc_daily_oc_growth, get_daily_sharpe_annualized
from localsettings import GSA_ROOT
from s_common import Group

SADAILY_UPDATE_INTERVAL = 20
SADAILY_ECHO_INTERVAL = 60
TARGET_PNL_MIN = 0.25
TARGET_PNL_RED = 0.80
SADAILY_FLUCTUATION = 0.65
USE_OPEN_PRICE = False

GroupEchoLock = Lock()


# @jit(nopython=True)
def sim_SADaily(nS, group, h0 = SADAILY_FLUCTUATION):
    n = len(group[0][0])
    dpnl0 = np.zeros(n)
    dpnl1 = np.zeros(n)
    dpnlBySymbol = [np.zeros(n) for i in range(nS)]
    dvolume = np.zeros(n)
    for i in range(1, n):
        rsi_a = [(k, group[k][0][i]) for k in range(nS)]
        rsi_a.sort(key=lambda r: r[1])
        k0, k1 = rsi_a[0][0], rsi_a[-1][0]
        gap0, gap1 = group[k0][5][i], group[k1][5][i] # morning gap
        rsi0, rsi1 = rsi_a[0][1], rsi_a[-1][1]
        # print('RSI', rsi0, rsi1)
        meanRsi = sum([rsi_a[k][1] for k in range(nS)]) / nS
        if rsi0 <= meanRsi - h0 or rsi1 >= meanRsi + h0:
            # growth0, growth1 = group[k0][6][i], group[k1][6][i] # day growth
            O0, O1 = group[k0][1][i], group[k1][1][i]
            C0, C1 = group[k0][4][i], group[k1][4][i]
            H0, L1 = group[k0][2][i], group[k1][3][i]
            A0, A1 = group[k0][7], group[k1][7]
            # print(meanGap, A0)
            # exit()
            x0 = (meanRsi - rsi0) * A0
            x1 = (meanRsi - rsi1) * A1
            if x0 <= H0 - O0:
                # print('x0=', x0, '  ', O0, ' => ', O0 + x0)
                ds0 = x0 / O0
            else:
                # print('x0=', x0, '  ', O0, ' => ', C0, 'C')
                ds0 = (C0 - O0) / O0
            if abs(x1) <= O1 - L1:
                ds1 = abs(x1) / O1
            else:
                ds1 = (C1 - O1) / O1
            # if gap0 > -h0:
            #     ds0 = 0
            # if gap1 < h0:
            #     ds1 = 0
            # print(ds0, ds1, '    ', gap0, O0, C0)
            dpnl0[i] = ds0
            dpnl1[i] = ds1
            dpnlBySymbol[k0] += ds0
            dpnlBySymbol[k1] += ds1
            dvolume[i] = 1
    # print('dpnl0', np.sum(dpnl0))
    # print('dpnl1', np.sum(dpnl1))
    return dpnl0 + dpnl1, dvolume, dpnlBySymbol

class SADailyStrategy:
    def __init__(self, centralSymbol, symbols, dates, symbolData):
        self.centralSymbol = centralSymbol
        self.symbols = symbols
        self.dates = dates
        self.group = Group(symbols, symbolData)

    def simulate(self):
        nS = self.group.nSymbols
        dpnl, dvolume, dpnlByStrategy = sim_SADaily(nS, self.group.data)
        if self.centralSymbol == 'lng' and False:
            fig, axs = plt.subplots(2,
                                    figsize=(24, 21), dpi=90, constrained_layout=True,
                                    gridspec_kw={'height_ratios': [8, 2]})
            axs[0].plot(np.cumsum(dpnl), color='blue', linewidth=2, label='Pnl')
            axs[1].plot(np.cumsum(dvolume), color='darkviolet', linewidth=2, label='NTrades')
            axs[0].legend(loc="upper left")
            axs[1].legend(loc="upper left")
            plt.show()
        if self.centralSymbol == 'lng' and False:
            cmap = plt.get_cmap('viridis')
            colors = [cmap(i) for i in np.linspace(0, 1, nS)]
            fig, axs = plt.subplots(2,
                                    figsize=(24, 21), dpi=90, constrained_layout=True,
                                    gridspec_kw={'height_ratios': [4, 4]})
            for k in range(nS):
                axs[0].plot(self.group.data[k][4], color=colors[k], linewidth=1, label=self.symbols[k])
                axs[1].plot(self.group.data[k][0], color=colors[k], linewidth=1)
            axs[0].legend(loc="upper left")
            axs[1].legend(loc="upper left")
            plt.show()

        # exit()
        return dpnl, dvolume, dpnlByStrategy


class SADailyStrategyRunner(Thread):
    def __init__(self, options: Dict[str, Any]):
        Thread.__init__(self, target=self.runner_loop, args=())
        self.options = options
        self.mdp = options['mdp'] if 'mdp' in options else MDProvider
        r_date = options['r_date']
        self.s_var = options['s_var']
        s_path0 = str(os.path.join(GSA_ROOT, r_date))
        s_path = os.path.join(s_path0, self.s_var)
        if not os.path.exists(s_path):
            raise RuntimeError(f'No strategy data for r.date {r_date}')
        s_meta_path = os.path.join(s_path, 'meta.json')
        if not os.path.exists(s_meta_path):
            raise RuntimeError(f'No strategy meta for r.date {r_date}')
        self.metaInfo = json.load(open(s_meta_path, 'r'))
        s_opt = options['s_opt']
        self.s_opt = s_opt
        self.groups = []
        for groupMeta in self.metaInfo['groups']:
            groupMeta['r_date'] = r_date
            groupMeta['s_path'] = s_path
            groupMeta['s_opt'] = s_opt
            groupMeta['s_var'] = self.s_var
            groupMeta['mdp'] = self.mdp
            print()
            print(groupMeta)
            self.groups.append(SADailyStrategyGroup(groupMeta))

        GroupEchoLock.acquire()
        print('----')
        print(f'{COLOR_GREEN}Strategy loaded. Groups: {len(self.groups)}{COLOR_0}')
        GroupEchoLock.release()

        print('Starting strategy...')
        for group in self.groups:
            group.start_group()
        Thread.start(self)

    def get_set_of_symbols(self) -> Set:
        symbolSet = set()
        for group in self.groups: # [:1]
            symbolSet = symbolSet.union(group.get_set_of_symbols())
        return symbolSet

    def runner_loop(self):
        lastSection = 0
        dt = 5 if 'I' in self.s_opt else 15
        while True:
            t0 = int(time.time())
            t0s = (t0 - dt) % SADAILY_UPDATE_INTERVAL
            sectionTs = t0 - t0s
            if sectionTs > lastSection:
                if lastSection != 0:
                    # print('--')
                    # print('t0:', time_est_s(t0))
                    self.update(t0)
                lastSection = sectionTs
            time.sleep(1)

    def update(self, t):
        self.echo()

    def echo(self):
        sPnl, sAUM, sPos = 0, 0, 0
        Pos = 0
        AUMBySymbol : Dict[str, float] = {}
        PosBySymbol : Dict[str, float] = {}
        PnlBySymbol : Dict[str, float] = {}
        SymbolCounterParts: Dict[str, Dict[str, Any]] = {}
        print('--')
        print(f'{COLOR_GREEN}Strategy: Daily Stat-Arb{COLOR_0}')
        print(f'{COLOR_DARKGREEN}Id   C.Symbol    Position AUM       Pnl{COLOR_0}')
        lastSector = ''
        sectorPnl, sectorPnlS0, sectorPos, sectorAUM = 0, 0, 0, 0
        for i, group in enumerate(self.groups):
            group: SADailyStrategyGroup
            csText = f'{group.centralSymbol}_{group.nS}'
            sector = group.groupInfo['sector']
            if sector != lastSector:
                if sectorPnlS0 != 0:
                    sectorPnl /= sectorPnlS0
                    print(f'                      {COLOR_DARKGRAY}{int(sectorPos):3d} {int(sectorAUM):3d}  {sectorPnl:8.3f}{COLOR_0}')
                    sectorPnl, sectorPnlS0, sectorPos, sectorAUM = 0, 0, 0, 0
                print(f'{COLOR_BLUE}{sector}{COLOR_0}')
                lastSector = sector
            if group.activated or group.finished:
                gAUM = 0
                gPos = 0
                if group.balancingMode == 'A':
                    for k in group.positions:
                        sym = group.symbols[k]
                        if sym not in PosBySymbol:
                            AUMBySymbol[sym] = 0.0
                            PosBySymbol[sym] = 0.0
                            PnlBySymbol[sym] = 0.0
                            SymbolCounterParts[sym] = {}
                        dPos = group.positions[k]
                        AUMBySymbol[sym] += abs(dPos)
                        PosBySymbol[sym] += dPos
                        gAUM += abs(dPos)
                        gPos += dPos
                        if group.state.get(sym) is not None:
                            PnlBySymbol[sym] += group.state[sym]['Pnl']
                else:
                    sym0, sym1 = group.symbols[group.k0], group.symbols[group.k1]
                    if sym0 not in PosBySymbol:
                        AUMBySymbol[sym0] = 0.0
                        PosBySymbol[sym0] = 0.0
                        PnlBySymbol[sym0] = 0.0
                        SymbolCounterParts[sym0] = {}
                    if sym1 not in PosBySymbol:
                        AUMBySymbol[sym1] = 0.0
                        PosBySymbol[sym1] = 0.0
                        PnlBySymbol[sym1] = 0.0
                        SymbolCounterParts[sym1] = {}
                    PosBySymbol[sym0] += 1
                    PosBySymbol[sym1] += -1
                    gAUM += 2
                    if group.state.get(sym0) is not None:
                        PnlBySymbol[sym0] += group.state[sym0]['Pnl']
                    if group.state.get(sym1) is not None:
                        PnlBySymbol[sym1] += group.state[sym1]['Pnl']
                    SymbolCounterParts[sym0][sym1] = 1
                    SymbolCounterParts[sym1][sym0] = 1
                stateText = ''
                if group.finished:
                    stateText += f'{COLOR_DARKRED}C{COLOR_0}'
                stateText += f'    {group.date}'
                sPos += gPos
                sAUM += gAUM
                Pos += gAUM
                sPnl += group.Pnl
                sectorPnl += group.Pnl
                sectorPnlS0 += gAUM
                sectorPos += gPos
                sectorAUM += gAUM
                print(f'{i:<3d}  {csText:<14s}    {int(gPos):2d} {int(gAUM):3d}  {group.Pnl:8.3f}  {stateText}')
            else:
                print(f'{i:<3d}  {csText:<14s}                      {group.date}')
        if sectorAUM != 0:
            sectorPnl /= sectorPnlS0
            print(f'                      {COLOR_DARKGRAY}{int(sectorPos):3d} {int(sectorAUM):3d}  {sectorPnl:8.3f}{COLOR_0}')
        if Pos > 0:
            sPnl /= Pos
            print(f'{COLOR_BLUE}                       {int(sPos):2d} {int(sAUM):3d}  {sPnl:8.4f}{COLOR_0}')

            print(f'{COLOR_GREEN}Position by symbol{COLOR_0}')
            PosBySymbolList = []
            for symbol in PosBySymbol:
                Pos1 = PosBySymbol[symbol]
                PosBySymbolList.append((symbol, Pos1))
            PosBySymbolList.sort(key=lambda x: x[1], reverse=True)
            for symbol, Pos1 in reversed(PosBySymbolList):
                cpList = list(SymbolCounterParts[symbol].keys())
                cpText = " ".join(cpList)
                print(f'{symbol:<10s}    {Pos1:6.1f}        {COLOR_DARKYELLOW}{cpText}{COLOR_0}')

            print(f'{COLOR_GREEN}Symbol Pnl{COLOR_0}')
            PnlBySymbolList = []
            for symbol in PnlBySymbol:
                Pnl1 = PnlBySymbol[symbol]
                if abs(Pnl1) > 0.00005:
                    PnlBySymbolList.append((symbol, Pnl1))
            PnlBySymbolList.sort(key=lambda x: x[1], reverse=True)
            for symbol, Pnl1 in reversed(PnlBySymbolList):
                print(f'{symbol:<10s}    {Pnl1:8.3f}        ')

            print(f'{COLOR_BLUE}                            {sPnl:8.4f}{COLOR_0}')


class SADailyStrategyGroup(Thread):
    def __init__(self, groupMeta: Dict[str, Any]):
        Thread.__init__(self, target=self.run_strategy, args=())
        self.mdp = groupMeta['mdp']
        self.inSession = False
        self.preSession = False
        self.date = ''
        self.state: Optional[Dict[str, Any]] = None
        self.ready = False
        self.activated = False
        self.finished = False
        self.sessionType = 'I' if 'I' in groupMeta['s_opt'] else 'P'
        self.balancingMode = 'A' if 'A' in groupMeta['s_opt'] else 'B'
        self.rsiValues = {}
        self.meanRsi = 0
        self.Pnl = 0
        self.k0 = -1
        self.k1 = -1
        self.positions = {}
        self.x0 = 0
        self.x1 = 0
        self.groupMeta = groupMeta
        self.centralSymbol = groupMeta['central_symbol']
        self.nc = groupMeta['nc']
        self.sector = groupMeta['sector']
        s_path = groupMeta['s_path']
        self.stateDir = s_path + '/' + self.sessionType
        print('-' * 36)
        print(self.stateDir)
        os.makedirs(self.stateDir, exist_ok=True)
        g_key = f"{self.centralSymbol}_{self.nc}"
        g_path = os.path.join(s_path, g_key)
        if not os.path.exists(g_path):
            r_date = self.groupMeta['r_data']
            raise RuntimeError(f'No group data for r.date {r_date}, {self.centralSymbol}')
        self.groupInfo = json.load(open(g_path, 'r'))
        self.symbols = self.groupInfo['symbols']
        self.nS = len(self.symbols)
        self.A = [0] * self.nS
        for k in range(self.nS):
            self.A[k] = self.groupInfo['elements'][k]['A']

        self.load_state()

        # for sym2 in self.groupInfo['symbols']:
        #     print(sym2, self.mdp.get_realtime_price(sym2) )

    def get_set_of_symbols(self) -> Set:
        return set(self.symbols)

    def start_group(self):
        Thread.start(self)

    def get_key(self) -> str:
        g_key = f"{self.centralSymbol}_{self.nc}"
        return g_key

    def get_state_path(self):
        g_key = self.get_key()
        state_fn = f"{g_key}_state.json"
        return os.path.join(self.stateDir, state_fn)

    def load_state(self):
        path = self.get_state_path()
        print(path)
        if os.path.exists(path):
            stateInfo = json.load(open(path, 'r'))
            # print(stateInfo)
            day1 = stateInfo['date']
            estT = estdatetime(int(time.time()))
            day = str(estT).split(' ')[0]
            if day != day1:
                # print('init', day, self.get_key())
                _, ok = self.init_state(day)
                if not self.ready and ok:
                    print(f'{self.get_key()} ready')
                self.ready = ok
                if not self.ready:
                    return
            self.date = stateInfo['date']
            self.activated = stateInfo['active']
            self.finished = stateInfo['completed']
            if self.date != '':
                self.state = {}
            if self.balancingMode == 'A':
                if stateInfo.get('positions') is None:
                    stateInfo['positions'] = {}
                self.positions = {int(k): stateInfo['positions'][k] for k in stateInfo['positions']}
                for k in range(self.nS):
                    sym = self.symbols[k]
                    openPrice = 0
                    self.state[sym] = {
                        'activated': False,
                        'closed': False,
                        'p0': 0,
                        'p1': 0,
                        'price': 0,
                        'pC': stateInfo['elements'][k]['pC'],
                        'pO': openPrice,
                        'Pnl': 0,
                        'x': 0,
                    }
                    if 'price' in stateInfo['elements'][k]:
                        self.state[sym]['price'] = stateInfo['elements'][k]['price']
                    if k in self.positions:
                        sym = self.symbols[k]
                        self.state[sym]['p0'] = stateInfo['elements'][k]['p0']
                        self.state[sym]['p1'] = stateInfo['elements'][k]['p1']
                        self.state[sym]['x'] = stateInfo['elements'][k]['x']
                        if 'price' in stateInfo['elements'][k]:
                            self.state[sym]['price'] = stateInfo['elements'][k]['price']
                        self.state[sym]['pC'] = stateInfo['elements'][k]['pC']
            else:
                self.k0 = stateInfo['k0']
                self.k1 = stateInfo['k1']
                self.x0 = stateInfo['x0']
                self.x1 = stateInfo['x1']
                if self.date != '':
                    for k in range(self.nS):
                        sym = self.symbols[k]
                        openPrice = 0
                        # if stateInfo['elements'][k].get('pO') is not None:
                        #     if stateInfo['elements'][k]['pO'] is None:
                        #         stateInfo['elements'][k]['pO'] = self.mdp.get_open(sym)
                        #     openPrice = stateInfo['elements'][k]['pO']
                        #     if openPrice is None:
                        #         print("!!! (1)")
                        #         exit()
                        # else:
                        #     openPrice = self.mdp.get_open(sym)
                        #     if openPrice is None:
                        #         print("!!! (2)")
                        #         exit()
                        # if openPrice is None:
                        #     print("!!!")
                        #     exit()
                        self.state[sym] = {
                            'activated': False,
                            'closed': False,
                            'p0': 0,
                            'p1': 0,
                            'price': 0,
                            'pC': stateInfo['elements'][k]['pC'],
                            'pO': openPrice,
                            'Pnl': 0,
                            'x': 0,
                        }
                        if 'price' in stateInfo['elements'][k]:
                            self.state[sym]['price'] = stateInfo['elements'][k]['price']
                    if self.k0 != -1:
                        # raise RuntimeError('1')
                        sym0 = self.symbols[self.k0]
                        self.state[sym0]['p0'] = stateInfo['I0']['p0']
                        self.state[sym0]['p1'] = stateInfo['I0']['p1']
                        self.state[sym0]['activated'] = stateInfo['I0']['activated']
                        self.state[sym0]['closed'] = stateInfo['I0']['closed']
                        if USE_OPEN_PRICE:
                            if stateInfo['I0'].get('pO') is not None:
                                if stateInfo['I0']['pO'] is None:
                                    print('!', stateInfo['I0']['pO'])
                                    exit()
                                self.state[sym0]['p0'] = stateInfo['I0']['pO']
                            else:
                                self.state[sym0]['p0'] = self.mdp.get_open(sym0)
                            if self.state[sym0]['p0'] is None:
                                print("!")
                                exit()
                    if self.k1 != -1:
                        # raise RuntimeError('1')
                        sym1 = self.symbols[self.k1]
                        self.state[sym1]['p0'] = stateInfo['I1']['p0']
                        self.state[sym1]['p1'] = stateInfo['I1']['p1']
                        self.state[sym1]['activated'] = stateInfo['I1']['activated']
                        self.state[sym1]['closed'] = stateInfo['I1']['closed']
                        if USE_OPEN_PRICE:
                            if stateInfo['I1'].get('pO') is not None:
                                if stateInfo['I1']['pO'] is None:
                                    print('!', stateInfo['I0']['pO'])
                                    exit()
                                self.state[sym1]['p0'] = stateInfo['I1']['pO']
                            else:
                                self.state[sym1]['p0'] = self.mdp.get_open(sym1)
                            if self.state[sym1]['p0'] is None:
                                print("!")
                                exit()
            self.ready = True

    def save_state(self):
        stateInfo : Dict[str, Any] = {
            "date": self.date,
            "active": self.activated,
            "completed": self.finished,
        }
        if self.balancingMode == 'A':
            stateInfo['positions'] = self.positions
        else:
            stateInfo["k0"] = self.k0
            stateInfo["k1"] = self.k1
            stateInfo["x0"] = self.x0
            stateInfo["x1"] = self.x1
        if self.date != '':
            stateInfo['elements'] = []
            for k in range(self.nS):
                sym = self.symbols[k]
                m = {'pC': self.state[sym]['pC']}
                if 'price' in self.state[sym]:
                    m['price'] = self.state[sym]['price']
                if self.balancingMode == 'A':
                    if k in self.positions:
                        m['p0'] = self.state[sym]['p0']
                        m['p1'] = self.state[sym]['p1']
                        m['x'] = self.state[sym]['x']
                stateInfo['elements'].append(m)
        if self.k0 != -1:
            sym0 = self.symbols[self.k0]
            stateInfo['I0'] = {
                'p0': self.state[sym0]['p0'],
                'p1': self.state[sym0]['p1'],
                'activated': self.state[sym0]['activated'],
                'closed': self.state[sym0]['closed'],
            }
        if self.k1 != -1:
            sym1 = self.symbols[self.k1]
            stateInfo['I1'] = {
                'p0': self.state[sym1]['p0'],
                'p1': self.state[sym1]['p1'],
                'activated': self.state[sym1]['activated'],
                'closed': self.state[sym1]['closed'],
            }
        path = self.get_state_path()
        with open(path, 'w') as f:
            json.dump(stateInfo, f, indent=4)

    # returns: updated, ok
    def init_state(self, day: str) -> Tuple[bool, bool]:
        if day != self.date:
            # print('trying to update...')
            self.state = {}
            for k in range(self.nS):
                sym = self.symbols[k]
                # pO = self.mdp.get_open(sym)
                pC = self.mdp.get_previous_close(sym)
                # print('pC', sym, pC)
                if pC is None:
                    return False, False
                self.state[sym] = {
                    'activated': False,
                    'closed': False,
                    'p0': 0,
                    'p1': 0,
                    'price': 0,
                    'pC': pC,
                    'Pnl': 0,
                    'x': 0,
                }
            self.preSession = False
            self.inSession = False
            self.activated = False
            self.finished = False
            self.date = day
            # print(f"Group inited: {self.centralSymbol}_{self.nc} for date {day}")
            return True, True
        return False, True

    def run_strategy(self):
        lastSection = 0
        dt = 25 if 'I' in self.sessionType == 'I' else 0
        while True:
            t0 = int(time.time())
            t0s = (t0 - dt) % SADAILY_UPDATE_INTERVAL
            sectionTs = t0 - t0s
            if sectionTs > lastSection:
                self.update(t0)
                lastSection = sectionTs
            time.sleep(1)

    def update(self, t):
        estT = estdatetime(t)
        # print(estT)
        shift = str(estT).split('-')[-1]
        day = str(estT).split(' ')[0]
        changed, ok = self.init_state(day)
        if not ok:
            print(f'Init failed for {self.centralSymbol}_{self.nc}, {day}')
            return
        # print(self.sessionType)
        if self.sessionType == 'I':
            sessionStartT = dateutil.parser.parse(f'{day} 10:30:00-{shift}', )
            preSessionT0 = dateutil.parser.parse(f'{day} 09:33:00-{shift}', )  # active interval
            preSessionT1 = sessionStartT
        else:
            sessionStartT = dateutil.parser.parse(f'{day} 10:05:00-{shift}', )
            preSessionT0 = dateutil.parser.parse(f'{day} 06:15:00-{shift}', )   # active interval
            preSessionT1 = sessionStartT
        sessionEndT = dateutil.parser.parse(f'{day} 15:30:00-{shift}', )    # defined by the strategy
        self.preSession = preSessionT0 < estT < preSessionT1
        self.inSession = sessionStartT <= estT < sessionEndT
        # print('Time:', estT)
        # print('preSessionT0', preSessionT0, 'preSessionT1', preSessionT1, preSessionT0 < estT < preSessionT1)
        # print(sessionStartT, sessionEndT)
        # print(sessionStartT - estT)
        # print(self.preSession, self.inSession)
        if self.preSession and not self.activated:
            if not self.finished and not self.activated:
                h0 = SADAILY_FLUCTUATION
                rsi_a = []
                rsi_r = []
                meanRsi = 0.0
                prices = [0] * self.nS
                for k in range(self.nS):
                    sym = self.symbols[k]
                    st = self.state[sym]
                    price = self.mdp.get_realtime_price(sym)
                    if price is None:
                        print(f'Get price failed for {sym}')
                        return
                    prices[k] = price
                    st['price'] = price
                    # print(sym, price)
                    dp = price - st['pC']
                    rsiVal = dp / self.A[k]
                    meanRsi += rsiVal
                    self.rsiValues[sym] = rsiVal
                    rsi_a.append(rsiVal)
                    rsi_r.append((k, rsiVal))
                meanRsi /= self.nS
                self.meanRsi = meanRsi
                rsi_r.sort(key=lambda r: r[1])

                if self.balancingMode == 'A':
                    rsi_c = [(k, rsiVal, abs(rsiVal)) for k, rsiVal in rsi_r if rsiVal <= meanRsi - h0 or rsiVal >= meanRsi + h0]
                    rsi_c.sort(key=lambda r: -r[2])
                    rsi_c = rsi_c[:4]
                    if len(rsi_c) > 0:
                        self.activated = True
                        changed = True
                    for r1 in rsi_c:
                        k, rsiVal = r1[:2]
                        sym = self.symbols[k]
                        if rsiVal <= meanRsi - h0:
                            self.positions[k] = 1
                            self.state[sym]['k'] = k
                            self.state[sym]['x'] = (meanRsi - rsiVal) * self.A[k]
                            self.state[sym]['activated'] = True
                            self.state[sym]['p0'] = prices[k]
                        elif rsiVal >= meanRsi + h0:
                            self.positions[k] = -1
                            self.state[sym]['k'] = k
                            self.state[sym]['x'] = (meanRsi - rsiVal) * self.A[k]
                            self.state[sym]['activated'] = True
                            self.state[sym]['p0'] = prices[k]
                else:
                    k0, baseRsi0 = rsi_r[0]
                    k1, baseRsi1 = rsi_r[-1]
                    sym0, sym1 = self.symbols[k0], self.symbols[k1]
                    price0 = self.mdp.get_ask(sym0)
                    pC0 = self.state[sym0]['pC']
                    dp0 = price0 - pC0
                    rsi0 = dp0 / self.A[k0]
                    price1 = self.mdp.get_bid(sym1)
                    pC1 = self.state[sym1]['pC']
                    dp1 = price1 - pC1
                    rsi1 = dp1 / self.A[k1]
                    # print(rsi_r)
                    if (rsi0 <= meanRsi - h0 or rsi1 >= meanRsi + h0) and rsi0 <= meanRsi <= rsi1:
                        self.k0 = k0
                        self.k1 = k1
                        self.state[sym0]['k'] = k0
                        self.state[sym1]['k'] = k1
                        self.x0 = (meanRsi - rsi0) * self.A[k0]
                        self.x1 = (meanRsi - rsi1) * self.A[k1]
                        self.state[sym0]['x'] = self.x0
                        self.state[sym1]['x'] = self.x1
                        # print(f'{sym0}: {self.x0},  {sym1}: {self.x1}')
                        self.state[sym0]['activated'] = True
                        self.state[sym0]['p0'] = price0
                        self.state[sym1]['activated'] = True
                        self.state[sym1]['p0'] = price1
                        changed = True
                        self.activated = True
                        if False:
                            GroupEchoLock.acquire(timeout=5)
                            print('--')
                            print(f'{sym0:<12s}  p:{price0:8.2f}    mid:{self.mdp.get_realtime_price(sym0):8.2f}    rsi: {baseRsi0:8.3f}  {rsi0:8.3f}  {meanRsi:8.3f}')
                            print(f'{sym1:<12s}  p:{price1:8.2f}    mid:{self.mdp.get_realtime_price(sym1):8.2f}    rsi: {baseRsi1:8.3f}  {rsi1:8.3f}  {meanRsi:8.3f}')
                            print(f'{sym0:<12s}  {rsi0:8.3f}  pC:{pC0:8.2f}  {price0:8.2f}    x0:{self.x0:8.2f}')
                            print(f'{sym1:<12s}  {rsi1:8.3f}  pC:{pC1:8.2f}  {price1:8.2f}    x1:{self.x1:8.2f}')
                            targetPnl = self.x0 / self.state[sym0]['p0'] - self.x1 / self.state[sym1]['p0']
                            print(f'targetPnl: ', targetPnl)
                            GroupEchoLock.release()
                            return False, False
            self.echo(t)
        elif self.inSession or (self.preSession and self.activated):
            if not self.finished and self.activated:
                if self.balancingMode == 'A':
                    currentPnl, targetPnl = 0, 0
                    for k in range(self.nS):
                        if k in self.positions:
                            Pos1 = self.positions[k]
                            sym = self.symbols[k]
                            cp = self.mdp.get_realtime_price(sym)
                            if cp is None:
                                print(f'Get price failed for {sym}')
                                return
                            self.state[sym]['price'] = cp
                            if self.state[sym]['closed']:
                                currentPnl += Pos1 * (self.state[sym]['p1'] - self.state[sym]['p0']) / self.state[sym]['p0']
                            else:
                                # print(self.positions)
                                # print(k, self.centralSymbol)
                                currentPnl += Pos1 * (cp - self.state[sym]['p0']) / self.state[sym]['p0']
                            x = self.state[sym]['x']
                            targetPnl += Pos1 * x / self.state[sym]['p0']
                            if not self.state[sym]['closed'] and (
                                    (Pos1 > 0 and cp >= self.state[sym]['p0'] + x) or
                                    (Pos1 < 0 and cp <= self.state[sym]['p0'] + x)
                            ):
                                self.state[sym]['p1'] = cp
                                self.state[sym]['Pnl'] = Pos1 * (cp - self.state[sym]['p0']) / self.state[sym]['p0']
                                self.state[sym]['closed'] = True
                                self.state[sym]['activated'] = False
                    if not self.activated:
                        print("a!")
                        exit()
                    c1 = True
                    if self.activated and len(self.positions) == 0:
                        print("!")
                        exit()
                    # print("positions", self.positions)
                    if len(self.positions) > 0:
                        for k in range(self.nS):
                            # print(k)
                            if k in self.positions:
                                sym = self.symbols[k]
                                # print(sym, self.state[sym]['closed'])
                                if not self.state[sym]['closed']:
                                    c1 = False
                                    break
                    if c1:
                        # print(self.centralSymbol, self.nc, 'all closed', self.positions)
                        self.activated = False
                        self.finished = True
                    if self.activated and currentPnl >= targetPnl * TARGET_PNL_RED:
                        # print(self.centralSymbol, self.nc, 'closed by pnl', currentPnl, targetPnl, self.positions)
                        for k in range(self.nS):
                            if k in self.positions:
                                Pos1 = self.positions[k]
                                sym = self.symbols[k]
                                cp = self.state[sym]['price']
                                self.state[sym]['p1'] = cp
                                self.state[sym]['Pnl'] = Pos1 * (cp - self.state[sym]['p0']) / self.state[sym]['p0']
                                self.state[sym]['closed'] = True
                                self.state[sym]['activated'] = False
                        self.activated = False
                        self.finished = True

                else: # if self.balancingMode == 'B'
                    sym0, sym1 = self.symbols[self.k0], self.symbols[self.k1]
                    if self.mdp.get_realtime_price(sym0) is None or self.mdp.get_realtime_price(sym0) is None:
                        return
                    cp0 = self.mdp.get_bid(sym0)
                    if cp0 is None:
                        print(f'Get price failed for {sym0}')
                        return
                    cp1 = self.mdp.get_ask(sym1)
                    if cp1 is None:
                        print(f'Get price failed for {sym1}')
                        return
                    self.state[sym0]['price'] = cp0
                    self.state[sym1]['price'] = cp1
                    if USE_OPEN_PRICE:
                        raise '1'
                        if self.state[sym0]['pO'] is None or self.state[sym1]['pO']:
                            print("! None")
                            exit()
                        self.state[sym0]['p0'] = self.state[sym0]['pO']
                        self.state[sym1]['p0'] = self.state[sym1]['pO']
                    currentPnl = (cp0 - self.state[sym0]['p0']) / self.state[sym0]['p0'] - (cp1 - self.state[sym1]['p0']) / self.state[sym1]['p0']
                    targetPnl = self.x0 / self.state[sym0]['p0'] - self.x1 / self.state[sym1]['p0']
                    if targetPnl < 0.01:
                        targetPnl = 0.005 + 0.5 * targetPnl
                    if (
                            ((cp0 >= self.state[sym0]['p0'] + self.x0 or cp1 <= self.state[sym1]['p0'] + self.x1) and currentPnl >= targetPnl * TARGET_PNL_MIN)
                             or currentPnl >= targetPnl * TARGET_PNL_RED
                    ) and True:
                        # closing the positions
                        self.state[sym0]['p1'] = cp0
                        self.state[sym0]['Pnl'] = (cp0 - self.state[sym0]['p0']) / self.state[sym0]['p0']
                        self.state[sym0]['closed'] = True
                        self.state[sym0]['activated'] = False
                        self.state[sym1]['p1'] = cp1
                        self.state[sym1]['Pnl'] = -(cp1 - self.state[sym1]['p0']) / self.state[sym1]['p0']
                        self.state[sym1]['closed'] = True
                        self.state[sym1]['activated'] = False
                        self.finished = True
                        self.activated = False
                        changed = True
                    else:
                        self.state[sym0]['Pnl'] = (cp0 - self.state[sym0]['p0']) / self.state[sym0]['p0']
                        self.state[sym1]['Pnl'] = -(cp1 - self.state[sym1]['p0']) / self.state[sym1]['p0']
            self.echo(t)
        elif self.activated and t >= sessionEndT:
            # closing the positions
            if self.balancingMode == 'A':
                for k in range(self.nS):
                    if k in self.positions:
                        Pos1 = self.positions[k]
                        sym = self.symbols[k]
                        cp = self.mdp.get_realtime_price(sym)
                        self.state[sym]['p1'] = cp
                        self.state[sym]['Pnl'] = Pos1 * (cp - self.state[sym]['p0']) / self.state[sym]['p0']
                        self.state[sym]['closed'] = True
                        self.state[sym]['activated'] = False
                self.finished = True
                self.activated = False
                changed = True
            else:
                sym0, sym1 = self.symbols[self.k0], self.symbols[self.k1]
                cp0 = self.mdp.get_bid(sym0)
                if cp0 is None:
                    print(f'Get price failed for {sym0}')
                    return
                cp1 = self.mdp.get_ask(sym1)
                if cp1 is None:
                    print(f'Get price failed for {sym1}')
                    return
                self.state[sym0]['p1'] = cp0
                self.state[sym0]['Pnl'] = (cp0 - self.state[sym0]['p0']) / self.state[sym0]['p0']
                self.state[sym0]['closed'] = True
                self.state[sym0]['activated'] = False
                self.state[sym1]['p1'] = cp1
                self.state[sym1]['Pnl'] = -(cp1 - self.state[sym1]['p0']) / self.state[sym1]['p0']
                self.state[sym1]['closed'] = True
                self.state[sym1]['activated'] = False
                self.finished = True
                self.activated = False
                changed = True
            self.echo(t)
        else:
            if True:
                self.echo(t)

        if changed:
            # print("Saving...")
            self.save_state()

    def echo(self, t):
        if t % SADAILY_ECHO_INTERVAL != 0:
            return
        GroupEchoLock.acquire(timeout=5)
        groupText = ''
        groupText += 'S' if self.inSession else ('P' if self.preSession else 'X')
        if self.finished:
            groupText += '  completed'
        if self.activated:
            groupText += '  active'
        groupText += f'    {self.sector}'
        print(f'--\n{COLOR_GREEN}G {self.centralSymbol}_{self.nc}    {groupText}{COLOR_0}')
        print('t:', time_est_s(t))
        print(f'{COLOR_DARKGREEN}Id   Symbol         P.C     Price       Δ        P0    Target       P1        Pnl{COLOR_0}')
        sPnl = 0
        for k in range(self.nS):
            sym = self.symbols[k]
            st = self.state[sym]
            stateText = ''
            p0Text = ' ' * 8
            p1Text = ' ' * 8
            deltaText = ' ' * 8
            targetText = ' ' * 8
            pnlText = ' ' * 8
            pC = st['pC']
            price = st['price']
            priceText = f'{price:8.2f}' if price != 0 else f'{COLOR_DARKGRAY}{price:8.2f}{COLOR_0}'
            gapText = f'{price-pC:6.2f}' if price != 0 else ' ' * 6

            skip = False
            if self.activated or self.finished:
                Pos1 = (self.positions[k] if k in self.positions else 0) if self.balancingMode == 'A' else 0
                showSymbol = (k in self.positions) if self.balancingMode == 'A' else (k == self.k0 or k == self.k1)
                if showSymbol:
                    p0 = st['p0']
                    p1 = st['p1']
                    Pnl1 = 0
                    if self.balancingMode == 'A':
                        stateText = f'{Pos1:2d} '
                    else:
                        stateText += '0  ' if k == self.k0 else '1  '
                    p0Text, targetText, pnlText = '', '', ''
                    if p0 > 0:
                        p0Text = f'{p0:8.2f}'
                        if self.balancingMode == 'A':
                            target = self.state[sym]['p0'] + self.state[sym]['x']
                        else:
                            target = p0 + self.x0 if k == self.k0 else p0 + self.x1
                        targetText = f'{target:8.2f}'
                        # Pnl1 = 0
                        if self.finished:
                            p1Text = f'{p1:8.2f}'
                            deltaText = f'{p1 - p0:6.2f}' if price != 0 else ' ' * 6
                            if self.balancingMode == 'A':
                                Pnl1 = Pos1 * (p1 - p0)/p0
                            else:
                                Pnl1 = (p1 - p0)/p0 if k == self.k0 else (p0 - p1)/p0
                            # st[self.symbols[k]]['Pnl'] = Pnl1
                        else:
                            deltaText = f'{price - p0:6.2f}' if price != 0 else ' ' * 6
                            if self.balancingMode == 'A':
                                Pnl1 = Pos1 * (price - p0)/p0
                            else:
                                Pnl1 = (price - p0)/p0 if k == self.k0 else (p0 - price)/p0
                            # st[self.symbols[k]]['Pnl'] = Pnl1
                        pnlText = f'{Pnl1:8.3f}'
                    sPnl += Pnl1

                    # if st['closed']:
                    #     p0, p1 = st['p0'], st['p1']
                    #     stateText += f'{p0:8.2f}  {p1:8.2f}'
                    # elif st['activated']:
                    #     p0 = st['p0']
                    #     stateText += f'{p0:8.2f}'
                else:
                    skip = True
                    showGap = False

            if not skip:
                pC1 = st['pC']
                price1 = st['price']
                if self.activated or self.finished:
                    print(f'{k:<3d}  {sym:<8s}  {COLOR_DARKBLUE}{pC1:8.2f}{COLOR_0}' +
                          f'  {priceText}  {deltaText}  {p0Text}  {targetText}  {p1Text}  {pnlText}   {stateText}')
                else:
                    rsiText, drsiText = ' ' * 8, ' ' * 8
                    curPrice = self.mdp.get_realtime_price(sym)
                    if st['p0'] > 0 and curPrice is not None and curPrice > 0:
                        delta = curPrice - st['p0']
                        deltaText = f'{delta:6.2f}'
                    if sym in self.rsiValues:
                        rsiVal = self.rsiValues[sym]
                        rsiText = f'{rsiVal:8.3f}'
                        drsiText = f'{rsiVal-self.meanRsi:8.3f}'
                    print(f'{k:<3d}  {sym:<8s}  {COLOR_DARKBLUE}{pC1:8.2f}{COLOR_0}' +
                          f'  {priceText}  {deltaText}  {rsiText}  {drsiText}')
        if self.activated or self.finished:
            if self.balancingMode == 'A':
                targetPnl = 0
                for k in range(self.nS):
                    if k in self.positions:
                        Pos1 = self.positions[k]
                        sym = self.symbols[k]
                        # print(self.state[sym])
                        targetPnl += Pos1 * self.state[sym]['x'] / self.state[sym]['p0']
            else:
                sym0, sym1 = self.symbols[self.k0], self.symbols[self.k1]
                targetPnl = 0
                if self.state[sym1]['p0'] > 0:
                    targetPnl = self.x0 / self.state[sym0]['p0'] - self.x1 / self.state[sym1]['p0']
            print(f'{COLOR_BLUE}                                                     {targetPnl:8.3f}            {sPnl:8.3f}{COLOR_0}')
        self.Pnl = sPnl
        GroupEchoLock.release()
