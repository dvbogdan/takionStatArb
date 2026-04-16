import numpy as np
from common_math import calc_daily_oc_gaps, calc_daily_oc_growth


class Group:
    def __init__(self, symbols, symbolData):
        self.symbols = symbols
        self.symbolData = symbolData
        self.nSymbols = len(symbols)
        self.data = []
        for symbol in self.symbols:
            s = symbolData[symbol]
            O = list(s['O'])
            H = list(s['H'])
            L = list(s['L'])
            C = list(s['C'])
            gaps = calc_daily_oc_gaps(O, C)
            growth = calc_daily_oc_growth(O, C)
            amplitude = np.linalg.norm(gaps) / np.sqrt(len(gaps))
            rsi = list(gaps / amplitude)
            self.data.append([
                rsi, O, H, L, C, gaps, growth, amplitude,
            ])