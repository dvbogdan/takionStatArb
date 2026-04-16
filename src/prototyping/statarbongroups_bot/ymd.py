import threading
import time
from typing import Tuple, Optional

import yfinance as yf

YLastRequestTs = 0
YRDelta = 0.12
YLock = threading.Lock()

def y_realtime_price(ticker_symbol: str) -> float:
    global YLastRequestTs, YRDelta, YLock
    YLock.acquire()
    t = time.time()
    if t < YLastRequestTs + YRDelta:
        time.sleep(YRDelta)
    ticker = yf.Ticker(ticker_symbol)
    # fast_info дает быстрый доступ к текущей цене
    data = ticker.fast_info
    price = data['lastPrice']
    # if ticker_symbol == 'lng':
    #     print(f"Price {ticker_symbol}: {price}")
    YLastRequestTs = t
    YLock.release()
    return price

def y_previous_close(ticker_symbol: str) -> float:
    global YLastRequestTs, YRDelta, YLock
    YLock.acquire()
    t = time.time()
    if t < YLastRequestTs + YRDelta:
        time.sleep(YRDelta)
    ticker = yf.Ticker(ticker_symbol)
    # Получаем словарь с информацией
    info = ticker.info
    # Цена закрытия предыдущего дня
    previousClose = info.get('previousClose')
    # if ticker_symbol == 'lng':
    #     print(f"Цена закрытия {ticker_symbol}: {previousClose}")
    YLastRequestTs = t
    YLock.release()
    return previousClose

def y_open(ticker_symbol: str) -> float:
    global YLastRequestTs, YRDelta, YLock
    YLock.acquire()
    t = time.time()
    if t < YLastRequestTs + YRDelta:
        time.sleep(YRDelta)
    ticker = yf.Ticker(ticker_symbol)
    # Получаем словарь с информацией
    info = ticker.info
    # Цена закрытия предыдущего дня
    p = info.get('regularMarketOpen')
    # if ticker_symbol == 'lng':
    #     print(f"Open price {ticker_symbol}: {p}")
    YLastRequestTs = t
    YLock.release()
    return p

def y_sector(ticker_symbol: str) -> Tuple[str, str]:
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    # Выводим сектор
    sector = info.get('sector')
    industry = info.get('industry')
    return sector, industry

def y_meta(ticker_symbol: str) -> Tuple[str, str, str, str]:
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    # Выводим сектор
    sector = info.get('sector')
    industry = info.get('industry')
    shortPct = info.get('shortPercentOfFloat')
    quoteType = info.get('quoteType')
    return sector, industry, shortPct, quoteType

Y_UPDATE_DT = 120

class MDProviderImpl:
    def __init__(self):
        self.data = {}

    def load_quote(self, symbol: str) -> None:
        price = None
        try:
            price = y_realtime_price(symbol)
        except:
            print(f"{symbol} is not available (price)")
        if price is not None:
            self.data[symbol] = {
                'price': y_realtime_price(symbol),
                'ts': time.time()
            }

    def get_realtime_price(self, symbol: str) -> Optional[float]:
        if symbol not in self.data:
            self.load_quote(symbol)
        else:
            lastTs = self.data[symbol]['ts']
            ts = time.time()
            if ts > lastTs + Y_UPDATE_DT:
                self.load_quote(symbol)
        return self.data[symbol]['price'] if symbol in self.data else None

    @staticmethod
    def get_previous_close(symbol) -> Optional[float]:
        try:
            return y_previous_close(symbol)
        except:
            return None

    @staticmethod
    def get_open(symbol) -> Optional[float]:
        try:
            return y_open(symbol)
        except Exception as e1:
            print(e1)
            return None

    @staticmethod
    def get_sector(symbol: str) -> Tuple[str, str]:
        return y_sector(symbol)

    @staticmethod
    def get_symbol_meta(symbol: str) -> Tuple[str, str, str, str]:
        return y_meta(symbol)

MDProvider = MDProviderImpl()
