import asyncio
import time
from typing import Tuple, Optional, Dict, List
from startt import start_interaction, set_trader
from s_sadaily import SADailyStrategyRunner


class Trader(object):
    def __init__(self):
        self.strategyRunners = {}
        self.symbols: List[str] = []
        self.ask: Dict[str, float] = {}
        self.bid: Dict[str, float] = {}
        self.prices: Dict[str, float] = {}
        self.previousClose: Dict[str, float] = {}
        self.lastEchoTs = 0

    def process_md_message(self, msg_dict: dict) -> list:
        # print('MD')
        # print(msg_dict)
        for item in msg_dict['data']:
            sym = item['symbol']
            bid1, ask1 = item['bid'], item['ask']
            if bid1 is None or ask1 is None:
                continue
            if bid1 <= 0 or ask1 <= 0:
                continue
            self.ask[sym] = ask1
            self.bid[sym] = bid1
            self.prices[sym] = 0.5 * (ask1 + bid1)
            if 'close' in item:
                self.previousClose[sym] = item['close']

        t0 = int(time.time())
        t0s = t0 % 30
        sectionTs = t0 - t0s
        if sectionTs > self.lastEchoTs:
            self.lastEchoTs = sectionTs
            extra = ''
            if len(self.previousClose) >= len(self.symbols) - 10:
                extra = 'missing: ' + ' '.join(sorted([sym for sym in self.symbols if (sym.upper() not in self.previousClose)]))
            print(f'MD: {len(self.prices)}/{len(self.previousClose)} symbols loaded    {extra}')

        return []

    def process_order_report(self, msg: dict):
        print(msg)

    def get_subscription_list(self) -> Tuple[list, list]:
        all_symbol_set = set()
        for strategyClass, r in self.strategyRunners.items():
            all_symbol_set = all_symbol_set.union(r.get_set_of_symbols())
        all_symbol_list = list(sorted(list(all_symbol_set)))
        print(f'Subscribing to {len(all_symbol_list)} symbols')
        self.symbols = all_symbol_list
        all_tickers_to_subscribe = [sym.upper() for sym in all_symbol_list] # ['SPY', 'QQQ']
        # print(all_tickers_to_subscribe)
        # all_tickers_to_subscribe = ['SPY', 'QQQ', 'AA', 'CENX', 'XLY']
        # all_tickers_to_subscribe = ['SPY', 'QQQ', 'CMC']
        # print(all_tickers_to_subscribe)
        all_indicators = []
        return list(set(all_tickers_to_subscribe)), list(set(all_indicators))

    def send_market_data_to_mq(self, log: list):
        # print(log)
        ...

    def send_news_data_to_mq(self, log: list):
        ...

    def get_previous_close(self, sym1: str) -> Optional[float]:
        sym = sym1.upper()
        return self.previousClose[sym] if sym in self.previousClose else None

    def get_open(self, sym: str) -> Optional[float]:
        raise RuntimeError('Not supported')

    def get_realtime_price(self, sym1: str) -> Optional[float]:
        sym = sym1.upper()
        return self.prices[sym] if sym in self.prices else None

    def get_ask(self, sym1: str) -> float:
        sym = sym1.upper()
        return self.ask[sym]

    def get_bid(self, sym1: str) -> float:
        sym = sym1.upper()
        return self.bid[sym]

    def load(self, strategyClass, options):
        if strategyClass == 'SADaily':
            self.strategyRunners[strategyClass] = SADailyStrategyRunner(options)

    @staticmethod
    def run_trader():
        host = '10.101.3.83'
        asyncio.run(start_interaction(host))

MainTrader = Trader()
set_trader(MainTrader)

def run_strategy_daily(r_date: str, s_var: str, s_opt: str):
    options = {
        'r_date': r_date,
        's_var': s_var,
        's_opt': s_opt,
        'mdp': MainTrader,
    }
    MainTrader.load('SADaily', options)
    MainTrader.run_trader()

