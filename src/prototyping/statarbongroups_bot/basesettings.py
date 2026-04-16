

DAYS_A_YEAR = 365
T_DAYS_A_YEAR = 252
N_YEARS_DAILY_S = 3
DATA_LEN = N_YEARS_DAILY_S * T_DAYS_A_YEAR

N_ACTIVE_SYMBOLS = 2000
N_MAX_ETF = 550
MIN_GROUP_SIZE = 5
MAX_GROUP_SIZE = 35
MAX_N_APPR_GROUPS = 3
GROUP_Q_LIST = [0.55, 0.75, 0.90, 0.95, 0.975]

EQ_WHITE = [
    'aat', 'aer', 'agesy', 'agx', 'all', 'allt', 'amsc', 'annsf', 'aple', 'app', 'arqt', 'avgo',
    'b', 'bkng', 'blbd', 'bmy', 'brk-b', 'bset',
    'caap', 'ccl', 'cde', 'cls', 'cnq', 'cop', 'cpa', 'cps', 'crdo', 'crgy', 'curi', 'cvsa',
    'dcth', 'dec', 'dis', 'drs', 'dy', 'dxpe', 'eat', 'embj', 'eslt', 'ezpw', 'flut', 'fn',
    'gass', 'ghm', 'gm', 'gsl', 'gtls', 'ibdsf', 'incy', 'kgc',
    'lite', 'lng', 'mbgaf', 'mfc', 'mu', 'nem', 'ngd', 'noc', 'ohi', 'okta', 'oust',
    'pam', 'parr', 'pltr', 'powl', 'ppc',
    'rcl', 'rdw', 'skyw', 'spot', 'ssrm', 'stng', 'strl', 'syf',
    'tigo', 'tln', 'tmus', 'ttmi', 'twlo', 'uber', 'uhs', 'unfi',
    'vici', 'visn', 'vlrs', 'vrt', 'w', 'wldn',
    'xom',
]

SYMBOL_BLACKLIST = [
    'nail',
    # takion
    'bitf',
    # delisted
    'rnam',
]

ALL_GROUPS_LIMIT_T = 30 # 60

GROUP_LIMIT = {
    'Basic Materials': 10,
    'Energy': 10,
    'Financial Services': 10,
    'Generic': 14,
}







