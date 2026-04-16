
COLOR_0 = "\033[39m"
COLOR_BLACK = "\033[30m"
COLOR_DARKRED = "\033[31m"
COLOR_DARKGREEN = "\033[32m"
COLOR_DARKYELLOW = "\033[33m"
COLOR_DARKBLUE = "\033[34m"
COLOR_DARKMAGENTA = "\033[35m"
COLOR_DARKCYAN = "\033[36m"
COLOR_LIGHTGRAY = "\033[37m"
COLOR_DARKGRAY = "\033[90m"
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_ORANGE = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_MAGENTA = "\033[95m"
COLOR_CYAN = "\033[96m"
COLOR_WHITE = "\033[97m"

WORKER_CHAR = "⊳"
STRATEGY_CHAR = "⊚"
STATUS_CHAR = STRATEGY_CHAR
CHECK_CHAR = "√"


def get_status_color(status):
    if status == "unavailable":
        return COLOR_RED
    elif status == "playback":
        return COLOR_BLUE
    elif status == "warmup":
        return COLOR_DARKMAGENTA
    elif status == "delayed":
        return COLOR_DARKYELLOW
    elif status == "online":
        return COLOR_GREEN
    else:
        return COLOR_DARKRED


def str_order(s):
    q = 0.01
    a = q
    _order = 0
    for x in s[:5]:
        _order += ord(x) * a
        a *= q
    return _order
