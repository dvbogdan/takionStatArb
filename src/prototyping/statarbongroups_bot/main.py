# This is a sample Python script.
import optparse

from analysis import select_symbols_daily
from mdaccess import load_stooq_daily, load_polygon_daily, read_symbol_info
from trader import run_strategy_daily

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option("-A", "--action", dest="action",)
    parser.add_option("-R", "--rdate", dest="rdate",)
    parser.add_option("-O", "--sopt", dest="sopt",)
    parser.add_option("-S", "--svar", dest="svar", )
    (options, args) = parser.parse_args()

    action = "load_stooq_daily"
    if options.action is not None:
        action = options.action

    s_opt = ''
    if options.sopt is not None:
        s_opt = options.sopt

    s_var = 'S'
    if options.svar is not None:
        s_var = options.svar

    if action == "load_polygon_daily":
        load_polygon_daily()
    elif action == "load_stooq_daily":
        load_stooq_daily()
    elif action == "select_symbols_daily":
        select_symbols_daily(options.rdate, s_var)
    elif action == "run_strategy_daily":
        run_strategy_daily(options.rdate, s_var, s_opt)
    elif action == "read_symbol_info":
         read_symbol_info('', options.rdate)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
