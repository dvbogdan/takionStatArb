from datetime import datetime, timedelta
from dateutil import parser as datetime_parser
from dateutil.tz import gettz
from functools import wraps
from pytz import timezone
from time import time
from typing import Union, Tuple

import pycommon.time.helper as timehelper
import pytrade.db.calendar as calendarmod
import pytrade.db.calendar.holiday as holiday

TimeInSeconds = int
Resolution = TimeInSeconds

DAY = 24 * 3600


def estdatetime(t: Union[TimeInSeconds, int, str, datetime]) -> datetime:
    if type(t) is datetime:
        return t
    elif type(t) is TimeInSeconds or type(t) is int:
        return datetime.fromtimestamp(t, tz=timezone("US/Eastern"))
    elif type(t) is str:
        return datetime_parser.parse(
            t + " EST", tzinfos={"EST": gettz("America/New York")}
        )  # '%Y-%m-%d %H:%M:%S EST'
    else:
        raise RuntimeError("Unrecognized argument type: {}".format(type(t)))


def utcdatetime(t: Union[TimeInSeconds, int, str, datetime]) -> datetime:
    if type(t) is datetime:
        return t
    elif type(t) is TimeInSeconds or type(t) is int:
        return datetime.fromtimestamp(t, tz=timezone("UTC"))
    elif type(t) is str:
        if "-" not in t:
            t = t[:4] + "-" + t[4:6] + "-" + t[6:]
        if ":" not in t:
            t += " 00:00:00"
        return datetime_parser.parse(
            t + " UTC", tzinfos={"UTC": gettz("UTC")}
        )  # '%Y-%m-%d %H:%M:%S UTC'
    else:
        raise RuntimeError("Unrecognized argument type: {}".format(type(t)))


def utcts(t: Union[TimeInSeconds, int, str, datetime]) -> TimeInSeconds:
    return int(utcdatetime(t).timestamp())


def today() -> str:
    return utcdatetime(datetime.now()).strftime("%Y%m%d")

def today_and_ts() -> Tuple[str, int]:
    ts1 = utcts(datetime.now())
    return utcdatetime(ts1).strftime("%Y%m%d"), ts1


def get_previous_day(day: str) -> str:
    date = utcdatetime(day)
    prev_date = date - timedelta(days=1)
    prev_day = prev_date.strftime("%Y%m%d")
    return prev_day


def num_days_in_week(day: str):
    day1 = (utcdatetime(day) + timedelta(days=7)).strftime("%Y%m%d")
    simStartDate, simEndDate = timehelper.get_start_end_date(day, day1)
    dateListNoWe = timehelper.get_date_range_no_we_hol(
        calendarmod.USEQHG, simStartDate, simEndDate
    )
    return len(dateListNoWe)


def num_working_days_in_interval(day: str, day1: str):
    simStartDate, simEndDate = timehelper.get_start_end_date(day, day1)
    dateListNoWe = timehelper.get_date_range_no_we_hol(
        calendarmod.USEQHG, simStartDate, simEndDate
    )
    print(dateListNoWe)
    return len(dateListNoWe)


def num_days_in_interval(day: str, day1: str):
    return (utcdatetime(day1) - utcdatetime(day)).days


def get_phase_from_time(t: datetime, period: TimeInSeconds) -> TimeInSeconds:
    return int(t.timestamp()) % period


def serialize_resolution(r: Resolution) -> str:
    return "{}s".format(r)


def parse_resolution(s: str) -> Resolution:
    if type(s) is int:
        return s
    if not s or s[-1] != "s":
        raise RuntimeError("Invalid format of resolution: {}".format(s))
    return int(s[:-1])


def ts_and_time_s(t: TimeInSeconds) -> str:
    return "{} ({})".format(t, utcdatetime(t).strftime("%Y%m%d %H:%M:%S"))


def ts_and_time_est_s(t: TimeInSeconds) -> str:
    return "{} ({})".format(t, estdatetime(t).strftime("%Y%m%d %H:%M:%S"))


def time_est_s(t: TimeInSeconds) -> str:
    return "{}".format(estdatetime(t).strftime("%Y%m%d %H:%M:%S"))


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        print("func:%r took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap
