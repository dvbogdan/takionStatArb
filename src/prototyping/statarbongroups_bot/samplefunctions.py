import json
from datetime import datetime, timedelta, date
import math

from app.database import get_clickhouse_client
from app.server.const import frequency_to_table
import logging

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def close_to_close_formula(tickers, from_timestamp, to_timestamp, frequency, filter_conditions="",
                           formula_name="close_to_close", operator="AND"):
    """
    Executes the 'close_to_close' formula and returns the data, using the filter conditions
    passed directly as a pre-constructed WHERE clause extension.

    IMPORTANT: We skip any date where NOT all of the requested tickers are present
    (after the filter conditions are applied).
    """
    table_name = frequency_to_table[frequency]
    tickers_list = ", ".join([f"'{ticker}'" for ticker in tickers])
    tickers_count = len(tickers)  # We'll need this to ensure we have all tickers
    if operator.upper() == "AND":
        having_clause = f"HAVING count(DISTINCT symbol) = {tickers_count}"
    else:
        # OR => do not restrict to having all symbols
        having_clause = ""

    ticker_selects = []
    for ticker in tickers:
        ticker_selects.append(f"anyIf({formula_name}, symbol='{ticker}') AS `{ticker}`")
    ticker_selects_str = ",\n        ".join(ticker_selects)

    client = get_clickhouse_client()

    # We do the calculation in a subquery (CTE: "raw_data") and filter out rows
    # that don't pass your filter_conditions. Then we collect timestamps that
    # have all tickers in "valid_timestamps". Finally, we select only from those
    # timestamps in the outer query.
    #
    # This approach ensures that if ANY ticker is missing (or filtered out)
    # on a particular date, that date is entirely skipped in the final result.
    #
    main_query = f"""
    WITH 
    raw_data AS (
        SELECT 
            timestamp,
            symbol,
            (close - lagInFrame(close, 1) OVER (PARTITION BY symbol ORDER BY timestamp))
                / nullIf(lagInFrame(close, 1) OVER (PARTITION BY symbol ORDER BY timestamp), 0) * 100 
            AS {formula_name},
            close
        FROM {table_name} FINAL
        WHERE 
            timestamp >= toDateTime({int(from_timestamp)}, 'UTC')
            AND timestamp < toDateTime({int(to_timestamp)}, 'UTC')
            AND symbol IN ({tickers_list})
        ORDER BY timestamp ASC
    ),

    -- valid_timestamps are those timestamps where *all* of the requested tickers
    -- survived the filter conditions.
    valid_timestamps AS (
        SELECT timestamp
        FROM raw_data
        WHERE 1 = 1 {filter_conditions}
        GROUP BY timestamp
        {having_clause}  -- <-- only present if operator=AND
    )

    SELECT 
        t.timestamp,
        {ticker_selects_str}
    FROM raw_data AS t
    WHERE 
        t.timestamp IN (SELECT timestamp FROM valid_timestamps)
    GROUP BY t.timestamp
    ORDER BY t.timestamp ASC
    """

    result = client.query(main_query)
    data_rows = result.result_rows

    response_data = []
    for row in data_rows:
        # row[0] is the timestamp (as a Python datetime),
        # row[1..] are the pivoted formula values in the order we built above
        timestamp_dt = row[0]
        timestamp_ms = int(datetime.timestamp(timestamp_dt) * 1000)

        row_dict = {"Date": timestamp_ms}
        for i, ticker in enumerate(tickers, start=1):
            val = row[i]
            if val is None or math.isnan(val):
                val = None
            row_dict[ticker] = val

        response_data.append(row_dict)

    return response_data


def intra_formula(
    tickers,
    from_timestamp,
    to_timestamp,
    frequency,
    filter_conditions="",
    formula_name="intra",
    operator="AND"
):
    """
    Executes the 'intra' formula and returns the data, skipping timestamps
    where NOT all tickers are present after applying filters, and preserving
    the order of tickers in the response.

    :param tickers: List of ticker symbols
    :param from_timestamp: Start timestamp (int)
    :param to_timestamp: End timestamp (int)
    :param frequency: Data frequency ('1' for minute, '1D' for daily)
    :param filter_conditions: Additional filter conditions (as a WHERE clause snippet)
    :param formula_name: The column name for the formula (defaults to "intra")
    :return: List of dictionaries containing the data (one row per timestamp).
    """
    table_name = frequency_to_table[frequency]
    tickers_list = ", ".join([f"'{ticker}'" for ticker in tickers])
    tickers_count = len(tickers)

    # Build ticker-select clauses for pivoting
    # E.g. for each ticker T, "anyIf({formula_name}, symbol='T') AS `T`"
    ticker_selects = []
    for ticker in tickers:
        ticker_selects.append(f"anyIf({formula_name}, symbol = '{ticker}') AS `{ticker}`")
    ticker_selects_str = ",\n        ".join(ticker_selects)

    if operator.upper() == "AND":
        having_clause = f"HAVING count(DISTINCT symbol) = {tickers_count}"
    else:
        # OR => do not restrict to having all symbols
        having_clause = ""

    client = get_clickhouse_client()

    # Use a subquery (raw_data) to calculate the formula, then filter out timestamps
    # that don't have *all* tickers (valid_timestamps). Finally, pivot by ticker
    # in the outer query to preserve column ordering.
    sql_query = f"""
    WITH
    raw_data AS (
        SELECT 
            timestamp,
            symbol,
            (close - open) / nullIf(open, 0) * 100 AS {formula_name}
        FROM {table_name} FINAL
        WHERE 
            timestamp >= toDateTime({int(from_timestamp)}, 'UTC')
            AND timestamp <= toDateTime({int(to_timestamp)}, 'UTC')
            AND symbol IN ({tickers_list})
        ORDER BY timestamp ASC
    ),

    valid_timestamps AS (
        SELECT timestamp
        FROM raw_data
        WHERE 1=1 {filter_conditions}
          AND {formula_name} IS NOT NULL
        GROUP BY timestamp
        {having_clause}  -- <-- only present if operator=AND
    )

    SELECT
        t.timestamp,
        {ticker_selects_str}
    FROM raw_data AS t
    WHERE
        t.{formula_name} IS NOT NULL
        AND t.timestamp IN (SELECT timestamp FROM valid_timestamps)
    GROUP BY t.timestamp
    ORDER BY t.timestamp ASC
    """

    logger.info(f"Executing SQL query: {sql_query}")

    try:
        result = client.query(sql_query)
        data_rows = result.result_rows

        response_data = []

        # Each row is of the form:
        #   row[0] = timestamp (datetime object)
        #   row[1] = formula value for tickers[0]
        #   row[2] = formula value for tickers[1]
        #   ...
        for row in data_rows:
            timestamp_dt = row[0]
            timestamp_ms = int(datetime.timestamp(timestamp_dt) * 1000)

            row_dict = {"Date": timestamp_ms}
            # Fill in each ticker's pivoted value in the correct order
            for i, ticker in enumerate(tickers, start=1):
                val = row[i]
                if val is None or math.isnan(val):
                    val = None
                row_dict[ticker] = val

            response_data.append(row_dict)

        return response_data

    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        raise e


def gap_formula(
    tickers,
    from_timestamp,
    to_timestamp,
    frequency,
    filter_conditions="",
    formula_name="gap",
    operator="AND"
):
    """
    Executes the 'gap' formula and returns the data, calculating gaps based on
    the previous close price for each day, skipping dates where NOT all tickers
    are present after applying filters, and preserving the order of the tickers
    in the final JSON.

    :param tickers: List of ticker symbols
    :param from_timestamp: Start timestamp (int, seconds since epoch)
    :param to_timestamp: End timestamp (int, seconds since epoch)
    :param frequency: Data frequency (must be '1D' for gap formula)
    :param filter_conditions: Additional filter conditions (as a WHERE-clause snippet)
    :param formula_name: Name for the resulting formula column (default='gap')
    :return: List of dictionaries, each containing a 'Date' (in ms) and each ticker's gap value.
    """

    # gap formula only supports daily data
    if frequency != "1D":
        raise ValueError("Gap formula is only supported for daily frequency data ('1D')")

    table_name = frequency_to_table[frequency]
    tickers_list = ", ".join([f"'{ticker}'" for ticker in tickers])
    tickers_count = len(tickers)

    # Build pivot expressions for each ticker
    # E.g. anyIf(gap, symbol = 'XYZ') AS `XYZ`
    ticker_selects = []
    for ticker in tickers:
        ticker_selects.append(f"anyIf({formula_name}, symbol = '{ticker}') AS `{ticker}`")
    ticker_selects_str = ",\n        ".join(ticker_selects)

    if operator.upper() == "AND":
        having_clause = f"HAVING count(DISTINCT symbol) = {tickers_count}"
    else:
        # OR => do not restrict to having all symbols
        having_clause = ""

    client = get_clickhouse_client()

    # Main query:
    #
    #  1. raw_data subquery computes the gap per symbol at each date
    #  2. valid_timestamps only keeps those dates with *all* tickers present (post-filter)
    #  3. final SELECT pivots the gap formula into separate columns per ticker,
    #     preserving ticker order from the original list
    #
    main_query = f"""
    WITH
    raw_data AS (
        SELECT
            timestamp,
            symbol,
            (open - lagInFrame(close, 1) OVER (PARTITION BY symbol ORDER BY timestamp))
                / nullIf(lagInFrame(close, 1) OVER (PARTITION BY symbol ORDER BY timestamp), 0) * 100
            AS {formula_name}
        FROM {table_name} FINAL
        WHERE
            timestamp >= toDateTime({int(from_timestamp)}, 'UTC')
            AND timestamp < toDateTime({int(to_timestamp)}, 'UTC')
            AND symbol IN ({tickers_list})
        ORDER BY timestamp ASC
    ),

    valid_timestamps AS (
        SELECT timestamp
        FROM raw_data
        WHERE {formula_name} IS NOT NULL
              {filter_conditions}
        GROUP BY timestamp
        {having_clause}  -- <-- only present if operator=AND
    )

    SELECT
        t.timestamp,
        {ticker_selects_str}
    FROM raw_data AS t
    WHERE
        t.{formula_name} IS NOT NULL
        AND t.timestamp IN (SELECT timestamp FROM valid_timestamps)
    GROUP BY t.timestamp
    ORDER BY t.timestamp ASC
    """

    logger.info(f"Executing gap formula SQL query:\n{main_query}")

    try:
        result = client.query(main_query)
        data_rows = result.result_rows

        response_data = []

        # Each row is of the form:
        #   [timestamp_datetime, gap_for_tickers[0], gap_for_tickers[1], ...]
        for row in data_rows:
            timestamp_dt = row[0]
            timestamp_ms = int(datetime.timestamp(timestamp_dt) * 1000)

            row_dict = {"Date": timestamp_ms}
            for i, ticker in enumerate(tickers, start=1):
                val = row[i]
                # Convert NaN to None
                if val is None or math.isnan(val):
                    val = None
                row_dict[ticker] = val

            response_data.append(row_dict)

        return response_data

    except Exception as e:
        logger.error(f"Error executing gap formula query: {str(e)}")
        raise e


def opn_formula(tickers, from_timestamp, to_timestamp, time_str, filter_conditions="", formula_name="", operator="AND"):
    """
    Calculates the opn_{time_str} formula, i.e.:
        (open_X - day_open) / day_open * 100
    for each date, using:
    - The official daily open from ohlcv_day (9:30 AM open).
    - The minute-level price up to the specified time_str (from ohlcv_minute).

    Skips any date where NOT all tickers are present after applying filters
    (when operator="AND"). The result is pivoted, preserving the ticker order.

    :param tickers: List of ticker symbols
    :param from_timestamp: Start time in seconds since epoch (int)
    :param to_timestamp: End time in seconds since epoch (int)
    :param time_str: A string "HH_MM" representing the target time to compare
                    (e.g. "10_30" for 10:30 AM).
    :param filter_conditions: Additional SQL filter conditions (default: "")
    :param formula_name: Optional. If empty, it will be set to "opn_{time_str}" internally.
    :param operator: "AND" (default) to require all tickers each day, or "OR" otherwise
    :return: List of dicts, each with 'Date' (in ms) + each ticker's opn_{time_str} value.
    """

    # If no specific formula_name is provided, default to "opn_{time_str}"
    if not formula_name:
        formula_name = f"opn_{time_str}"

    # We'll still use "1" to get the correct table name for minute data
    frequency = "1"
    minute_table_name = frequency_to_table[frequency]  # e.g. "ohlcv_minute"

    tickers_list = ", ".join(f"'{ticker}'" for ticker in tickers)
    tickers_count = len(tickers)

    # Parse the target time (e.g. "10_30" -> hour=10, minute_val=30)
    hour, minute_val = map(int, time_str.split('_'))

    # Build the pivot clauses for the final SELECT
    # e.g. anyIf(opn_10_30, symbol='XYZ') AS `XYZ`
    ticker_selects = [
        f"anyIf({formula_name}, symbol = '{ticker}') AS `{ticker}`"
        for ticker in tickers
    ]
    ticker_selects_str = ",\n        ".join(ticker_selects)

    # If operator=AND, we require that *all* symbols appear each date
    if operator.upper() == "AND":
        having_clause = f"HAVING count(DISTINCT symbol) = {tickers_count}"
    else:
        # OR => no restriction that all tickers must appear each day
        having_clause = ""

    # ----------------------------------------------------------------------
    # Construct the query using multiple CTEs:
    #   1) day_data: get official open from daily table
    #   2) minute_data: get minute price up to hour:minute_val
    #   3) raw_data: join & compute (open_X - day_open)/day_open * 100
    #   4) valid_dates: apply optional filter/having to ensure all symbols
    #   5) final SELECT: pivot
    # ----------------------------------------------------------------------
    sql_query = f"""
    WITH

    day_data AS (
        SELECT
            toDate(timestamp) AS date,
            symbol,
            -- daily table 'open' = official 9:30 open
            any(open) AS day_open
        FROM ohlcv_day
        WHERE
            toDate(timestamp) >= toDate(toDateTime({int(from_timestamp)}, 'UTC'))
            AND toDate(timestamp) <= toDate(toDateTime({int(to_timestamp)}, 'UTC'))
            AND symbol IN ({tickers_list})
        GROUP BY date, symbol
    ),

    minute_data AS (
        SELECT
            toDate(timestamp) AS date,
            symbol,
            -- Pick the last open price up to HH:MM (inclusive)
            argMaxIf(
                open,
                timestamp,
                (toHour(timestamp) < {hour})
                 OR (toHour(timestamp) = {hour} AND toMinute(timestamp) <= {minute_val})
            ) AS open_X
        FROM {minute_table_name} FINAL
        WHERE
            toDate(timestamp) >= toDate(toDateTime({int(from_timestamp)}, 'UTC'))
            AND toDate(timestamp) <= toDate(toDateTime({int(to_timestamp)}, 'UTC'))
            AND symbol IN ({tickers_list})
        GROUP BY date, symbol
    ),

    raw_data AS (
        SELECT
            dd.date,
            dd.symbol,
            dd.day_open,
            md.open_X,
            (md.open_X - dd.day_open) / nullIf(dd.day_open, 0) * 100 AS {formula_name}
        FROM day_data dd
        JOIN minute_data md USING (date, symbol)
        -- We only want rows where both opens exist
        WHERE dd.day_open IS NOT NULL
          AND md.open_X IS NOT NULL
    ),

    valid_dates AS (
        SELECT date
        FROM raw_data
        WHERE
            {formula_name} IS NOT NULL
            {filter_conditions}
        GROUP BY date
        {having_clause}
    )

    SELECT
        rd.date,
        {ticker_selects_str}
    FROM raw_data rd
    WHERE
        rd.{formula_name} IS NOT NULL
        AND rd.date IN (SELECT date FROM valid_dates)
    GROUP BY rd.date
    ORDER BY rd.date ASC
    """

    logger.info(f"Executing opn formula SQL query:\n{sql_query}")

    client = get_clickhouse_client()
    try:
        result = client.query(sql_query)
        data_rows = result.result_rows

        response_data = []

        # Each row is of the form:
        #   row[0] = date (type Date in ClickHouse)
        #   row[1..] = pivoted {formula_name} values for each ticker
        for row in data_rows:
            date_obj = row[0]
            # Convert date to Python datetime at midnight
            date_ts = datetime.combine(date_obj, datetime.min.time())
            date_ms = int(date_ts.timestamp() * 1000)

            row_dict = {"Date": date_ms}
            # For each ticker in order, get the pivoted value
            for i, ticker in enumerate(tickers, start=1):
                val = row[i]
                # Convert NaN to None
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    val = None
                row_dict[ticker] = val

            response_data.append(row_dict)

        # Log the final response
        logger.info(f"response_data: {json.dumps(response_data)}")

        return response_data

    except Exception as e:
        logger.error(f"Error executing query for opn formula: {str(e)}")
        raise e


import math
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


# def cls_formula(
#         tickers,
#         from_timestamp,
#         to_timestamp,
#         time_str,
#         filter_conditions="",
#         formula_name="cls",
#         operator="AND"
# ):
#     """
#     Cls_X formula:
#       (open_X - prev_close) / prev_close * 100 (%)
#
#     Steps:
#       1. Extract open_X from the minute table (using argMaxIf over the interval
#          from market open until the target time, e.g. 9_35).
#       2. Join with the daily ohlcv_day table to get the previous day’s close.
#       3. Compute the percent change.
#       4. Optionally filter and pivot the result.
#
#     Arguments:
#       - tickers: list of ticker symbols.
#       - from_timestamp: starting timestamp (seconds since epoch) for data retrieval.
#       - to_timestamp: ending timestamp (seconds since epoch) for data retrieval.
#       - time_str: target time in the format "HH_MM" (e.g. "9_35").
#       - filter_conditions: additional SQL conditions to apply.
#       - formula_name: name for the computed formula column.
#       - operator: if "AND", only days with all tickers present are returned; if "OR", days with any ticker are returned.
#     """
#     # Table names
#     minute_table = "ohlcv_minute"
#     daily_table = "ohlcv_day"
#
#     # Build ticker list for SQL IN clause
#     tickers_list = ", ".join([f"'{ticker}'" for ticker in tickers])
#     tickers_count = len(tickers)
#
#     # Parse the target time "HH_MM"
#     hour, minute_val = map(int, time_str.split('_'))
#     target_minutes = hour * 60 + minute_val
#
#     # Update the market open minutes to 9:00 AM since that's when your data begins.
#     market_open_minutes = 9 * 60  # 9:00 AM in minutes (540 minutes)
#
#     # Build pivot expressions: anyIf({formula_name}, symbol='TICKER') AS TICKER
#     ticker_selects = []
#     for ticker in tickers:
#         ticker_selects.append(f"anyIf({formula_name}, symbol = '{ticker}') AS {ticker}")
#     ticker_selects_str = ",\n        ".join(ticker_selects)
#
#     # For operator "AND" we require all tickers to be present on a day.
#     if operator.upper() == "AND":
#         having_clause = f"HAVING count(DISTINCT symbol) = {tickers_count}"
#     else:
#         having_clause = ""
#
#     client = get_clickhouse_client()  # Assumes you have this helper defined
#
#     # Main query: note the join uses subtractDays to get the previous day's close.
#     main_query = f"""
#     WITH
#     raw_data AS (
#         SELECT
#             oX.date AS dt,
#             oX.symbol,
#             (oX.open_X - pc.prev_close) / nullIf(pc.prev_close, 0) * 100 AS {formula_name}
#         FROM
#         (
#             SELECT
#                 symbol,
#                 toDate(timestamp) AS date,
#                 argMaxIf(
#                     nullIf(open, 0),
#                     timestamp,
#                     (toHour(timestamp) * 60 + toMinute(timestamp)) <= {target_minutes}
#                     AND (toHour(timestamp) * 60 + toMinute(timestamp)) >= {market_open_minutes}
#                 ) AS open_X
#             FROM {minute_table} FINAL
#             WHERE
#                 timestamp >= toDateTime({int(from_timestamp)}, 'UTC')
#                 AND timestamp < toDateTime({int(to_timestamp)}, 'UTC')
#                 AND symbol IN ({tickers_list})
#             GROUP BY
#                 symbol,
#                 toDate(timestamp)
#         ) AS oX
#         LEFT JOIN
#         (
#             SELECT
#                 symbol,
#                 toDate(timestamp) AS date,
#                 close AS prev_close
#             FROM {daily_table}
#             WHERE
#                 symbol IN ({tickers_list})
#         ) AS pc
#         ON
#             oX.symbol = pc.symbol
#             AND pc.date = subtractDays(oX.date, 1)
#         WHERE
#             oX.open_X IS NOT NULL
#             AND pc.prev_close IS NOT NULL
#             AND pc.prev_close > 0
#         ORDER BY
#             oX.date ASC
#     ),
#
#     valid_dates AS (
#         SELECT dt
#         FROM raw_data
#         WHERE {formula_name} IS NOT NULL
#               {filter_conditions}
#         GROUP BY dt
#         {having_clause}
#     )
#
#     SELECT
#         rd.dt,
#         {ticker_selects_str}
#     FROM raw_data AS rd
#     WHERE
#         rd.{formula_name} IS NOT NULL
#         AND rd.dt IN (SELECT dt FROM valid_dates)
#     GROUP BY rd.dt
#     ORDER BY rd.dt ASC
#     """
#
#     logger.info(f"Executing CLS formula SQL query:\n{main_query}")
#     result = client.query(main_query)
#     data_rows = result.result_rows
#
#     # Process query results into a JSON-like structure.
#     response_data = []
#     for row in data_rows:
#         # Convert the ClickHouse date (row[0]) into a Python datetime at midnight
#         date_obj = row[0]
#         if isinstance(date_obj, date) and not isinstance(date_obj, datetime):
#             date_obj = datetime.combine(date_obj, datetime.min.time())
#         date_ms = int(date_obj.timestamp() * 1000)
#
#         row_dict = {"Date": date_ms}
#         for i, ticker in enumerate(tickers, start=1):
#             val = row[i]
#             # If the value is None or NaN, set it to None for JSON compatibility.
#             if val is None or (isinstance(val, float) and math.isnan(val)):
#                 val = None
#             row_dict[ticker] = val
#
#         response_data.append(row_dict)
#
#     return response_data


def cls_formula(
        tickers,
        from_timestamp,
        to_timestamp,
        time_str,
        filter_conditions="",
        formula_name="cls",
        operator="AND"
):
    """
    Cls_X formula using the previous trading day's close.

    Calculation:
         (open_X - prev_close) / prev_close * 100

    Steps:
      1. Extract open_X from the minute table at a specified target time (e.g. 9_35).
      2. Compute the previous trading day's close (using a window function over the daily table).
      3. Compute the percent change.
      4. Optionally filter and pivot the result.

    Note:
      - With this approach, a Monday’s data will use the Friday close from the daily table.
      - Make sure your ClickHouse version supports window functions.
    """
    # Table names
    minute_table = "ohlcv_minute"
    daily_table = "ohlcv_day"

    # Build ticker list for SQL IN clause
    tickers_list = ", ".join([f"'{ticker}'" for ticker in tickers])
    tickers_count = len(tickers)

    # Parse target time "HH_MM"
    hour, minute_val = map(int, time_str.split('_'))
    target_minutes = hour * 60 + minute_val

    # Update market open time to 9:00 AM (since your data starts at 9:00)
    market_open_minutes = 9 * 60  # 9:00 AM (540 minutes)

    # Build pivot expressions: e.g. anyIf(cls, symbol='TICKER') AS TICKER
    ticker_selects = []
    for ticker in tickers:
        ticker_selects.append(f"anyIf({formula_name}, symbol = '{ticker}') AS {ticker}")
    ticker_selects_str = ",\n        ".join(ticker_selects)

    # For operator "AND", require all tickers to be present on a day.
    if operator.upper() == "AND":
        having_clause = f"HAVING count(DISTINCT symbol) = {tickers_count}"
    else:
        having_clause = ""

    client = get_clickhouse_client()  # Assumes this helper function is defined elsewhere

    # Main query:
    # - The CTE 'daily_prev' computes the previous trading day’s close using lag().
    # - The CTE 'raw_data' extracts the open_X from the minute table and joins with daily_prev on the same trading day.
    #   For each symbol and trading day (oX.date), dp.prev_close is the close of the previous trading day.
    main_query = f"""
    WITH
    daily_prev AS (
       SELECT
          symbol,
          toDate(timestamp) AS current_date,
          lagInFrame(close) OVER (PARTITION BY symbol ORDER BY toDate(timestamp)) AS prev_close
       FROM {daily_table}
       WHERE symbol IN ({tickers_list})
    ),
    raw_data AS (
       SELECT
           oX.date AS dt,
           oX.symbol,
           (oX.open_X - dp.prev_close) / nullIf(dp.prev_close, 0) * 100 AS {formula_name}
       FROM
       (
           SELECT
               symbol,
               toDate(timestamp) AS date,
               argMaxIf(
                   nullIf(open, 0),
                   timestamp,
                   (toHour(timestamp) * 60 + toMinute(timestamp)) <= {target_minutes}
                   AND (toHour(timestamp) * 60 + toMinute(timestamp)) >= {market_open_minutes}
               ) AS open_X
           FROM {minute_table} FINAL
           WHERE
               timestamp >= toDateTime({int(from_timestamp)}, 'UTC')
               AND timestamp < toDateTime({int(to_timestamp)}, 'UTC')
               AND symbol IN ({tickers_list})
           GROUP BY
               symbol,
               toDate(timestamp)
       ) AS oX
       LEFT JOIN daily_prev dp
         ON oX.symbol = dp.symbol AND oX.date = dp.current_date
       WHERE
           oX.open_X IS NOT NULL
           AND dp.prev_close IS NOT NULL
           AND dp.prev_close > 0
       ORDER BY
           oX.date ASC
    ),
    valid_dates AS (
       SELECT dt
       FROM raw_data
       WHERE {formula_name} IS NOT NULL
             {filter_conditions}
       GROUP BY dt
       {having_clause}
    )
    SELECT
       rd.dt,
       {ticker_selects_str}
    FROM raw_data AS rd
    WHERE
       rd.{formula_name} IS NOT NULL
       AND rd.dt IN (SELECT dt FROM valid_dates)
    GROUP BY rd.dt
    ORDER BY rd.dt ASC
    """

    logger.info(f"Executing CLS formula SQL query:\n{main_query}")
    result = client.query(main_query)
    data_rows = result.result_rows

    # Process the query results into JSON-friendly format.
    response_data = []
    for row in data_rows:
        # row[0] is the date from ClickHouse (type Date); convert it to a Python datetime at midnight.
        date_obj = row[0]
        if isinstance(date_obj, date) and not isinstance(date_obj, datetime):
            date_obj = datetime.combine(date_obj, datetime.min.time())
        date_ms = int(date_obj.timestamp() * 1000)

        row_dict = {"Date": date_ms}
        for i, ticker in enumerate(tickers, start=1):
            val = row[i]
            # Convert any NaN to None for JSON compatibility.
            if val is None or (isinstance(val, float) and math.isnan(val)):
                val = None
            row_dict[ticker] = val

        response_data.append(row_dict)

    return response_data