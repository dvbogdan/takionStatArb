
import asyncio
import json
import time
import traceback
from typing import Tuple

from tkconsts import *
from tkmessages import *
import tkmessages as messages

class Trader:
    def __init__(self):
        ...

    def process_md_message(self, msg_dict: dict) -> list:
        print(msg_dict)

    def process_order_report(self, msg: dict):
        print(msg)

    def get_subscription_list(self) -> Tuple[list, list]:
        # print('get_subscription_list')
        all_tickers_to_subscribe = ['SPY', 'QQQ']
        all_indicators = []
        return list(set(all_tickers_to_subscribe)), list(set(all_indicators))

    def send_market_data_to_mq(self):
        ...

trader = None # Trader()
seq_counter = 1
session_key = ''
session_host = ''

def set_trader(t):
    global trader
    trader = t

async def send_future(writer: asyncio.StreamWriter, msg: dict):
    global seq_counter
    msg[SEQ] = seq_counter
    seq_counter += 1
    writer.write(f'{json.dumps(msg)}\n\n'.encode('utf8'))
    await writer.drain()


async def send_tcp_message(writer: asyncio.StreamWriter, msg: dict):
    await asyncio.wait_for(send_future(writer, msg),
                           timeout=TIMEOUT_GRACE_PERIOD)

async def reply(writer: asyncio.StreamWriter, json_msg: dict):
    if MESSAGE_ID not in json_msg:
        return
    msg_type = json_msg[MESSAGE_ID]
    try:
        if msg_type == LOGON_TYPE:
            handle_logon(json_msg)
        elif msg_type == MARKET_DATA_TYPE:
            await handle_market_data(writer, json_msg)
        elif msg_type == NEWS_TYPE:
            handle_news(json_msg)
        elif msg_type == ORDER_RESPONSE:
            pass
        elif msg_type == ORDER_REPORT:
            handle_order_report(json_msg)
        elif msg_type == SUBSCRIBE:
            pass
    except Exception:
        print(f'Error while handling {msg_type!r} message:')
        print(traceback.format_exc())

async def handle_market_data(writer: asyncio.StreamWriter, msg: dict):

    orders_data = trader.process_md_message(msg)
    market_data = msg[DATA]
    if orders_data:
        for order_data in orders_data:
            order = order_data[ORDER_DATA]
            order[SESSION_KEY] = session_key
            order[ORDER][DATA][ACCOUNT_ID_KEY] = ACCOUNT_ID
            await send_tcp_message(
                writer,
                order
            )
        trader.send_order_log_to_mq(log=orders_data)
    trader.send_market_data_to_mq(log=market_data)


def handle_order_report(msg: dict):
    trader.process_order_report(msg)


def handle_news(msg: dict):
    trader.send_news_data_to_mq(log=[msg])

def handle_logon(msg: dict):
    global session_key
    if ERROR in msg:
        print(f'Error on logon: {msg[ERROR]}')
        return
    if SESSION_KEY not in msg:
        print(f'Error on logon, full message: {msg}')
        return
    session_key = msg[SESSION_KEY]
    print(f'Received successfull logon response, session_key: {session_key}')


async def handle_message(writer: asyncio.StreamWriter, msg: str):
    try:
        json_msg = json.loads(msg)
    except json.JSONDecodeError:
        print(f'Received a msg with incorrect format: {msg[:200]!r}')
        return
    await reply(writer, json_msg)


async def send_logon(writer: asyncio.StreamWriter):
    logon = messages.logon()
    logon[ACCOUNT_ID_KEY] = ACCOUNT_ID
    logon[VERSION_OF_APP_KEY] = VERSION_OF_APP
    await send_tcp_message(writer, logon)
    print('Logon sent')


async def send_subscribe(writer: asyncio.StreamWriter):
    while True:
        if session_key:
            subscribe = messages.subscribe()
            tickers, _ = trader.get_subscription_list()
            subscribe[SYMBOLS] = tickers
            subscribe[SESSION_KEY] = session_key
            await send_tcp_message(writer, subscribe)
            print('Subscribe sent')
            break
        else:
            await asyncio.sleep(1)


async def send_keepalive(writer: asyncio.StreamWriter):
    try:
        while True:
            if session_key:
                while True:
                    keep_alive = messages.keep_alive()
                    keep_alive[SESSION_KEY] = session_key
                    await send_tcp_message(writer, keep_alive)
                    print('Keep alive sent')
                    await asyncio.sleep(30)
            else:
                await asyncio.sleep(5)
    except ConnectionError as e:
        print(e)
        print(f'Connection lost in keepalive')
        await start_connection()
    except TimeoutError as e:
        print(e)
        print(f'Timeout error in keepalive')
        await start_connection()

async def handle_server(reader: asyncio.StreamReader,
                        writer: asyncio.StreamWriter):
    try:
        while True:
            try:
                raw = await asyncio.wait_for(
                    reader.readuntil(b'\n\n'),
                    timeout=TIMEOUT_GRACE_PERIOD)
            except asyncio.IncompleteReadError:
                print('Server closed the connection')
                break
            except asyncio.LimitOverrunError as e:
                print(f'Inbound message exceeds buffer limit: {e}; '
                      f'aborting connection')
                break
            msg = raw[:-2].decode('utf-8', 'ignore')
            if msg:
                await handle_message(writer, msg)
    except ConnectionError as e:
        print(f'Connection lost in handle_server: {e}')
    except asyncio.TimeoutError:
        print(f'Timeout (no bytes for {TIMEOUT_GRACE_PERIOD}s) in handle_server')
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f'handle_server error: {type(e).__name__}: {e.args}')
        print(traceback.format_exc())
    finally:
        if not writer.is_closing():
            writer.close()

async def start_connection():
    global seq_counter, session_key, session_host
    port = 11111
    seq_counter = 1
    session_key = ''

    reader = None
    writer = None

    for i in range(N_ATTEMPTS):
        try:
            reader, writer = await asyncio.open_connection(
                session_host, port, limit=4 * 1024 * 1024)
        except ConnectionError as e:
            print(e)
            print(f'Connection attempt no {i+1} out of {N_ATTEMPTS}'
                  f' failed...\n'
                  f'Trying to reconnect again in 5 seconds...')
            await asyncio.sleep(RECONNECTION_FREQUENCY)
            print(f'Trying connection to host {session_host}:{port}')

        if reader and writer:
            print(f'Connected successfully')
            break

    await send_logon(writer)

    await asyncio.gather(handle_server(reader, writer),
                         send_subscribe(writer),
                         send_keepalive(writer))

async def start_interaction(host: str):
    global session_host
    session_host = host
    await start_connection()


if __name__ == "__main__":
    # 192.168.137.11 / 10.101.3.83:
    # host = '192.168.137.11'
    host = '10.101.3.83'
    # host = '91.219.61.233'
    asyncio.run(start_interaction(host))