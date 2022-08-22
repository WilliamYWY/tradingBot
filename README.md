# Trading Bot
## Introduction
Using Binance API to simulate transaction strategies.  
Functions include:  
* market evaluation 
* hold & buy & sell evaluations
## Requirements
* Python 
* binance-futures-connector
* Pandas
* Numpy
## Reference 
[Binance API doc](https://binance-docs.github.io/apidocs/futures/en/#general-info)
## Installation 
```python
pip install binance-futures-connector
```
## Content
### **Intro**
First, go to [Binance Future testnet](https://testnet.binancefuture.com/en/futures/BTCUSDT) to register an API key to access service provide by Binance.  
Note that this key and the environment is only for testing, no need to worry about any losses or mistake you make (or money you earn) with this API key.
> NOTE: The price and trade are different from the real Binance Future.  
  
**Secretly** store your API key and Secret Key in .env file.  
> NOTE: In .env file 
> FUTURE_VIRTUAL_KEY = *******************  
> FUTURE_VIRTUAL_SECRET = **************** 
  
Now we can start!

### **Setting up**
First we import required packages  
```python
from binance.um_futures import UMFutures
import numpy as np
import pandas as pd
import datetime
import os
from dotenv import load_dotenv
import time
from enum import Enum
```
Start a new connection  
```python
load_dotenv()
#create client connection
client = UMFutures(base_url="https://testnet.binancefuture.com",
                    key=os.getenv("FUTURE_VIRTUAL_KEY"),
                    secret=os.getenv("FUTURE_VIRTUAL_SECRET"))
```
With **client**, we can fetch data and make transactions

### **Fetch market data**
We first define a structure for our data point  
Here we fetch k-lines  
```python
class sim_data():
    def __init__(self, client: UMFutures, crypto, start_time, end_time, interval):
        self.client = client
        self.start_time = int(start_time)
        self.end_time = int(end_time)
        self.interval = interval
        self.crypto = crypto
        self.data = self._fetch_data()

    def _fetch_data(self,):
        data = self.client.klines(self.crypto, self.interval, startTime=self.start_time, endTime=self.end_time)
        return data
    
    def __getitem__(self, x):
        return self.data[x]
    
    def __len__(self):
        return len(self.data)
```
> NOTE: Since Binance API has call-limit, we can only fetch 500 data points in one call
  
Define a function to organize data we needed with respect to:
- Start time ("2021-01-01")
- End time ("2021-01-01")
- Sep (Interval: "1h", "1m")
- crypto (Symbol: "BTCUSDT")

```python
def fetch_data(client:UMFutures, crypto:str, sep:str, start_time:str = None, end_time:str = None, start_time_int: int = None, end_time_int: int = None):    
    data_list = []
    if start_time and end_time:
        start = datetime.datetime.strptime(start_time, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_time, "%Y-%m-%d")
    elif start_time_int and end_time_int:
        start = datetime.datetime.fromtimestamp(start_time_int)
        end = datetime.datetime.fromtimestamp(end_time_int)
    cur_start = start
    # Data sep by hours
    if sep[-1] == "h": 
        limit = 500 // (24 // int(sep[:-1]))
        while cur_start + datetime.timedelta(days=limit) <= end:
            cur_end = cur_start + datetime.timedelta(days=limit)
            cur_start_stamp = datetime.datetime.timestamp(cur_start)*1000
            cur_end_stamp = datetime.datetime.timestamp(cur_end)*1000 - 1
            data = sim_data(client, crypto, cur_start_stamp, cur_end_stamp, sep)
            data_list.append(data)
            cur_start = cur_end
            time.sleep(0.1)
    # Data sep by minutes
    elif sep[-1] == "m": 
        limit = 500 // (60 // int(sep[:-1]))
        while cur_start + datetime.timedelta(hours=limit) <= end:
            cur_end = cur_start + datetime.timedelta(hours=limit)
            cur_start_stamp = datetime.datetime.timestamp(cur_start)*1000
            cur_end_stamp = datetime.datetime.timestamp(cur_end)*1000 - 1
            data = sim_data(client, crypto, cur_start_stamp, cur_end_stamp, sep)
            data_list.append(data)
            cur_start = cur_end
            time.sleep(0.1)
    # Pick up rest data
    if cur_start < end:
        cur_start_stamp = datetime.datetime.timestamp(cur_start)*1000
        cur_end_stamp = datetime.datetime.timestamp(end)*1000 - 1
        data = sim_data(client, crypto, cur_start_stamp, cur_end_stamp, sep)
        data_list.append(data)
    # log
    print(
        "---------DATA INFO---------\n",
        f"Fetch Data: {crypto}\n",
        f"Data Start from: {datetime.datetime.fromtimestamp(data_list[0][0][0]/1000)}\n",
        f"Data End at: {datetime.datetime.fromtimestamp(data_list[-1][-1][6]/1000)}\n",
        f"Sep. Interval: {sep}"
    )

    return data_list
```
This function will return a list storing sim_data with data within start time and end time

### **Market Simulation**
Now we create a object that will loop through all the data point and store corresponding info   
Parameters:
- market_data (data fectched by function above)
- bot (we will cover later)
- info_out (path storing info)
```python
class market():
    def __init__(self, market_data: list, bot: TradingBot_v2, info_out):
        self.market_data = market_data
        self.bot = bot
        self.market_history = pd.DataFrame(columns=["open time", "close time", "open price", "close price", "high", "low", "color", "action"])
        self.info_out = info_out
    def _write_info(self,):
        with open(os.path.join(self.info_out, "info.txt"), "w") as f:
            f.write(
                "----------info----------\n"+
                f"Crypto: {self.bot.crypto}\n"+
                f"Simulation time: {datetime.datetime.fromtimestamp(self.market_data[0][0][0]/1000)} - {datetime.datetime.fromtimestamp(self.market_data[-1][-1][6]/1000)}\n"+
                f"Time Interval: {self.market_data[0].interval}\n"+
                f"Strategy: {self.bot.strategy.name}\n"+
                f"Start Balance: {self.bot.balance}\n"+
                f"Leverage: {self.bot.leverage}\n"+
                f"Invest Ratio: {self.bot.invest_ratio}\n"+
                f"TRX FEE Ratio: {self.bot.TRX_FEE_ratio}\n"+
                f"Stop Loss Ratio: {self.bot.stop_loss_ratio}\n"+
                f"Stop Profit Ratio: {self.bot.stop_profit_ratio}\n" 
            )
    def start(self):
        #loop through data point
        self._write_info()
        for data in self.market_data:
            for mini_data in data:
                action = self.bot.eval(mini_data) #get action
                if action == Status.BUY: 
                    self._fill_df(mini_data, "BUY")
                elif action == Status.SELL:
                    self._fill_df(mini_data, "SELL")
                else:
                    self._fill_df(mini_data)

    def _fill_df(self, data, action=None):
        # Fill info into dataframe
        self.market_history.loc[len(self.market_history)] = [
            datetime.datetime.fromtimestamp(data[0]/1000),
            datetime.datetime.fromtimestamp(data[6]/1000),
            data[1],
            data[4],
            data[2],
            data[3],
            "RED" if float(data[4]) <= float(data[1]) else "GREEN",
            action
        ]
```
### **Enums**
Here we create some useful enum objects  
```python
class Stop_status(Enum):
    LOSS = 0
    PROFIT = 1

class Status(Enum):
    OBERVING = 0
    SELL = 1
    BUY = 2
    HOLDING_BUY = 3
    HOLDING_SELL = 4
```

### **Strategy**
Strategy is the core of a trading bot  
It will tell the bot when to place or liquidate an order  
  
There are three main function that will be called by the bot:  
- eval_market  (return None)  (this function will eval the market and update the status based on strategy)
- eval_hold    (return None or Status(Enum)) (Status.BUY to place long order, Status.SELL to place short order)
- eval_cross   (return Bool) (True to cross the order else Flase)
  
#### Here we take an example  
Assume that we want to place an order every n klines with same color.  
That is:
- Place a short order after n Red klines
- Place a long order after n Green klines
  
Then we immediately cross the order when the n+1th kline closed.  
For example, with 1 hour data interval, at 1:59PM, 2:59PM and 3:59PM the klines were all red, thus we place a **SHORT** order at 4:00PM,  
then we cross the order at 4:59PM.

Below is the code that fulfill the requirements  
Parameters:
- decision_time (oberve n klines before action)
- limit_hold_time (hold the order for how long)
```python
class interval_than_make_one():
    def __init__(self, decision_time, limit_hold_time,status=None):
        self.status = status
        self.decision_time=decision_time
        self.observed_time = 0
        self.limit_hold_time = limit_hold_time
        self.name = f"Observe_{decision_time}_and_hold_{limit_hold_time}"
    
    def eval_market(self, data):
        open_price = float(data[1])
        closed_price = float(data[4])
        if open_price >= closed_price and self.status == Status.SELL:
            self.observed_time += 1
        elif open_price >= closed_price and self.status != Status.SELL:
            self.status = Status.SELL
            self.observed_time = 1
        elif open_price < closed_price and self.status == Status.BUY:
            self.observed_time += 1
        elif open_price < closed_price and self.status != Status.BUY:
            self.status = Status.BUY
            self.observed_time = 1
    
    def eval_hold(self,):
        if self.observed_time >= self.decision_time and self.status != None:
            self.observed_time = 0
            return self.status
        else:
            return None
    
    def eval_cross(self, **kwargs):
        if kwargs["hold_time"] >= self.limit_hold_time:
            return True
        else:
            return False
```













