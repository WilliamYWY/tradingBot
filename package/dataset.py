import pandas as pd
import datetime
import os
import time


# limit 20 days or 480 data point
class sim_data: #iterable
    def __init__(self, client, crypto, start_time, end_time, interval):
        self.client = client
        self.start_time = int(start_time)
        self.end_time = int(end_time)
        self.interval = interval
        self.crypto = crypto
        self.data = self._fetch_data()

    def _fetch_data(
        self,
    ):
        data = self.client.klines(
            self.crypto, self.interval, startTime=self.start_time, endTime=self.end_time
        )
        return data

    def __getitem__(self, x):
        return self.data[x]

    def __len__(self):
        return len(self.data)


def fetch_data(
    client,
    crypto: str,
    sep: str,
    start_time: str = None,
    end_time: str = None,
    start_time_int: int = None,
    end_time_int: int = None,
):
    data_list = []
    #fetch with string date
    if start_time and end_time:
        start = datetime.datetime.strptime(start_time, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_time, "%Y-%m-%d")
    #fetch with timestamp
    elif start_time_int and end_time_int:
        start = datetime.datetime.fromtimestamp(start_time_int)
        end = datetime.datetime.fromtimestamp(end_time_int)
    cur_start = start
    # Data sep by hours
    if sep[-1] == "h":
        limit = 500 // (24 // int(sep[:-1]))
        while cur_start + datetime.timedelta(days=limit) <= end:
            cur_end = cur_start + datetime.timedelta(days=limit)
            cur_start_stamp = datetime.datetime.timestamp(cur_start) * 1000
            cur_end_stamp = datetime.datetime.timestamp(cur_end) * 1000 - 1
            data = sim_data(client, crypto, cur_start_stamp, cur_end_stamp, sep)
            data_list.append(data)
            cur_start = cur_end
            time.sleep(0.1) #prevent limit error
    # Data sep by minutes
    elif sep[-1] == "m":
        limit = 500 // (60 // int(sep[:-1]))
        while cur_start + datetime.timedelta(hours=limit) <= end:
            cur_end = cur_start + datetime.timedelta(hours=limit)
            cur_start_stamp = datetime.datetime.timestamp(cur_start) * 1000
            cur_end_stamp = datetime.datetime.timestamp(cur_end) * 1000 - 1
            data = sim_data(client, crypto, cur_start_stamp, cur_end_stamp, sep)
            data_list.append(data)
            cur_start = cur_end
            time.sleep(0.6) #prevent limit error
    # Pick up rest data
    if cur_start < end:
        cur_start_stamp = datetime.datetime.timestamp(cur_start) * 1000
        cur_end_stamp = datetime.datetime.timestamp(end) * 1000 - 1
        data = sim_data(client, crypto, cur_start_stamp, cur_end_stamp, sep)
        data_list.append(data)
    # log
    print(
        "---------DATA INFO---------\n",
        f"Fetch Data: {crypto}\n",
        f"Data Start from: {datetime.datetime.fromtimestamp(data_list[0][0][0]/1000)}\n",
        f"Data End at: {datetime.datetime.fromtimestamp(data_list[-1][-1][6]/1000)}\n",
        f"Sep. Interval: {sep}",
    )

    return data_list


def fetch_data_from_csv(data_pth, start_date, end_date):
    global data_lock
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    start_stamp = datetime.datetime.timestamp(start) * 1000
    end_stamp = datetime.datetime.timestamp(end) * 1000

    hours = ["2020-2021.csv", "2021-2022.csv", "2022-now.csv"]
    hour_file = os.path.join(data_pth, "hours")
    hours_df = pd.read_csv(os.path.join(hour_file, hours[0]), index_col=0)
    for i in range(1, len(hours)):
        df = pd.read_csv(os.path.join(hour_file, hours[i]), index_col=0)
        hours_df = pd.concat([hours_df, df], ignore_index=True)
    start_index = hours_df.loc[hours_df["0"] == start_stamp].index[0]
    end_index = hours_df.loc[hours_df["0"] == end_stamp].index[0]

    hours_df = hours_df.iloc[start_index:end_index]
    print(
        "---------DATA INFO---------\n",
        f"Fetch Data: BTCUSDT\n",
        f"Data Start from: {start_date}\n",
        f"Data End at: {end_date}\n",
        f"Sep. Interval: 1h",
    )
    return hours_df


def fetch_sim_data_from_csv():
    data_pth = "PATH/"
    minutes = [
        "2020_01-2020_05.csv",
        "2020_05-2021_01.csv",
        "2021_01-2022_05.csv",
        "2022_05-2022_09.csv",
    ]
    min_file = os.path.join(data_pth, "minutes")

    minutes_df = pd.read_csv(os.path.join(min_file, minutes[0]), index_col=0)
    for i in range(1, len(minutes)):
        df = pd.read_csv(os.path.join(min_file, minutes[i]), index_col=0)
        minutes_df = pd.concat([minutes_df, df], ignore_index=True)

    return minutes_df
