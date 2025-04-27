from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from collections import defaultdict
import time
import psutil
from datetime import datetime
import os
import matplotlib.pyplot as plt

class TrainingLogger(keras.callbacks.Callback):
    def __init__(self, log_path="logs/training_metrics.csv"):
        super().__init__()
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        header = [
            "timestamp","epoch","loss","accuracy",
            "val_loss","val_accuracy",
            "cpu_percent","mem_used_mb","mem_total_mb"
        ]
        pd.DataFrame([header]).to_csv(log_path, index=False, header=False)
        self.log_path = log_path

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        ts = datetime.now().isoformat()
        cpu = psutil.cpu_percent()
        vm = psutil.virtual_memory()
        mem_used = vm.used / 1024**2
        mem_total = vm.total / 1024**2

        row = {
            "timestamp": ts,
            "epoch": epoch,
            "loss": logs.get("loss"),
            "accuracy": logs.get("accuracy"),
            "val_loss": logs.get("val_loss"),
            "val_accuracy": logs.get("val_accuracy"),
            "cpu_percent": cpu,
            "mem_used_mb": round(mem_used,1),
            "mem_total_mb": round(mem_total,1),
        }
        pd.DataFrame([row]).to_csv(self.log_path, mode="a", index=False, header=False)


def fetch_multiple_seasons(season_years):
    all_games = []
    for season in season_years:
        print(f"Fetching season {season}…")
        gf = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable="Regular Season"
        )
        games = gf.get_data_frames()[0]
        all_games.append(games)
        time.sleep(1)
    return pd.concat(all_games, ignore_index=True)


def aggregate_team_stats(games):
    team_stats = defaultdict(lambda: {
        'points_scored':0,'points_allowed':0,'games':0,'wins':0
    })
    for _, g in games.iterrows():
        t, m, pts, res = g['TEAM_NAME'], g['MATCHUP'], g['PTS'], g['WL']
        opp = m.split('vs.')[-1] if 'vs.' in m else m.split('@')[-1]
        opp_row = games[
            (games['TEAM_ABBREVIATION']==opp)&
            (games['GAME_ID']==g['GAME_ID'])
        ]
        opp_pts = opp_row.iloc[0]['PTS'] if not opp_row.empty else np.nan
        team_stats[t]['points_scored']   += pts
        team_stats[t]['points_allowed']  += opp_pts if not np.isnan(opp_pts) else 0
        team_stats[t]['games']           += 1
        team_stats[t]['wins']            += (res=='W')
    for t, s in team_stats.items():
        gp = s['games']
        s['ppg']    = s['points_scored']/gp if gp else 0
        s['oppg']   = s['points_allowed']/gp   if gp else 0
        s['win_pct']= s['wins']/gp if gp else 0
    return pd.DataFrame.from_dict(team_stats, orient='index')


def create_real_matchups(ts_df, n=10000):
    X, y, teams = [], [], ts_df.index.tolist()
    if not teams:
        raise RuntimeError("No teams!")
    for _ in range(n):
        t1, t2 = np.random.choice(teams, 2, False)
        s1, s2 = ts_df.loc[t1], ts_df.loc[t2]
        feat = [s1['ppg'], s1['oppg'], s1['win_pct'],
                s2['ppg'], s2['oppg'], s2['win_pct']]
        X.append(feat)
        y.append(1.0 if (s1['ppg'] - s2['oppg']) > (s2['ppg'] - s1['oppg']) else 0.0)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def build_model(inp_shape):
    m = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=inp_shape),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64,  activation='relu'),
        keras.layers.Dense(1,   activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m


if __name__ == '__main__':
    # record program start
    program_start = time.time()

    seasons = ['2020-21', '2021-22', '2022-23']
    games   = fetch_multiple_seasons(seasons)
    stats   = aggregate_team_stats(games)
    X, y    = create_real_matchups(stats)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test  = y_test.astype(np.float32)

    model = build_model((X_train.shape[1],))

    logger = TrainingLogger(log_path="logs/training_metrics.csv")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[logger]
    )

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {acc:.3f}")

    os.makedirs("plots", exist_ok=True)
    df = pd.read_csv("logs/training_metrics.csv")
    plt.figure()
    plt.plot(df["epoch"], df["mem_used_mb"], marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Memory Used (MB)")
    plt.title("RAM Usage per Epoch")
    plt.grid(True)
    plt.savefig("plots/memory_usage.png", bbox_inches='tight')
    print("Saved memory usage plot to plots/memory_usage.png")

    # record program end and log run time
    program_end = time.time()
    total_seconds = program_end - program_start
    run_time_str = str(datetime.utcfromtimestamp(total_seconds).strftime("%H:%M:%S"))
    os.makedirs("logs", exist_ok=True)
    with open("logs/run_time.txt", "w") as f:
        f.write(f"Total run time: {run_time_str} (HH:MM:SS)\n")
    print(f"Total run time: {run_time_str}")
