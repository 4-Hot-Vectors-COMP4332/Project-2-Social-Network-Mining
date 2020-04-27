import pandas as pd
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    ans = pd.read_csv("data/valid_ans.csv", usecols=["src", "dst", "score"])
    pred = pd.read_csv(r"data/valid_pred.csv", usecols=["src", "dst", "score"])
    df = pd.merge(ans, pred, how="left", on=["src", "dst"])
    df.fillna(0, inplace=True)
    auc_score = roc_auc_score(df["score_x"], df["score_y"])
    print("auc score:", auc_score)
