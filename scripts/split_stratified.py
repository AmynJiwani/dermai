import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv("data_index.csv")

    # Strata = Condition x AgeGroup
    df["strata"] = df["condition"].astype(str) + "||" + df["age_group"].astype(str)

    # If a strata has too few samples, stratify() can fail.
    # We'll move extremely small strata into TRAIN only (safe MVP choice).
    counts = df["strata"].value_counts()
    rare = set(counts[counts < 3].index)  # tweak if needed

    df_rare = df[df["strata"].isin(rare)]
    df_main = df[~df["strata"].isin(rare)]

    # 80/10/10 on main
    train_df, temp_df = train_test_split(
        df_main,
        test_size=0.2,
        random_state=42,
        stratify=df_main["strata"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["strata"],
    )

    # Add rare strata back into train
    train_df = pd.concat([train_df, df_rare], ignore_index=True)

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    print("Split sizes:")
    print("  train:", len(train_df))
    print("  val:  ", len(val_df))
    print("  test: ", len(test_df))
    print("Rare strata moved to train:", len(df_rare))

    # Quick check: ensure no overlap
    t = set(train_df["path"])
    v = set(val_df["path"])
    s = set(test_df["path"])
    print("\nOverlap checks (should all be 0):")
    print(" train ∩ val :", len(t & v))
    print(" train ∩ test:", len(t & s))
    print(" val ∩ test  :", len(v & s))

if __name__ == "__main__":
    main()
