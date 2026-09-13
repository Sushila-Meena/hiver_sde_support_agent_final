import pandas as pd

REQUIRED = [
    "tweet_id", "author_id", "inbound", "created_at", "text",
    "response_tweet_id", "in_response_to_tweet_id",
]


def load_twcs(raw_csv):
    df = pd.read_csv(raw_csv)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
    df["in_response_to_tweet_id"] = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")
    df["inbound"] = df["inbound"].astype(str).str.lower().eq("true")
    return df


def load_pairs(raw_csv, company_author_id="AppleSupport"):
    df = load_twcs(raw_csv)
    replies = df[(df.author_id.astype(str) == str(company_author_id)) & df.in_response_to_tweet_id.notna()].copy()
    customer = df[["tweet_id", "author_id", "inbound", "created_at", "text"]].rename(
        columns={"text": "customer_text", "author_id": "customer_author_id", "created_at": "customer_created_at", "inbound": "customer_inbound"}
    )
    pairs = replies.merge(customer, left_on="in_response_to_tweet_id", right_on="tweet_id", how="inner")
    pairs = pairs[
        pairs.customer_inbound & pairs.customer_text.notna() & pairs.text.notna()
    ]
    out = pd.DataFrame({
        "customer_tweet_id": pairs["tweet_id_y"],
        "customer_text": pairs.customer_text.astype(str).str.strip(),
        "support_reply": pairs.text.astype(str).str.strip(),
        "created_at": pairs.customer_created_at,
    })
    out = out[(out.customer_text.str.len() >= 4) & (out.support_reply.str.len() >= 4)]
    out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")
    return out.drop_duplicates("customer_tweet_id").sort_values("created_at").reset_index(drop=True)
