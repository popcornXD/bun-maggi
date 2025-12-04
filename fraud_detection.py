import pandas as pd
from pathlib import Path
import sys

# ---------------------------
# Utility helpers
# ---------------------------
def list_columns_notice(df, label):
    print(f"\nColumns found in {label}:")
    for c in df.columns:
        print(" -", c)

def ask_column(name, df_columns, default_candidates=None):
    """
    Ask the user which column to use for `name`.
    - df_columns: list-like of available columns (strings)
    - default_candidates: ordered list of possible defaults (strings)
    Returns selected column name (must be one of df_columns).
    """
    default = None
    if default_candidates:
        for cand in default_candidates:
            if cand in df_columns:
                default = cand
                break

    prompt_lines = []
    prompt_lines.append(f"\nWhich column should be used for '{name}'?")
    if default:
        prompt_lines.append(f"Press Enter to accept default: '{default}'")
    else:
        prompt_lines.append("No good default detected; you must type one of the column names shown above.")
    prompt_lines.append("Type the column name exactly (or press Enter to accept default): ")
    prompt = "\n".join(prompt_lines)

    # Show available columns as short reference (first line only if many)
    print(prompt)
    sel = input("> ").strip()
    if sel == "" and default:
        sel = default

    # validate
    if sel not in df_columns:
        print(f" -> ERROR: '{sel}' is not a valid column name. Please choose from the list above.")
        # show columns again and re-ask once
        list_columns_notice(pd.DataFrame(columns=df_columns), "available columns")
        sel = input("Try again (enter column name exactly): ").strip()
        if sel == "" and default:
            sel = default
    if sel not in df_columns:
        raise KeyError(f"Invalid column selected for '{name}': '{sel}'")
    return sel

def auto_find_column(columns, keywords):
    """
    Return first column that contains any keyword in its name (case-insensitive).
    keywords: list of substrings to look for, ordered by priority.
    """
    cols_low = [c.lower() for c in columns]
    for kw in keywords:
        for c, low in zip(columns, cols_low):
            if kw.lower() in low:
                return c
    return None

# ---------------------------
# Dealer detection: flag by average kg per transaction
# ---------------------------
def detect_dealer_by_avg_tx(df_trans, mapping, avg_tx_percentile=0.95, min_avg_kg=None):
    """
    Flag dealers based on average kg per transaction (Avg_KG_per_Tx).
    - avg_tx_percentile: percentile threshold to use (e.g., 0.95 flags top 5% by avg).
    - min_avg_kg: optional absolute minimum average kg to flag (used in addition to percentile).
    Returns (flagged_df, summary_df).
    """
    trans = df_trans.copy()
    # rename using mapping
    dealer_col = mapping["transactions"]["dealer_id"]
    qty_col = mapping["transactions"]["quantity"]
    trans = trans.rename(columns={dealer_col: "DealerID", qty_col: "Quantity_KG"})

    # coerce numeric
    trans["Quantity_KG"] = pd.to_numeric(trans["Quantity_KG"], errors="coerce").fillna(0.0)

    # compute per-dealer metrics
    dealer_counts = trans["DealerID"].value_counts().rename("Transaction_Count")
    dealer_sums = trans.groupby("DealerID")["Quantity_KG"].sum().rename("Total_KG")
    summary = pd.concat([dealer_counts, dealer_sums], axis=1).fillna(0)
    summary = summary.reset_index().rename(columns={"index": "DealerID"})
    # avoid division by zero
    summary["Avg_KG_per_Tx"] = summary["Total_KG"] / summary["Transaction_Count"]
    summary["Avg_KG_per_Tx"] = summary["Avg_KG_per_Tx"].fillna(0.0)

    # diagnostics
    print("\n[Dealer diagnostics] Top 20 by average kg per transaction:")
    print(summary.sort_values("Avg_KG_per_Tx", ascending=False).head(20).to_string(index=False))
    print("\n[Dealer diagnostics] Top 20 by total kg sold:")
    print(summary.sort_values("Total_KG", ascending=False).head(20).to_string(index=False))
    print("\n[Dealer diagnostics] Top 20 by transaction count:")
    print(summary.sort_values("Transaction_Count", ascending=False).head(20).to_string(index=False))

    # percentile threshold and optional absolute floor
    avg_threshold = summary["Avg_KG_per_Tx"].quantile(avg_tx_percentile) if not summary.empty else 0.0
    if min_avg_kg is not None:
        avg_threshold = max(avg_threshold, float(min_avg_kg))

    print(f"\nUsing threshold -> Avg_KG_per_Tx >= {avg_threshold:.2f} (percentile={avg_tx_percentile}, min_avg_kg={min_avg_kg})")

    # flag only on average per tx
    summary["Flag_AvgTx"] = summary["Avg_KG_per_Tx"] >= avg_threshold
    flagged = summary[summary["Flag_AvgTx"]].sort_values(by="Avg_KG_per_Tx", ascending=False).reset_index(drop=True)

    return flagged, summary

# ---------------------------
# Load CSVs (with interactive path or default)
# ---------------------------
def load_csv_interactive():
    print("Enter the path to the farmers CSV (press Enter to use default './farmers_200_with_villageid.csv'):")
    farmers_path = input("> ").strip() or "./farmers_200_with_villageid.csv"
    print("Enter the path to the transactions CSV (press Enter to use default './transactions_1000_with_villageid.csv'):")
    transactions_path = input("> ").strip() or "./transactions_1000_with_villageid.csv"

    pf = Path(farmers_path)
    pt = Path(transactions_path)
    if not pf.exists():
        raise FileNotFoundError(f"Farmers file not found: {pf.resolve()}")
    if not pt.exists():
        raise FileNotFoundError(f"Transactions file not found: {pt.resolve()}")

    df_farmers = pd.read_csv(pf)
    df_trans = pd.read_csv(pt)
    return df_farmers, df_trans

# ---------------------------
# Interactive mapping + robust detections
# ---------------------------
def get_column_mapping(df_farmers, df_trans):
    # Show headers
    list_columns_notice(df_farmers, "FARMERS CSV")
    list_columns_notice(df_trans, "TRANSACTIONS CSV")

    fam_cols = list(df_farmers.columns)
    trans_cols = list(df_trans.columns)

    # Ask for FarmerID mapping (must be present in both)
    fam_id_default = auto_find_column(fam_cols, ["farmerid", "farmer_id", "id", "farmer"])
    trans_fid_default = auto_find_column(trans_cols, ["farmerid", "farmer_id", "id", "farmer"])

    print("\n-- Map Farmer ID (must exist in both files) --")
    fam_id_col = ask_column("FarmerID (farmers file)", fam_cols, [fam_id_default] if fam_id_default else None)
    trans_fid_col = ask_column("FarmerID (transactions file)", trans_cols, [trans_fid_default] if trans_fid_default else None)

    # Ask for Farmer name column (optional)
    name_default = auto_find_column(fam_cols, ["name", "farmer_name"])
    print("\n-- Map Farmer Name (optional) --")
    try:
        name_col = ask_column("Name (farmers file) [optional]", fam_cols, [name_default] if name_default else None)
    except KeyError:
        name_col = None

    # Land size column in farmers_df
    land_default = auto_find_column(fam_cols, ["land", "acre", "area"])
    print("\n-- Map Land Size (in acres) from farmers file --")
    land_col = ask_column("LandSize / area (farmers file)", fam_cols, [land_default] if land_default else None)

    # Transaction id, dealer id, quantity, village columns in transactions file
    trans_id_default = auto_find_column(trans_cols, ["transactionid", "txn", "tran_id", "transaction_id", "transaction"])
    dealer_default = auto_find_column(trans_cols, ["dealerid", "dealer_id", "dealer"])
    qty_default = auto_find_column(trans_cols, ["quantity", "quantity_kg", "qty", "kg"])
    village_default = auto_find_column(trans_cols, ["village", "villageid", "village_id", "village_name"])

    print("\n-- Map required Transaction-side columns (transactions file) --")
    trans_id_col = ask_column("TransactionID (transactions file)", trans_cols, [trans_id_default] if trans_id_default else None)
    dealer_col = ask_column("DealerID (transactions file)", trans_cols, [dealer_default] if dealer_default else None)
    qty_col = ask_column("Quantity (transactions file) (kg)", trans_cols, [qty_default] if qty_default else None)

    # Village is optional
    print("\n-- Map Village ID/Name (optional, transactions file) --")
    try:
        village_col = ask_column("VillageID/Name (transactions file) [optional]", trans_cols, [village_default] if village_default else None)
    except KeyError:
        village_col = None

    mapping = {
        "farmers": {
            "farmer_id": fam_id_col,
            "name": name_col,
            "land": land_col
        },
        "transactions": {
            "farmer_id": trans_fid_col,
            "transaction_id": trans_id_col,
            "dealer_id": dealer_col,
            "quantity": qty_col,
            "village": village_col
        }
    }
    print("\nColumn mapping established:")
    print(mapping)
    return mapping

# ---------------------------
# Fraud detection functions (use mapping)
# ---------------------------
def detect_land_mismatch_with_mapping(df_farmers, df_trans, mapping, limit_per_acre=100):
    # Copy to avoid side effects
    farmers = df_farmers.copy()
    trans = df_trans.copy()

    # Rename the selected columns to canonical names for easier code
    farmers = farmers.rename(columns={
        mapping["farmers"]["farmer_id"]: "FarmerID",
        mapping["farmers"]["land"]: "LandSize_Acres"
    })
    if mapping["farmers"]["name"]:
        farmers = farmers.rename(columns={mapping["farmers"]["name"]: "Name"})

    trans = trans.rename(columns={
        mapping["transactions"]["farmer_id"]: "FarmerID",
        mapping["transactions"]["transaction_id"]: "TransactionID",
        mapping["transactions"]["dealer_id"]: "DealerID",
        mapping["transactions"]["quantity"]: "Quantity_KG"
    })
    if mapping["transactions"]["village"]:
        trans = trans.rename(columns={mapping["transactions"]["village"]: "VillageID"})

    # Make types compatible for merge
    farmers["FarmerID"] = farmers["FarmerID"].astype(str)
    trans["FarmerID"] = trans["FarmerID"].astype(str)

    # Merge left so we keep all transactions
    merged = pd.merge(trans, farmers, on="FarmerID", how="left", suffixes=("_trans", "_farm"))

    # Coerce LandSize and Quantity to numeric
    merged["LandSize_Acres"] = pd.to_numeric(merged.get("LandSize_Acres"), errors="coerce")
    merged["Quantity_KG"] = pd.to_numeric(merged.get("Quantity_KG"), errors="coerce")

    # Warnings for missing data
    if merged["LandSize_Acres"].isna().any():
        print(f"Warning: {merged['LandSize_Acres'].isna().sum()} transactions have missing/invalid land size.")
    if merged["Quantity_KG"].isna().any():
        print(f"Warning: {merged['Quantity_KG'].isna().sum()} transactions have missing/invalid quantity.")
    if "Name" not in merged.columns:
        print("Note: Farmer 'Name' column not provided or not detected; results will show FarmerID instead.")

    merged["Max_Allowed_KG"] = merged["LandSize_Acres"] * float(limit_per_acre)

    suspicious = merged[merged["Quantity_KG"] > merged["Max_Allowed_KG"]].copy()
    # select nice output columns
    out_cols = ["TransactionID", "FarmerID"]
    if "Name" in suspicious.columns:
        out_cols.append("Name")
    out_cols += ["LandSize_Acres", "Quantity_KG", "Max_Allowed_KG", "DealerID"]
    if "VillageID" in suspicious.columns:
        out_cols.append("VillageID")

    # ensure columns exist
    out_cols = [c for c in out_cols if c in suspicious.columns]
    return suspicious[out_cols].reset_index(drop=True)

# ---------------------------
# Main interactive runner
# ---------------------------
def main():
    print("=" * 70)
    print("🔍 FERTILIZER FRAUD DETECTION - INTERACTIVE COLUMN MAPPING")
    print("=" * 70)

    # Load CSVs
    df_farmers, df_trans = load_csv_interactive()
    print(f"\n✅ Loaded {len(df_farmers)} farmers and {len(df_trans)} transactions")

    # Ask user to map columns (interactive)
    mapping = get_column_mapping(df_farmers, df_trans)

    # Run checks
    print("\n" + "=" * 70)
    print("🚨 CHECK 1: Suspicious Farmers (Over-buying)")
    print("=" * 70)
    suspicious_farmers = detect_land_mismatch_with_mapping(df_farmers, df_trans, mapping, limit_per_acre=100)
    if not suspicious_farmers.empty:
        print(f"\n⚠️  Found {len(suspicious_farmers)} suspicious transaction(s):")
        print(suspicious_farmers.to_string(index=False))
    else:
        print("\n✅ No suspicious farmers found!")

    print("\n" + "=" * 70)
    print("🚨 CHECK 2: Suspicious Dealers (High Average kg per Transaction)")
    print("=" * 70)

    # ---- changed behavior: flag by average kg per transaction only ----
    suspicious_dealers, dealer_summary = detect_dealer_by_avg_tx(df_trans, mapping,
                                                                 avg_tx_percentile=0.95,
                                                                 min_avg_kg=None)
    if not suspicious_dealers.empty:
        print(f"\n⚠️  Found {len(suspicious_dealers)} suspicious dealer(s) by average kg/tx:")
        print(suspicious_dealers.to_string(index=False))
    else:
        print("\n✅ No suspicious dealers found by average-kg-per-transaction threshold!")

    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Total Farmers: {len(df_farmers)}")
    print(f"Total Transactions: {len(df_trans)}")
    print(f"Suspicious Transactions: {len(suspicious_farmers)}")
    print(f"Suspicious Dealers: {len(suspicious_dealers)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
