"""Download the M5 Forecasting Accuracy data (Walmart, CC-BY-4.0) from Zenodo and extract what we use."""
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/api/records/12636070/files/m5-forecasting-accuracy.zip/content"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = ("calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv")


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = RAW / "m5-forecasting-accuracy.zip"
    if not zip_path.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW, members=list(FILES))
    for f in FILES:
        print(f"{f}: {(RAW / f).stat().st_size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
