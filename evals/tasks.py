"""Benchmark: seeded dirty datasets + questions with ground-truth answers.

Ground truth is computed from the *clean* frame before corruption is applied, so a task
is only passed when the agent correctly undoes the dirt. Each task also ships a reference
solution (plain pandas on the dirty files); tests run it through the real sandbox to prove
every expected answer is reachable and that the benchmark itself is correct.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

UNANSWERABLE = "UNANSWERABLE"


@dataclass
class Task:
    id: str
    dataset: str
    question: str
    expected: str | float | list[str]
    kind: str  # number | label | list | unanswerable
    reference_code: str
    tolerance: float = 0.01
    traps: list[str] = field(default_factory=list)
    data_dir: Path = Path(".")

    def check(self, value: str | None) -> bool:
        if value is None:
            return False
        value = value.strip()
        if self.kind == "unanswerable":
            return UNANSWERABLE in value.upper()
        if self.kind == "number":
            parsed = parse_number(value)
            return parsed is not None and abs(parsed - float(self.expected)) <= self.tolerance
        if self.kind == "label":
            return _norm(value) == _norm(str(self.expected))
        if self.kind == "list":
            return [_norm(v) for v in value.split(",")] == [_norm(v) for v in self.expected]
        raise ValueError(self.kind)


def _norm(text: str) -> str:
    return text.strip().strip("'\"").strip().casefold()


def parse_number(text: str) -> float | None:
    cleaned = re.sub(r"[€$£%\s,]", "", text)
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return None if math.isnan(number) else number


def _assert_no_tie(series: pd.Series, top: int = 1) -> None:
    ordered = series.sort_values(ascending=False).to_numpy()
    if len(ordered) > top and ordered[top - 1] == ordered[top]:
        raise ValueError("benchmark seed produced an ambiguous tie; pick another seed")


# --------------------------------------------------------------------------- orders

_PRODUCTS = {"Widget": 12.5, "Gadget": 49.9, "Doohickey": 8.75, "Gizmo": 120.0,
             "Thingamajig": 33.3, "Whatsit": 5.2, "Contraption": 210.0, "Sprocket": 18.0}
_COUNTRY_SPELLINGS = {
    "FR": ["FR", "France", "france ", "FRANCE"], "DE": ["DE", "Germany", "germany"],
    "ES": ["ES", "Spain"], "IT": ["IT", "Italy", "italy "], "GB": ["GB", "United Kingdom", "UK"],
}
_ORDERS_PRELUDE = """import pandas as pd
df = pd.read_csv('data/orders.csv').drop_duplicates('order_id')
df['amt'] = df['amount'].str.replace('€', '').str.replace(',', '').astype(float)
cmap = {'france': 'FR', 'fr': 'FR', 'germany': 'DE', 'de': 'DE', 'spain': 'ES', 'es': 'ES',
        'italy': 'IT', 'it': 'IT', 'united kingdom': 'GB', 'uk': 'GB', 'gb': 'GB'}
df['cc'] = df['country'].str.strip().str.lower().map(cmap)
df['date'] = pd.to_datetime(df['order_date'], format='mixed')
"""


def _orders(root: Path, rng: np.random.Generator, n: int = 400) -> list[Task]:
    names = list(_PRODUCTS)
    clean = pd.DataFrame({
        "order_id": [f"O-{i:04d}" for i in range(1, n + 1)],
        "date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 181, n), unit="D"),
        "cc": rng.choice(list(_COUNTRY_SPELLINGS), n, p=[0.3, 0.25, 0.15, 0.15, 0.15]),
        "product": rng.choice(names, n),
        "quantity": rng.integers(1, 6, n),
        "customer_id": [f"C-{c:03d}" for c in rng.integers(1, 120, n)],
    })
    amount = np.round(clean["quantity"] * clean["product"].map(_PRODUCTS), 2)
    clean["amount"] = np.where(rng.random(n) < 0.06, -amount, amount)

    raw = pd.DataFrame({
        "order_id": clean["order_id"],
        "order_date": [d.strftime("%Y-%m-%d") if rng.random() < 0.5 else d.strftime("%d %b %Y")
                       for d in clean["date"]],
        "country": [rng.choice(_COUNTRY_SPELLINGS[c]) for c in clean["cc"]],
        "customer_id": clean["customer_id"],
        "product": clean["product"],
        "quantity": clean["quantity"],
        "amount": [f"-€{abs(a):,.2f}" if a < 0 else f"€{a:,.2f}" for a in clean["amount"]],
    })
    raw = pd.concat([raw, raw.sample(25, random_state=int(rng.integers(1 << 30)))])
    raw = raw.sample(frac=1, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    folder = root / "orders"
    folder.mkdir(parents=True, exist_ok=True)
    raw.to_csv(folder / "orders.csv", index=False)

    by_country = clean.groupby("cc")["amount"].sum()
    by_product = clean.groupby("product")["quantity"].sum()
    by_month = clean.groupby(clean["date"].dt.strftime("%Y-%m"))["amount"].sum()
    _assert_no_tie(by_country)
    _assert_no_tie(by_product, top=3)
    _assert_no_tie(by_month)
    common = ["duplicate order_id rows", "currency strings", "country spelling variants"]
    return [
        Task("orders_revenue", "orders",
             "What is the total revenue (sum of the amount column, in EUR, refunds counting as "
             "negative amounts) after removing duplicate order_id rows?",
             round(float(clean["amount"].sum()), 2), "number",
             _ORDERS_PRELUDE + "print(f\"RESULT={df['amt'].sum():.2f}\")",
             traps=["duplicate order_id rows", "currency strings"]),
        Task("orders_top_country", "orders",
             "Which country has the highest total revenue (after removing duplicate order_id "
             "rows)? Answer with its ISO 3166-1 alpha-2 code.",
             by_country.idxmax(), "label",
             _ORDERS_PRELUDE + "print('RESULT=' + df.groupby('cc')['amt'].sum().idxmax())",
             traps=common),
        Task("orders_top3_products", "orders",
             "Which 3 products have the highest total quantity (all rows, after removing duplicate "
             "order_id rows)? List them in descending order, comma-separated.",
             list(by_product.sort_values(ascending=False).index[:3]), "list",
             _ORDERS_PRELUDE + "print('RESULT=' + ','.join(df.groupby('product')['quantity']"
             ".sum().sort_values(ascending=False).index[:3]))",
             traps=["duplicate order_id rows"]),
        Task("orders_peak_month", "orders",
             "Which calendar month (format YYYY-MM) had the highest total revenue, after removing "
             "duplicate order_id rows?",
             by_month.idxmax(), "label",
             _ORDERS_PRELUDE + "print('RESULT=' + df.groupby(df['date'].dt.strftime('%Y-%m'))"
             "['amt'].sum().idxmax())",
             traps=["duplicate order_id rows", "mixed date formats", "currency strings"]),
        Task("orders_refund_rate", "orders",
             "What percentage of orders are refunds (negative amount), after removing duplicate "
             "order_id rows? Round to 1 decimal.",
             round(float((clean["amount"] < 0).mean() * 100), 1), "number",
             _ORDERS_PRELUDE + "print(f\"RESULT={(df['amt'] < 0).mean() * 100:.1f}\")",
             tolerance=0.05, traps=["duplicate order_id rows", "currency strings"]),
        Task("orders_aov_france", "orders",
             "What is the average order value (mean of amount, refunds included) for orders from "
             "France, after removing duplicate order_id rows? Round to 2 decimals.",
             round(float(clean.loc[clean["cc"] == "FR", "amount"].mean()), 2), "number",
             _ORDERS_PRELUDE + "print(f\"RESULT={df.loc[df['cc'] == 'FR', 'amt'].mean():.2f}\")",
             traps=common),
        Task("orders_unanswerable", "orders",
             "What is the average customer satisfaction score across all orders?",
             UNANSWERABLE, "unanswerable",
             "import pandas as pd\n"
             "assert 'satisfaction' not in ' '.join(pd.read_csv('data/orders.csv').columns)\n"
             "print('RESULT=UNANSWERABLE')",
             traps=["the requested column does not exist"]),
    ]


# --------------------------------------------------------------------------- sensors

def _sensors(root: Path, rng: np.random.Generator) -> list[Task]:
    stations = ["North", "South", "East"]
    hours = pd.date_range("2025-06-01", periods=24 * 20, freq="h")
    frames = []
    for station, offset in zip(stations, (0.0, 2.5, -1.5)):
        temp = 20 + offset + 6 * np.sin((hours.hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 1.5, len(hours))
        humidity = 70 - 1.5 * (temp - 20) + rng.normal(0, 4, len(hours))
        frames.append(pd.DataFrame({
            "timestamp": hours, "station": station,
            "temp_c": np.round(temp, 1), "humidity_pct": np.round(np.clip(humidity, 5, 100), 1),
        }))
    clean = pd.concat(frames).sort_values(["timestamp", "station"]).reset_index(drop=True)

    missing_temp = np.zeros(len(clean), dtype=bool)
    for station, rate in zip(stations, (0.08, 0.03, 0.015)):
        idx = np.flatnonzero(clean["station"].to_numpy() == station)
        missing_temp[rng.choice(idx, int(len(idx) * rate), replace=False)] = True
    missing_hum = rng.random(len(clean)) < 0.03

    raw = clean.copy()
    raw["temp_c"] = raw["temp_c"].astype(object)
    raw["humidity_pct"] = raw["humidity_pct"].astype(object)
    raw.loc[missing_temp, "temp_c"] = -999
    raw.loc[missing_hum, "humidity_pct"] = "N/A"
    folder = root / "sensors"
    folder.mkdir(parents=True, exist_ok=True)
    raw.to_csv(folder / "sensors.csv", index=False)

    valid_temp = clean["temp_c"].where(~missing_temp)
    valid_hum = clean["humidity_pct"].where(~missing_hum)
    both = valid_temp.notna() & valid_hum.notna()
    missing_by_station = pd.Series(missing_temp).groupby(clean["station"]).sum()
    _assert_no_tie(missing_by_station)
    prelude = ("import pandas as pd\ndf = pd.read_csv('data/sensors.csv', na_values=['N/A'])\n"
               "df['temp'] = df['temp_c'].where(df['temp_c'] != -999)\n")
    return [
        Task("sensors_mean_temp", "sensors",
             "What is the mean temperature (temp_c, in degrees C) across all stations, excluding "
             "missing readings? Round to 2 decimals.",
             round(float(valid_temp.mean()), 2), "number",
             prelude + "print(f\"RESULT={df['temp'].mean():.2f}\")",
             traps=["-999 sentinel for missing temperature"]),
        Task("sensors_hot_readings", "sensors",
             "How many readings have a temperature strictly above 25 degrees C, excluding missing "
             "readings?",
             int((valid_temp > 25).sum()), "number",
             prelude + "print(f\"RESULT={int((df['temp'] > 25).sum())}\")",
             tolerance=0.0, traps=["-999 sentinel for missing temperature"]),
        Task("sensors_correlation", "sensors",
             "What is the Pearson correlation between temperature and humidity, using only rows "
             "where both are valid? Round to 3 decimals.",
             round(float(np.corrcoef(valid_temp[both], valid_hum[both])[0, 1]), 3), "number",
             prelude + "d = df.dropna(subset=['temp', 'humidity_pct'])\n"
             "print(f\"RESULT={d['temp'].corr(d['humidity_pct']):.3f}\")",
             tolerance=0.002, traps=["-999 sentinel", "N/A strings in a numeric column"]),
        Task("sensors_most_missing_station", "sensors",
             "Which station has the most missing temperature readings (-999)?",
             missing_by_station.idxmax(), "label",
             prelude + "print('RESULT=' + df[df['temp_c'] == -999]['station'].value_counts().idxmax())",
             traps=["-999 sentinel for missing temperature"]),
    ]


# --------------------------------------------------------------------------- hr

_DEPARTMENTS = pd.DataFrame({
    "dept_id": ["D01", "D02", "D03", "D04", "D05"],
    "dept_name": ["Engineering", "Research", "Sales", "Support", "Finance"],
    "location": ["Paris", "Berlin", "London", "Madrid", "Paris"],
})
_BASE_SALARY = {"D01": 72000, "D02": 76000, "D03": 55000, "D04": 42000, "D05": 64000}


def _hr(root: Path, rng: np.random.Generator, n: int = 150) -> list[Task]:
    dept = rng.choice(list(_BASE_SALARY) + ["D99"], n, p=[0.25, 0.2, 0.2, 0.15, 0.15, 0.05])
    salary = np.array([_BASE_SALARY.get(d, 50000) for d in dept]) + rng.normal(0, 8000, n)
    clean = pd.DataFrame({
        "emp_id": [f"E{i:03d}" for i in range(1, n + 1)],
        "dept_id": dept,
        "salary": np.round(salary).astype(float),
        "hire_date": pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 365 * 6, n), unit="D"),
    })
    clean.loc[rng.random(n) < 0.04, "salary"] = np.nan

    raw = clean.copy()
    raw["hire_date"] = [d.strftime("%Y-%m-%d") if rng.random() < 0.5 else d.strftime("%d %b %Y")
                        for d in clean["hire_date"]]
    folder = root / "hr"
    folder.mkdir(parents=True, exist_ok=True)
    raw.to_csv(folder / "employees.csv", index=False)
    _DEPARTMENTS.to_csv(folder / "departments.csv", index=False)

    joined = clean.merge(_DEPARTMENTS, on="dept_id")
    by_location = joined.groupby("location")["salary"].mean()
    _assert_no_tie(by_location)
    research = joined.loc[joined["dept_name"] == "Research", "salary"]
    return [
        Task("hr_top_location", "hr",
             "Which office location has the highest average salary? Employees whose department is "
             "not listed in departments.csv should be ignored.",
             by_location.idxmax(), "label",
             "import pandas as pd\nemp = pd.read_csv('data/employees.csv')\n"
             "dep = pd.read_csv('data/departments.csv')\nm = emp.merge(dep, on='dept_id')\n"
             "print('RESULT=' + m.groupby('location')['salary'].mean().idxmax())",
             traps=["join across two files", "orphan dept_id", "missing salaries"]),
        Task("hr_median_research", "hr",
             "What is the median salary in the Research department (ignoring missing salaries)?",
             round(float(research.median()), 2), "number",
             "import pandas as pd\nemp = pd.read_csv('data/employees.csv')\n"
             "dep = pd.read_csv('data/departments.csv')\nm = emp.merge(dep, on='dept_id')\n"
             "print(f\"RESULT={m.loc[m['dept_name'] == 'Research', 'salary'].median():.2f}\")",
             traps=["join across two files", "missing salaries"]),
        Task("hr_hired_2022", "hr",
             "How many employees in employees.csv were hired in 2022?",
             int((clean["hire_date"].dt.year == 2022).sum()), "number",
             "import pandas as pd\nemp = pd.read_csv('data/employees.csv')\n"
             "print(f\"RESULT={int((pd.to_datetime(emp['hire_date'], format='mixed').dt.year == 2022).sum())}\")",
             tolerance=0.0, traps=["mixed date formats"]),
    ]


def generate(root: Path, seed: int = 11) -> list[Task]:
    """Write all datasets under `root` and return the tasks.

    Deterministic for a given seed. If a seed yields an ambiguous answer (a tie), the next
    seed is tried, so the same input seed always produces the same benchmark.
    """
    root = Path(root)
    for attempt in range(100):
        rng = np.random.default_rng(seed + attempt)
        try:
            tasks = _orders(root, rng) + _sensors(root, rng) + _hr(root, rng)
        except ValueError:
            continue
        for task in tasks:
            task.data_dir = root / task.dataset
        return tasks
    raise RuntimeError("could not generate an unambiguous benchmark")
