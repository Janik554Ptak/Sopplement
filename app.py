import io
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from slugify import slugify


# ---------- Utilities ----------
def sanitize_sql_name(name: str, maxlen: int = 63) -> str:
    base = slugify(name, separator="_", lowercase=True)
    base = re.sub(r"[^a-z0-9_]", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    if not base or not re.match(r"^[a-z_]", base):
        base = f"t_{base}" if base else "t_import"
    return base[:maxlen]

def guess_separator(sample_bytes: bytes, default: str = ",") -> str:
    head = sample_bytes[:4096].decode(errors="ignore")
    candidates = [",", ";", "\t", "|"]
    counts = {sep: head.count(sep) for sep in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] >= 2 else default

def get_engine_url_from_env() -> str:
    load_dotenv()
    req = ["PGUSER", "PGPASSWORD", "PGHOST", "PGPORT", "PGDATABASE"]
    missing = [k for k in req if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing env vars in .env: {', '.join(missing)}")
    return (
        f"postgresql+psycopg2://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
        f"@{os.environ['PGHOST']}:{os.environ['PGPORT']}/{os.environ['PGDATABASE']}"
    )

def ensure_schema(engine, schema: str, owner: str):
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        try:
            conn.execute(text(f'ALTER SCHEMA "{schema}" OWNER TO "{owner}"'))
        except Exception:
            pass


# ---------- Type validation helpers ----------
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?((\d+\.\d*)|(\.\d+)|(\d+))([eE][+-]?\d+)?$")

def _classify_cell(value: str, parse_dates: bool) -> str:
    # treat blanks as None (ignored for "establishing" type, but never trigger a mismatch)
    if value is None:
        return "blank"
    v = str(value).strip()
    if v == "" or v.lower() in {"na", "nan", "null", "none"}:
        return "blank"

    # bool first (explicit true/false)
    if v.lower() in {"true", "false"}:
        return "bool"

    # integer
    if _INT_RE.match(v):
        return "int"

    # float (but not int)
    if _FLOAT_RE.match(v):
        return "float"

    # datetime (optional)
    if parse_dates:
        try:
            ts = pd.to_datetime(v, errors="raise", utc=False)
            # pandas may parse integers like "20210101" as dates; keep this branch
            return "datetime"
        except Exception:
            pass

    return "string"

def validate_homogeneous_column_types(
    data_bytes: bytes,
    sep: str,
    parse_dates: bool,
    chunksize: int
) -> Dict[str, str]:
    """
    Scan the CSV once (chunks, dtype=str) and ensure each column keeps a single data type.
    Returns a dict {column_name: expected_type} on success.
    Raises ValueError with a detailed message on first mismatch.
    """
    # fresh buffer
    buf = io.BytesIO(data_bytes)

    # read strictly as strings to see raw tokens
    reader = pd.read_csv(
        buf, sep=sep, dtype=str, chunksize=chunksize, encoding="utf-8", quotechar='"'
    )

    expected: Dict[str, Optional[str]] = {}
    # global data row counter (1-based including header -> header is row 1; first data row is 2)
    # We will report 1-based row numbers for user-friendliness.
    header_seen = False
    rows_seen = 0

    for chunk in reader:
        if not header_seen:
            # initialize expected map
            for col in chunk.columns:
                expected[col] = None
            header_seen = True

        # iterate rows; vectorize per column would be faster, but per-cell gives precise location
        for i in range(len(chunk)):
            rows_seen += 1  # counts data rows; header is considered before this loop
            # compute display row number (header row is 1)
            display_row = rows_seen + 1
            row = chunk.iloc[i]
            for col in chunk.columns:
                val = row[col]
                t = _classify_cell(val, parse_dates=parse_dates)
                if t == "blank":
                    continue
                if expected[col] is None:
                    expected[col] = t
                elif t != expected[col]:
                    # build helpful error
                    # keep the raw preview but truncate if massive
                    preview = str(val)
                    if len(preview) > 120:
                        preview = preview[:117] + "..."
                    raise ValueError(
                        f"Mixed data types detected in column '{col}': expected {expected[col]}, "
                        f"but found {t} at row {display_row} (value: '{preview}')."
                    )

    # fill any Nones (all blank columns) as 'string' by convention
    return {k: (v if v is not None else "string") for k, v in expected.items()}


# ---------- Streamlit UI ----------
st.set_page_config(page_title="CSV → PostgreSQL Loader", layout="wide")
st.title("CSV → PostgreSQL Loader")

with st.sidebar:
    st.header("Database connection")

    # Pre-fill from .env but allow overriding from the sidebar
    load_dotenv()
    host = st.text_input("Host", os.getenv("PGHOST", "localhost"))
    port = st.text_input("Port", os.getenv("PGPORT", "5432"))
    db   = st.text_input("Database", os.getenv("PGDATABASE", "appdb"))
    user = st.text_input("User", os.getenv("PGUSER", "primary"))
    pwd  = st.text_input("Password", os.getenv("PGPASSWORD", "puding"), type="password")
    schema = st.text_input("Target schema", os.getenv("PGSCHEMA", "raw"))

    st.divider()
    st.header("Import options")
    if_exists = st.selectbox("If table exists", ["replace", "append", "fail"], index=0)
    sep_choice = st.selectbox("Delimiter", ["auto", ",", ";", "\\t", "|"], index=0)
    parse_dates = st.checkbox("Heuristically parse date/time columns", value=False)
    chunksize = st.number_input("Chunk size (rows)", min_value=10_000, max_value=1_000_000, value=50_000, step=10_000)

    connect_clicked = st.button("Test connection")

    # Build a URL from the sidebar values
    engine_url = (
        f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
    )

    if connect_clicked:
        try:
            engine = create_engine(engine_url, pool_pre_ping=True, future=True)
            ensure_schema(engine, schema, user)
            with engine.begin() as conn:
                conn.execute(text("SELECT 1"))
            st.success("Connected to PostgreSQL and schema is ready.")
        except Exception as e:
            st.error(f"Connection failed: {e}")

st.subheader("Drag & drop your CSV files")
uploaded: List[io.BytesIO] = st.file_uploader(
    "Drop multiple CSVs here", type=["csv"], accept_multiple_files=True
)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.write("**Files queued:**")
    if uploaded:
        for f in uploaded:
            st.write(f"- {f.name} ({f.size} bytes)")
    else:
        st.info("No files yet. Drag and drop CSV files above, or click to browse.")

with col_right:
    start_import = st.button("Upload & Import", use_container_width=True)

# ---------- Import logic ----------
def import_one_streamlit_file(st_file, engine, schema: Optional[str], if_exists: str,
                             parse_dates: bool, chunksize: int, sep_choice: str):
    # Read bytes to allow autodetect on head
    data_bytes = st_file.getvalue()
    # pick delimiter
    sep = guess_separator(data_bytes) if sep_choice == "auto" else ("\t" if sep_choice == "\\t" else sep_choice)

    # -------- NEW: Validate column homogeneity before import --------
    try:
        expected_types = validate_homogeneous_column_types(
            data_bytes=data_bytes,
            sep=sep,
            parse_dates=parse_dates,
            chunksize=chunksize
        )
        # You could log or display expected_types if helpful
    except ValueError as ve:
        # Surface the precise error; abort this file
        raise RuntimeError(f"Type validation failed: {ve}")

    # Reset a fresh buffer for pandas for the actual import
    buf = io.BytesIO(data_bytes)

    # Produce a safe table name from filename
    tbl_name = sanitize_sql_name(Path(st_file.name).stem)
    fq = f"{schema}.{tbl_name}" if schema else tbl_name

    # Stream in chunks for large files (keep your original import path)
    reader = pd.read_csv(
        buf, sep=sep, dtype_backend="pyarrow", chunksize=chunksize, encoding="utf-8", quotechar='"'
    )

    first = True
    total = 0
    for chunk in reader:
        # sanitize column names
        chunk.columns = [sanitize_sql_name(c) for c in chunk.columns]

        if parse_dates:
            # simple heuristic: try parsing cols that look like datetimes
            maybe_dt = [c for c in chunk.columns if re.search(r"(date|time|timestamp|dt)$", c)]
            for c in maybe_dt:
                try:
                    chunk[c] = pd.to_datetime(chunk[c], errors="ignore", utc=False)
                except Exception:
                    pass

        chunk.to_sql(
            name=tbl_name,
            con=engine,
            schema=schema if schema else None,
            if_exists="replace" if first and if_exists in ("replace", "fail") else "append",
            index=False,
            method="multi",
            chunksize=chunksize,
        )
        total += len(chunk)
        first = False
    return fq, total, sep

if start_import:
    if not uploaded:
        st.warning("Please add at least one CSV file.")
    else:
        try:
            engine = create_engine(engine_url, pool_pre_ping=True, future=True)
            ensure_schema(engine, schema, user)
        except Exception as e:
            st.error(f"Could not connect to PostgreSQL: {e}")
            st.stop()

        success, failures = [], []
        with st.status("Importing files...", expanded=True) as status:
            for st_file in uploaded:
                st.write(f"Working on **{st_file.name}** …")
                try:
                    fq, rows, sep_used = import_one_streamlit_file(
                        st_file, engine, schema, if_exists, parse_dates, chunksize, sep_choice
                    )
                    st.write(f"Imported `{st_file.name}` → **{fq}** (rows: {rows}, sep: '{sep_used}')")
                    success.append((st_file.name, fq, rows))
                except Exception as e:
                    st.write(f"`{st_file.name}` failed: {e}")
                    failures.append((st_file.name, str(e)))
            if failures:
                status.update(label=f"Completed with errors ({len(success)} success, {len(failures)} failed)", state="error")
            else:
                status.update(label=f"All files imported ({len(success)} success)", state="complete")

        if success:
            st.success("Done.")
        if failures:
            st.error("Some files failed. See log above.")
