import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        # NOTE: 'quote_requests.csv' was not included in the repository. If it is
        # missing, build a stand-in table (one row per historical quote) whose
        # 'response' text is derived from the quote metadata, so that
        # search_quote_history() can still join the two tables and match on
        # job type / event type / quote explanation.
        if os.path.exists("quote_requests.csv"):
            quote_requests_df = pd.read_csv("quote_requests.csv")
        else:
            _quotes_meta = pd.read_csv("quotes.csv")["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else {}
            )
            quote_requests_df = pd.DataFrame({
                "job": _quotes_meta.apply(lambda m: m.get("job_type", "")),
                "need_size": _quotes_meta.apply(lambda m: m.get("order_size", "")),
                "event": _quotes_meta.apply(lambda m: m.get("event_type", "")),
                "response": _quotes_meta.apply(
                    lambda m: f"{m.get('order_size', '')} order from a {m.get('job_type', '')} "
                              f"for a {m.get('event_type', '')}"
                ),
            })
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################
#
# Architecture overview (see workflow_diagram.png and REPORT.md)
# ---------------------------------------------------------------
# Framework: pydantic-ai
# Agents (4 in total, limit is 5):
#   1. Orchestrator Agent  - reads the customer request, delegates to the
#                            workers below and writes the customer reply.
#   2. Inventory Agent     - matches requested items to the catalog, checks
#                            stock, decides on supplier restocking.
#   3. Quoting Agent       - prices the order, applies bulk discounts and
#                            compares with historical quotes.
#   4. Sales Agent         - runs a financial health check and books the
#                            sale in the database.
#
# Data is passed between agents through a shared per-request "blackboard"
# (the OrderContext dataclass, injected as pydantic-ai `deps`). Agents decide
# WHICH tools to call and WHEN; the tools hold the business rules so numbers
# written to the database are always deterministic and auditable.
#
# Run modes
#   * LLM mode     (default when UDACITY_OPENAI_API_KEY is set): every agent is
#                  backed by an OpenAI model through the Vocareum proxy.
#   * Offline mode (no API key, or BEAVER_OFFLINE=1): every agent is backed by
#                  a pydantic-ai FunctionModel running a scripted, rule-based
#                  policy. The SAME agents, tools and database code are used,
#                  so the system can be evaluated without network access.

import re
import json
import difflib
from dataclasses import dataclass, field
from typing import Optional

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# Set up and load your env parameters and instantiate your model.
dotenv.load_dotenv()

OPENAI_BASE_URL = "https://openai.vocareum.com/v1"
OPENAI_MODEL_NAME = os.getenv("BEAVER_MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("UDACITY_OPENAI_API_KEY")
OFFLINE_MODE = os.getenv("BEAVER_OFFLINE") == "1" or not OPENAI_API_KEY

# ---------------------------------------------------------------------------
# Business policy constants
# ---------------------------------------------------------------------------
SHEETS_PER_REAM = 500            # a ream of paper is 500 sheets
MINIMUM_CASH_RESERVE = 2000.00   # never let a supplier purchase push cash below this
LARGE_ORDER_THRESHOLD = 500.00   # orders above this trigger a full financial health check
RESTOCK_TARGET_MULTIPLIER = 2    # after a sale, carried items are topped up to 2x min_stock_level
SUPPLIER_COST_RATIO = 0.70       # we buy from the supplier at 70% of our list price (30% gross margin);
                                 # the largest bulk discount (15%) therefore never sells below cost

# Bulk discount tiers: (minimum total units in the order, discount rate, label)
BULK_DISCOUNT_TIERS = [
    (20000, 0.15, "15% volume discount for orders of 20,000+ units"),
    (5000, 0.10, "10% bulk discount for orders of 5,000+ units"),
    (1000, 0.05, "5% bulk discount for orders of 1,000+ units"),
]

# Catalog lookup: item_name -> catalog record (from the starter's paper_supplies)
CATALOG = {item["item_name"]: item for item in paper_supplies}


# ---------------------------------------------------------------------------
# Shared blackboard passed to all agents as pydantic-ai `deps`
# ---------------------------------------------------------------------------
@dataclass
class LineItem:
    """One requested product line and everything the agents learn about it."""
    description: str                     # customer wording, e.g. "A4 glossy paper"
    quantity: int                        # quantity in catalog units (sheets/units)
    catalog_item: Optional[str] = None   # matched catalog item name
    status: str = "pending"              # pending | available | restock | unavailable
    reason: str = ""                     # customer-safe explanation
    stock_on_hand: int = 0
    restock_quantity: int = 0
    restock_eta: Optional[str] = None
    restock_placed: bool = False
    delivery_date: Optional[str] = None
    unit_price: float = 0.0
    line_total: float = 0.0
    sold: bool = False


@dataclass
class OrderContext:
    """Per-request state shared by the orchestrator and the worker agents."""
    request_text: str
    request_date: str
    deadline: Optional[str] = None
    line_items: List[LineItem] = field(default_factory=list)
    quote_subtotal: float = 0.0
    discount_rate: float = 0.0
    discount_label: str = ""
    quote_total: float = 0.0
    history_note: str = ""
    sale_completed: bool = False
    sale_note: str = ""
    agent_log: List[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        """Keep an internal audit trail (never shown to the customer)."""
        self.agent_log.append(message)


# ---------------------------------------------------------------------------
# Request parsing and catalog matching helpers (pure functions)
# ---------------------------------------------------------------------------
_NUMBER = r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)"
_UNIT = r"(sheets?|reams?|rolls?|packets?|packs?|boxes?|units?|pieces?|pads?)"
_TERMINATOR = (
    r"(?=,(?=\s*(?:and\s+)?\d)|,\s*and\b|;|\n|\.(?:\s|$)|\s+and\s+\d|"
    r"\s+for\s+(?:our|the|an?|my|your)\b|\s+in\s+time\b|\s+along\s+with\b|$)"
)
LINE_ITEM_PATTERN = re.compile(
    _NUMBER + r"\s+(?:" + _UNIT + r"\s+(?:of\s+)?)?(.+?)" + _TERMINATOR,
    re.IGNORECASE,
)
DEADLINE_PATTERN = re.compile(r"\bby\s+([A-Z][a-z]+\s+\d{1,2},\s*\d{4})")
REQUEST_DATE_PATTERN = re.compile(r"Date of request:\s*(\d{4}-\d{2}-\d{2})")

# Ordered keyword rules: the first rule whose keywords all appear wins.
# Product-specific words come first, generic paper words last.
CATALOG_KEYWORD_RULES = [
    (["washi"], "Decorative adhesive tape (washi tape)"),
    (["tape"], "Decorative adhesive tape (washi tape)"),
    (["streamer"], "Party streamers"),
    (["napkin"], "Paper napkins"),
    (["plate"], "Paper plates"),
    (["disposable", "cup"], "Disposable cups"),
    (["cup"], "Paper cups"),
    (["envelope"], "Envelopes"),
    (["sticky"], "Sticky notes"),
    (["notepad"], "Notepads"),
    (["invitation"], "Invitation cards"),
    (["flyer"], "Flyers"),
    (["table cover"], "Table covers"),
    (["tablecloth"], "Table covers"),
    (["folder"], "Presentation folders"),
    (["name tag"], "Name tags with lanyards"),
    (["party bag"], "Paper party bags"),
    (["poster board"], "Large poster paper (24x36 inches)"),
    (["24\" x 36"], "Large poster paper (24x36 inches)"),
    (["24x36"], "Large poster paper (24x36 inches)"),
    (["banner", "roll"], "Rolls of banner paper (36-inch width)"),
    (["banner"], "Banner paper"),
    (["100 lb"], "100 lb cover stock"),
    (["cover stock"], "100 lb cover stock"),
    (["80 lb"], "80 lb text paper"),
    (["250 gsm"], "250 gsm cardstock"),
    (["220 gsm"], "220 gsm poster paper"),
    (["poster paper"], "Poster paper"),
    (["cardstock"], "Cardstock"),
    (["card stock"], "Cardstock"),
    (["heavyweight"], "Heavyweight paper"),
    (["glossy"], "Glossy paper"),
    (["matte"], "Matte paper"),
    (["photo"], "Photo paper"),
    (["construction"], "Construction paper"),
    (["kraft"], "Kraft paper"),
    (["butcher"], "Butcher paper"),
    (["crepe"], "Crepe paper"),
    (["wrapping"], "Wrapping paper"),
    (["glitter"], "Glitter paper"),
    (["decorative paper"], "Decorative paper"),
    (["letterhead"], "Letterhead paper"),
    (["legal"], "Legal-size paper"),
    (["patterned"], "Patterned paper"),
    (["uncoated"], "Uncoated paper"),
    (["bright"], "Bright-colored paper"),
    (["colored"], "Colored paper"),
    (["colorful"], "Colored paper"),
    (["coloured"], "Colored paper"),
    (["recycled"], "Recycled paper"),
    (["eco"], "Eco-friendly paper"),
    (["letter"], "Letter-sized paper"),
    (["a4"], "A4 paper"),
    (["printer"], "Standard copy paper"),
    (["printing"], "Standard copy paper"),
    (["copy"], "Standard copy paper"),
    (["white paper"], "Standard copy paper"),
]

# Paper formats we do not sell at all (checked before keyword rules).
UNSUPPORTED_FORMATS = ["a3", "a5"]


def parse_customer_request(request_text: str) -> Dict:
    """Extract line items, deadline and request date from free-text request.

    Quantities given in reams are converted to sheets. Returns a dict with keys
    'items' (list of {description, quantity}), 'deadline' and 'request_date'.
    """
    items = []
    for match in LINE_ITEM_PATTERN.finditer(request_text):
        number, unit, description = match.groups()
        description = description.strip(" .-:\"'")
        if not re.search(r"[A-Za-z]{3,}", description):
            continue  # e.g. stray numbers from dates
        quantity = int(number.replace(",", ""))
        if unit and unit.lower().startswith("ream"):
            quantity *= SHEETS_PER_REAM
            description = f"{description} ({number} reams)"
        items.append({"description": description, "quantity": quantity})

    deadline = None
    deadline_match = DEADLINE_PATTERN.search(request_text)
    if deadline_match:
        try:
            deadline = datetime.strptime(
                deadline_match.group(1).replace(",", ", ").replace("  ", " "), "%B %d, %Y"
            ).strftime("%Y-%m-%d")
        except ValueError:
            deadline = None

    date_match = REQUEST_DATE_PATTERN.search(request_text)
    request_date = date_match.group(1) if date_match else datetime.now().strftime("%Y-%m-%d")
    return {"items": items, "deadline": deadline, "request_date": request_date}


def match_catalog_item(description: str) -> Optional[str]:
    """Map a customer description (e.g. 'A4 glossy paper') to a catalog item name.

    Returns None when the item is not something Beaver's Choice sells.
    """
    text_lower = description.lower()
    if any(re.search(rf"\b{fmt}\b", text_lower) for fmt in UNSUPPORTED_FORMATS):
        return None
    for keywords, item_name in CATALOG_KEYWORD_RULES:
        if all(keyword in text_lower for keyword in keywords):
            return item_name
    # Fuzzy fallback for close spellings ("Kraft papers", "Cardstocks" ...)
    close = difflib.get_close_matches(description.title(), list(CATALOG), n=1, cutoff=0.8)
    return close[0] if close else None


def get_min_stock_level(item_name: str) -> int:
    """Return the configured minimum stock level (0 for items we do not normally carry)."""
    result = pd.read_sql(
        "SELECT min_stock_level FROM inventory WHERE item_name = :item_name",
        db_engine,
        params={"item_name": item_name},
    )
    return int(result.iloc[0]["min_stock_level"]) if not result.empty else 0


def get_unit_price(item_name: str) -> float:
    """List price per unit, taken from the inventory table or the full catalog."""
    result = pd.read_sql(
        "SELECT unit_price FROM inventory WHERE item_name = :item_name",
        db_engine,
        params={"item_name": item_name},
    )
    if not result.empty:
        return float(result.iloc[0]["unit_price"])
    return float(CATALOG[item_name]["unit_price"])


def get_supplier_unit_cost(item_name: str) -> float:
    """Wholesale cost per unit when restocking from the supplier (internal only)."""
    return round(get_unit_price(item_name) * SUPPLIER_COST_RATIO, 4)


def select_bulk_discount(total_units: int) -> tuple:
    """Return (rate, label) of the best bulk discount tier for this many units."""
    for minimum_units, rate, label in BULK_DISCOUNT_TIERS:
        if total_units >= minimum_units:
            return rate, label
    return 0.0, ""


def format_date(iso_date: Optional[str]) -> str:
    """Render an ISO date as e.g. 'April 15, 2025' for customer messages."""
    if not iso_date:
        return "as soon as possible"
    return datetime.strptime(iso_date[:10], "%Y-%m-%d").strftime("%B %d, %Y").replace(" 0", " ")


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent
def tool_get_inventory_snapshot(ctx: RunContext[OrderContext]) -> str:
    """Return every item currently in stock with its quantity (uses get_all_inventory)."""
    snapshot = get_all_inventory(ctx.deps.request_date)
    ctx.deps.log(f"inventory snapshot: {len(snapshot)} items in stock")
    return json.dumps({name: int(qty) for name, qty in snapshot.items()})


def tool_check_stock_level(ctx: RunContext[OrderContext], item_name: str) -> str:
    """Return the current stock of one catalog item as of the request date (uses get_stock_level)."""
    stock_df = get_stock_level(item_name, ctx.deps.request_date)
    return f"{item_name}: {int(stock_df['current_stock'].iloc[0])} units in stock"


def tool_check_supplier_delivery(ctx: RunContext[OrderContext], quantity: int) -> str:
    """Return when the supplier could deliver `quantity` units if ordered today (uses get_supplier_delivery_date)."""
    return get_supplier_delivery_date(ctx.deps.request_date, quantity)


def tool_assess_line_item(ctx: RunContext[OrderContext], line_index: int) -> str:
    """Assess one requested line: match it to the catalog, check stock, and if
    stock is short decide whether a supplier restock can arrive before the
    customer's deadline and can be paid for. Records the decision on the order.

    Returns a status string starting with AVAILABLE, RESTOCK_REQUIRED or UNAVAILABLE.
    """
    order = ctx.deps
    line = order.line_items[line_index]
    line.catalog_item = match_catalog_item(line.description)
    if line.catalog_item is None:
        line.status = "unavailable"
        line.reason = f"'{line.description}' is not part of our product catalog"
        return f"UNAVAILABLE: {line.reason}"

    stock = int(get_stock_level(line.catalog_item, order.request_date)["current_stock"].iloc[0])
    line.stock_on_hand = stock
    min_level = get_min_stock_level(line.catalog_item)
    stock_after_sale = stock - line.quantity

    # Reorder policy: cover any shortfall and, for carried items, top the shelf
    # back up to RESTOCK_TARGET_MULTIPLIER x min_stock_level after the sale.
    target_level = RESTOCK_TARGET_MULTIPLIER * min_level
    reorder_quantity = max(0, target_level - stock_after_sale) if stock_after_sale < min_level else 0
    reorder_quantity = max(reorder_quantity, -stock_after_sale)

    if stock_after_sale >= 0:
        line.status = "available"
        line.delivery_date = order.request_date
        line.restock_quantity = reorder_quantity
        if reorder_quantity:
            line.restock_eta = get_supplier_delivery_date(order.request_date, reorder_quantity)
            return (f"AVAILABLE: {stock} in stock for {line.quantity} requested "
                    f"({line.catalog_item}). Stock will fall below minimum level {min_level}; "
                    f"RESTOCK_RECOMMENDED {reorder_quantity} units (ETA {line.restock_eta}).")
        return f"AVAILABLE: {stock} in stock for {line.quantity} requested ({line.catalog_item})."

    # Not enough stock: can the supplier deliver in time?
    eta = get_supplier_delivery_date(order.request_date, reorder_quantity)
    line.restock_quantity = reorder_quantity
    line.restock_eta = eta
    if order.deadline and eta > order.deadline:
        line.status = "unavailable"
        on_hand = (f"we only have {stock:,} units of {line.catalog_item} on hand" if stock > 0
                   else f"{line.catalog_item} is currently out of stock")
        line.reason = (f"{on_hand} and a supplier "
                       f"delivery would not arrive until {format_date(eta)}, after your requested "
                       f"date of {format_date(order.deadline)}")
        return f"UNAVAILABLE: {line.reason}"

    restock_cost = reorder_quantity * get_supplier_unit_cost(line.catalog_item)
    cash = get_cash_balance(order.request_date)
    if cash - restock_cost < MINIMUM_CASH_RESERVE:
        line.status = "unavailable"
        line.reason = (f"we cannot source the additional {line.catalog_item} needed for this "
                       f"quantity at the moment")
        order.log(f"restock of {line.catalog_item} blocked by cash reserve (cash {cash:.2f})")
        return f"UNAVAILABLE: insufficient funds to restock ({line.reason})"

    line.status = "restock"
    line.delivery_date = eta
    return (f"RESTOCK_REQUIRED: only {stock} in stock for {line.quantity} requested "
            f"({line.catalog_item}); order {reorder_quantity} units from supplier, ETA {eta} "
            f"(deadline {order.deadline or 'none'}).")


def tool_place_supplier_restock(ctx: RunContext[OrderContext], line_index: int) -> str:
    """Place the supplier purchase order decided by assess_line_item for this line
    (records a 'stock_orders' transaction with create_transaction)."""
    order = ctx.deps
    line = order.line_items[line_index]
    if line.restock_placed or line.restock_quantity <= 0 or line.status == "unavailable":
        return "No restock needed for this line."
    cost = round(line.restock_quantity * get_supplier_unit_cost(line.catalog_item), 2)
    if get_cash_balance(order.request_date) - cost < MINIMUM_CASH_RESERVE:
        return "Restock skipped: it would breach the minimum cash reserve."
    transaction_id = create_transaction(
        line.catalog_item, "stock_orders", line.restock_quantity, cost, order.request_date
    )
    line.restock_placed = True
    order.log(f"restock #{transaction_id}: {line.restock_quantity} x {line.catalog_item} for ${cost:.2f}")
    return f"Supplier order #{transaction_id} placed: {line.restock_quantity} x {line.catalog_item}, ETA {line.restock_eta}."


# Tools for quoting agent
def tool_search_quote_history(ctx: RunContext[OrderContext], search_terms: List[str]) -> str:
    """Search historical quotes for similar requests (uses search_quote_history)."""
    matches = search_quote_history(search_terms, limit=5)
    summary = [
        {"total_amount": m["total_amount"], "order_size": m["order_size"],
         "event_type": m["event_type"], "explanation": str(m["quote_explanation"])[:160]}
        for m in matches
    ]
    if matches:
        amounts = [float(m["total_amount"]) for m in matches]
        ctx.deps.history_note = (
            f"{len(matches)} comparable past quotes ranged from ${min(amounts):,.2f} to "
            f"${max(amounts):,.2f}"
        )
    return json.dumps(summary)


def tool_calculate_quote(ctx: RunContext[OrderContext]) -> str:
    """Price every fulfillable line at list price and apply the bulk discount tier
    that matches the total number of units. Records the quote on the order."""
    order = ctx.deps
    sellable = [line for line in order.line_items if line.status in ("available", "restock")]
    if not sellable:
        return "Nothing to quote: no line can be fulfilled."
    subtotal = 0.0
    for line in sellable:
        line.unit_price = get_unit_price(line.catalog_item)
        line.line_total = round(line.unit_price * line.quantity, 2)
        subtotal += line.line_total
    total_units = sum(line.quantity for line in sellable)
    rate, label = select_bulk_discount(total_units)
    order.quote_subtotal = round(subtotal, 2)
    order.discount_rate = rate
    order.discount_label = label
    order.quote_total = round(subtotal * (1 - rate), 2)
    lines = [f"{l.quantity} x {l.catalog_item} @ ${l.unit_price:.2f} = ${l.line_total:,.2f}" for l in sellable]
    return (f"QUOTE: {'; '.join(lines)}. Subtotal ${order.quote_subtotal:,.2f}, "
            f"discount {rate:.0%} ({label or 'none'}), total ${order.quote_total:,.2f}.")


# Tools for ordering agent
def tool_check_cash_balance(ctx: RunContext[OrderContext]) -> str:
    """Return the company's cash balance as of the request date (uses get_cash_balance)."""
    return f"${get_cash_balance(ctx.deps.request_date):,.2f}"


def tool_financial_health_check(ctx: RunContext[OrderContext]) -> str:
    """Generate the company financial report (uses generate_financial_report) and
    summarise it; used as a health check before booking large orders."""
    report = generate_financial_report(ctx.deps.request_date)
    low_stock = [i["item_name"] for i in report["inventory_summary"] if i["stock"] < 100]
    summary = (f"cash ${report['cash_balance']:,.2f}, inventory value "
               f"${report['inventory_value']:,.2f}, total assets ${report['total_assets']:,.2f}, "
               f"low-stock items: {low_stock or 'none'}")
    ctx.deps.log(f"health check: {summary}")
    return summary


def tool_finalize_sale(ctx: RunContext[OrderContext]) -> str:
    """Book the quoted sale: one 'sales' transaction per fulfillable line at the
    discounted price (uses create_transaction). Re-checks stock first."""
    order = ctx.deps
    sellable = [l for l in order.line_items if l.status in ("available", "restock") and not l.sold]
    if not sellable or order.quote_total <= 0:
        return "No sale to finalize."
    booked = []
    for line in sellable:
        stock = int(get_stock_level(line.catalog_item, order.request_date)["current_stock"].iloc[0])
        if stock < line.quantity:
            line.status = "unavailable"
            line.reason = f"stock of {line.catalog_item} changed before we could confirm your order"
            continue
        discounted_price = round(line.line_total * (1 - order.discount_rate), 2)
        transaction_id = create_transaction(
            line.catalog_item, "sales", line.quantity, discounted_price, order.request_date
        )
        line.sold = True
        booked.append(f"#{transaction_id} {line.quantity} x {line.catalog_item} ${discounted_price:,.2f}")
    order.sale_completed = bool(booked)
    order.quote_total = round(sum(l.line_total for l in order.line_items if l.sold) * (1 - order.discount_rate), 2)
    order.log("sale booked: " + "; ".join(booked))
    return "SALE BOOKED: " + ("; ".join(booked) if booked else "nothing booked")


# Tools for orchestrator agent (delegation + customer response)
def tool_get_order_status(ctx: RunContext[OrderContext]) -> str:
    """Return the shared order record (customer-safe fields only) as JSON."""
    order = ctx.deps
    return json.dumps({
        "request_date": order.request_date,
        "deadline": order.deadline,
        "lines": [
            {"index": i, "description": l.description, "quantity": l.quantity,
             "catalog_item": l.catalog_item, "status": l.status, "reason": l.reason,
             "delivery_date": l.delivery_date, "unit_price": l.unit_price, "sold": l.sold}
            for i, l in enumerate(order.line_items)
        ],
        "subtotal": order.quote_subtotal, "discount_rate": order.discount_rate,
        "discount_reason": order.discount_label, "total": order.quote_total,
        "sale_completed": order.sale_completed,
    })


def compose_customer_response(order: OrderContext) -> str:
    """Build the customer-facing reply from the shared order record.

    Contains the items, prices, discount rationale, delivery dates and reasons
    for anything we cannot supply - but no internal data (cash, margins, ids).
    """
    sold = [l for l in order.line_items if l.sold]
    unavailable = [l for l in order.line_items if l.status == "unavailable"]
    parts = []
    if sold:
        lines = "; ".join(f"{l.quantity:,} x {l.catalog_item} at ${l.unit_price:.2f} each "
                          f"(${l.line_total:,.2f})" for l in sold)
        latest_delivery = max(l.delivery_date or order.request_date for l in sold)
        parts.append(f"Thank you for your order! We have confirmed: {lines}.")
        if order.discount_rate:
            parts.append(f"Subtotal ${sum(l.line_total for l in sold):,.2f}; we applied a "
                         f"{order.discount_label} to reward your order size, bringing your "
                         f"total to ${order.quote_total:,.2f}.")
        else:
            parts.append(f"Your total is ${order.quote_total:,.2f} at our standard list prices "
                         f"(bulk discounts start at 1,000 units).")
        if order.history_note:
            parts.append("Prices follow our published list prices and the same bulk-discount "
                         "approach we have used when quoting similar orders in the past.")
        deadline_note = f", ahead of your requested date of {format_date(order.deadline)}" if order.deadline else ""
        parts.append(f"Expected delivery: {format_date(latest_delivery)}{deadline_note}.")
    if unavailable:
        reasons = "; ".join(f"{l.quantity:,} x {l.description}: {l.reason}" for l in unavailable)
        prefix = "Unfortunately we could not include" if sold else "Unfortunately we are unable to fulfill your request. We could not supply"
        parts.append(f"{prefix} the following: {reasons}.")
        partial = [l for l in unavailable if l.catalog_item and l.stock_on_hand > 0]
        if partial:
            offers = ", ".join(f"{l.stock_on_hand:,} x {l.catalog_item}" for l in partial)
            parts.append(f"If helpful, we can supply {offers} right away or deliver the full "
                         f"quantity on a later date - just let us know.")
    if not order.line_items:
        parts.append("We could not identify any products and quantities in your message. "
                     "Please list the items and quantities you need and we will send a quote.")
    return " ".join(parts)


def _sellable_lines(order: OrderContext) -> List[LineItem]:
    return [l for l in order.line_items if l.status in ("available", "restock")]


def tool_compose_customer_response(ctx: RunContext[OrderContext]) -> str:
    """Return the customer reply built from the order record (send it as-is).

    Guard: refuses (asks the model to retry) if the workflow is incomplete, so
    the orchestrator can never confirm an order that was not booked.
    """
    order = ctx.deps
    if order.line_items and all(l.status == "pending" for l in order.line_items):
        raise ModelRetry("Inventory has not been checked yet. Call delegate_to_inventory_agent first.")
    if _sellable_lines(order) and order.quote_total <= 0:
        raise ModelRetry("Some lines can be fulfilled but no quote exists. Call delegate_to_quoting_agent.")
    if _sellable_lines(order) and not order.sale_completed and not order.sale_note:
        raise ModelRetry("The quote has not been finalized. Call delegate_to_sales_agent before replying.")
    return compose_customer_response(order)


# ---------------------------------------------------------------------------
# Offline (scripted) policies used when no LLM is available.
# Each policy receives the conversation so far and returns the next tool call,
# exactly as an LLM would. This keeps the agent/tool flow identical.
# ---------------------------------------------------------------------------
def _tool_history(messages: List[ModelMessage]) -> List[tuple]:
    """Return [(tool_name, content)] for every tool result seen so far."""
    history = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    history.append((part.tool_name, str(part.content)))
    return history


def _user_prompt(messages: List[ModelMessage]) -> str:
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    return str(part.content)
    return ""


def _call(tool_name: str, **args) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])


def _reply(text_content: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text_content)])


def inventory_policy(messages: List[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Snapshot inventory, then assess every line and place any needed restock."""
    history = _tool_history(messages)
    line_count = int(re.search(r"LINE_COUNT=(\d+)", _user_prompt(messages)).group(1))
    if not history:
        return _call("get_inventory_snapshot")
    assessed = [content for name, content in history if name == "assess_line_item"]
    last_name, last_content = history[-1]
    if last_name == "assess_line_item" and "RESTOCK" in last_content:
        return _call("place_supplier_restock", line_index=len(assessed) - 1)
    if len(assessed) < line_count:
        return _call("assess_line_item", line_index=len(assessed))
    return _reply("Inventory assessment complete: " + " | ".join(assessed))


def quoting_policy(messages: List[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Look up comparable historical quotes, then calculate the quote."""
    history = _tool_history(messages)
    if not history:
        terms = re.search(r"SEARCH_TERMS=(.*)", _user_prompt(messages)).group(1).split("|")
        return _call("search_quote_history", search_terms=[t for t in terms if t][:1])
    if history[-1][0] == "search_quote_history":
        return _call("calculate_quote")
    return _reply(history[-1][1])


def sales_policy(messages: List[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Check cash, run a health check for large orders, then book the sale."""
    history = _tool_history(messages)
    called = [name for name, _ in history]
    is_large = "LARGE_ORDER=yes" in _user_prompt(messages)
    if "check_cash_balance" not in called:
        return _call("check_cash_balance")
    if is_large and "financial_health_check" not in called:
        return _call("financial_health_check")
    if "finalize_sale" not in called:
        return _call("finalize_sale")
    return _reply(history[-1][1])


def orchestrator_policy(messages: List[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Parse -> inventory -> (quote -> sale if anything is sellable) -> reply."""
    history = _tool_history(messages)
    called = [name for name, _ in history]
    if "parse_request" not in called:
        return _call("parse_request")
    if "delegate_to_inventory_agent" not in called:
        if '"items": []' in dict(history)["parse_request"]:
            return _call("compose_customer_response")
        return _call("delegate_to_inventory_agent")
    sellable = "AVAILABLE" in dict(history)["delegate_to_inventory_agent"] or \
        "RESTOCK_REQUIRED" in dict(history)["delegate_to_inventory_agent"]
    if sellable and "delegate_to_quoting_agent" not in called:
        return _call("delegate_to_quoting_agent")
    if sellable and "delegate_to_sales_agent" not in called:
        return _call("delegate_to_sales_agent")
    if "compose_customer_response" not in called:
        return _call("compose_customer_response")
    return _reply(history[-1][1])


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(offline_policy):
    """Return an OpenAI-backed model, or a FunctionModel running `offline_policy`."""
    if OFFLINE_MODE:
        return FunctionModel(offline_policy)
    try:  # pydantic-ai >= 1.0
        from pydantic_ai.models.openai import OpenAIChatModel as OpenAIModelClass
    except ImportError:  # older pydantic-ai releases
        from pydantic_ai.models.openai import OpenAIModel as OpenAIModelClass
    from pydantic_ai.providers.openai import OpenAIProvider
    provider = OpenAIProvider(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    return OpenAIModelClass(OPENAI_MODEL_NAME, provider=provider)


# Set up your agents and create an orchestration agent that will manage them.
from pydantic_ai import Tool  # noqa: E402  (kept next to the agent definitions for readability)


def named_tools(functions) -> List[Tool]:
    """Wrap tool functions so the LLM sees clean names (without the 'tool_' prefix)."""
    return [Tool(function, name=function.__name__.replace("tool_", "", 1)) for function in functions]


inventory_agent = Agent(
    build_model(inventory_policy),
    deps_type=OrderContext,
    output_type=str,
    name="inventory_agent",
    system_prompt=(
        "You are the Inventory Agent for Beaver's Choice Paper Company. "
        "First call get_inventory_snapshot. Then call assess_line_item once for EVERY line "
        "index from 0 to LINE_COUNT-1. Whenever an assessment mentions RESTOCK_REQUIRED or "
        "RESTOCK_RECOMMENDED, immediately call place_supplier_restock for that line index. "
        "Use check_stock_level / check_supplier_delivery for stock questions. "
        "Finish with a one-line internal status per line, keeping the words AVAILABLE, "
        "RESTOCK_REQUIRED or UNAVAILABLE from the assessments."
    ),
    tools=named_tools([tool_get_inventory_snapshot, tool_check_stock_level,
                       tool_check_supplier_delivery, tool_assess_line_item,
                       tool_place_supplier_restock]),
)

quoting_agent = Agent(
    build_model(quoting_policy),
    deps_type=OrderContext,
    output_type=str,
    name="quoting_agent",
    system_prompt=(
        "You are the Quoting Agent for Beaver's Choice Paper Company. Call "
        "search_quote_history with ONE short keyword (the event type or main product) to see "
        "comparable historical quotes, then call calculate_quote, which applies the official "
        "bulk-discount tiers. Never invent prices or discounts. Return the calculate_quote result."
    ),
    tools=named_tools([tool_search_quote_history, tool_calculate_quote]),
)

sales_agent = Agent(
    build_model(sales_policy),
    deps_type=OrderContext,
    output_type=str,
    name="sales_agent",
    system_prompt=(
        "You are the Sales Agent for Beaver's Choice Paper Company. Call check_cash_balance. "
        "If the prompt says LARGE_ORDER=yes, call financial_health_check before booking. "
        "Then call finalize_sale exactly once and return its result."
    ),
    tools=named_tools([tool_check_cash_balance, tool_financial_health_check, tool_finalize_sale]),
)


# --- Orchestrator delegation tools -----------------------------------------
def tool_parse_request(ctx: RunContext[OrderContext]) -> str:
    """Extract line items (description + quantity), deadline and request date from
    the customer's original message and store them on the shared order record.
    (The message is read from the blackboard so it is never paraphrased.)"""
    order = ctx.deps
    parsed = parse_customer_request(order.request_text)
    order.deadline = parsed["deadline"] or order.deadline
    order.line_items = [LineItem(i["description"], i["quantity"]) for i in parsed["items"]]
    return json.dumps(parsed)


async def tool_delegate_to_inventory_agent(ctx: RunContext[OrderContext]) -> str:
    """Ask the Inventory Agent to check stock / restock every parsed line item."""
    order = ctx.deps
    prompt = (f"LINE_COUNT={len(order.line_items)}\nRequest date: {order.request_date}\n"
              f"Customer deadline: {order.deadline}\nLines: " +
              json.dumps([{"index": i, "description": l.description, "quantity": l.quantity}
                          for i, l in enumerate(order.line_items)]))
    result = await inventory_agent.run(prompt, deps=order, usage=ctx.usage)
    order.log(f"inventory agent: {result.output}")
    # Ground truth comes from the blackboard, not from the agent's wording.
    statuses = [f"line {i}: {l.status.upper().replace('RESTOCK', 'RESTOCK_REQUIRED')}"
                f"{' - ' + l.reason if l.reason else ''}" for i, l in enumerate(order.line_items)]
    return "; ".join(statuses)


async def tool_delegate_to_quoting_agent(ctx: RunContext[OrderContext]) -> str:
    """Ask the Quoting Agent to price the fulfillable lines (with bulk discounts)."""
    order = ctx.deps
    if not order.line_items or all(l.status == "pending" for l in order.line_items):
        raise ModelRetry("Call delegate_to_inventory_agent before asking for a quote.")
    if not _sellable_lines(order):
        return "Nothing to quote: no line can be fulfilled. Call compose_customer_response."
    sellable = [l.catalog_item for l in order.line_items if l.status in ("available", "restock")]
    event_match = re.search(r"\b(ceremony|parade|conference|reception|party|assembly|exhibition|"
                            r"show|gathering|performance|demonstration|concert|festival|meeting)\b",
                            order.request_text, re.IGNORECASE)
    terms = [event_match.group(1).lower()] if event_match else []
    terms += [name.split()[0].lower() for name in sellable]
    prompt = f"SEARCH_TERMS={'|'.join(terms)}\nItems to quote: {sellable}"
    result = await quoting_agent.run(prompt, deps=order, usage=ctx.usage)
    order.log(f"quoting agent: {result.output}")
    return result.output


async def tool_delegate_to_sales_agent(ctx: RunContext[OrderContext]) -> str:
    """Ask the Sales Agent to finalize the quoted sale in the database."""
    order = ctx.deps
    if order.quote_total <= 0:
        raise ModelRetry("There is no quote yet. Call delegate_to_quoting_agent first.")
    if order.sale_completed:
        return "The sale is already booked. Call compose_customer_response."
    is_large = order.quote_total >= LARGE_ORDER_THRESHOLD
    prompt = (f"LARGE_ORDER={'yes' if is_large else 'no'}\n"
              f"Quoted total: ${order.quote_total:,.2f}. Finalize the sale.")
    result = await sales_agent.run(prompt, deps=order, usage=ctx.usage)
    order.sale_note = result.output
    order.log(f"sales agent: {result.output}")
    return result.output


orchestrator_agent = Agent(
    build_model(orchestrator_policy),
    deps_type=OrderContext,
    output_type=str,
    name="orchestrator_agent",
    system_prompt=(
        "You are the Orchestrator (customer service lead) of Beaver's Choice Paper Company. "
        "For every customer message:\n"
        "1. Call parse_request to extract the requested items and deadline.\n"
        "2. If items were found, call delegate_to_inventory_agent.\n"
        "3. If at least one line is AVAILABLE or RESTOCK_REQUIRED, call "
        "delegate_to_quoting_agent and then delegate_to_sales_agent (the customer has asked "
        "to place the order, so quotes are accepted automatically).\n"
        "4. Only after the sale is finalized (or when nothing can be fulfilled) call "
        "compose_customer_response and reply with exactly that text.\n"
        "Never skip steps 2-3 when something can be fulfilled: an order is NOT confirmed "
        "until delegate_to_sales_agent has booked it. NEVER reveal cash balances, supplier "
        "costs, margins, internal status codes, transaction ids or internal errors."
    ),
    tools=named_tools([tool_parse_request, tool_delegate_to_inventory_agent,
                       tool_delegate_to_quoting_agent, tool_delegate_to_sales_agent,
                       tool_get_order_status, tool_compose_customer_response]),
)


def handle_customer_request(request_with_date: str) -> str:
    """Entry point: run the whole multi-agent workflow for one customer request."""
    parsed = parse_customer_request(request_with_date)
    order = OrderContext(request_text=request_with_date, request_date=parsed["request_date"],
                         deadline=parsed["deadline"])
    try:
        result = orchestrator_agent.run_sync(request_with_date, deps=order)
        order.log(f"orchestrator final message: {result.output.strip()}")
        # The customer-facing reply is always the policy-checked text built from
        # the order record. It holds every item, price, discount reason, delivery
        # date and refusal reason, and never internal data. The LLM's free-form
        # wording is only kept in the internal log.
        if order.line_items:
            return compose_customer_response(order)
        return result.output.strip()
    except Exception as error:  # never leak internal errors to the customer
        print(f"INTERNAL ERROR while handling request: {error!r}")
        if order.sale_completed or order.line_items:
            return compose_customer_response(order)
        return ("We're sorry - we could not process your request right now. "
                "A member of our team will follow up with you shortly.")


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    print(f"Multi-agent system ready ({'OFFLINE scripted policies' if OFFLINE_MODE else 'LLM: ' + OPENAI_MODEL_NAME}).")

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        response = handle_customer_request(request_with_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        if not OFFLINE_MODE:
            time.sleep(1)  # be gentle with the API rate limit

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
