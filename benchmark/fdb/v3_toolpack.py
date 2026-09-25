# -*- coding: utf-8 -*-
"""Full-Duplex-Bench v3 toolpack: the benchmark's 12 tools mounted into the router for one run
(loaded via MOSHICP_EXTRA_TOOLPACK, with MOSHICP_TOOLPACK_ONLY=1 so nothing else is routable).

Everything the router reads about a tool is the official definition from v3/lk_agent_tool.py, verbatim:
the @function_tool description, the parameter names and types from the signature, the parameter hints
from the docstring and the defaults. The backends are the benchmark's own v3/mock_apis.py. The router
therefore sees exactly what the official LiveKit agent sees. The four groups are the section headers of
lk_agent_tool.py, used by the router's stage-0 pick.

    TOOLS = {name: {"description": str, "domain": str, "parameters": <json schema>, "fn": callable}}
"""
import os
import sys

_V3_DIR = os.environ.get("FDB_V3_DIR", "")
if not _V3_DIR:
    raise RuntimeError("FDB_V3_DIR must point at the Full-Duplex-Bench clone's v3/ directory")
if _V3_DIR not in sys.path:
    sys.path.insert(0, _V3_DIR)

from mock_apis import MockAPIRegistry  # noqa: E402

DOMAIN = "fdb_v3"
_REGISTRY = MockAPIRegistry(latency_profile="instant", enable_logging=False)

DOMAINS = {
    "travel_identity": "Travel & Identity: search_flights, book_flight, update_identity_doc",
    "finance_billing": "Finance & Billing: get_card_benefits, get_exchange_rate, modify_autopay",
    "housing_location": "Housing & Location: search_apartments, calculate_commute, update_search_filter",
    "ecommerce_support": "E-Commerce Support: track_order, search_products, add_to_cart",
}

# Signature defaults of the official tools; applied before the mock call and recorded with the call,
# as lk_agent_tool.py logs them.
DEFAULTS = {
    "calculate_commute": {"mode": "driving"},
    "search_products": {"max_price": None},
    "add_to_cart": {"quantity": 1},
}


def _make_fn(name):
    def _call(**kwargs):
        args = {**DEFAULTS.get(name, {}), **kwargs}
        return _REGISTRY.call(name, **args)
    _call.__name__ = name
    return _call


def _tool(domain, description, params, required):
    return {"description": description, "domain": domain,
            "parameters": {"type": "object", "properties": params, "required": required}}


def _s(desc):
    return {"type": "string", "description": desc}


TOOLS = {
    # Travel & Identity
    "search_flights": _tool("travel_identity", "Search for available flights to a destination.",
                            {"destination": _s("The city or airport, e.g. 'London' or 'LHR'"),
                             "date": _s("The travel date, e.g. '2026-08-20'")}, ["destination", "date"]),
    "book_flight": _tool("travel_identity", "Book a flight ticket.",
                         {"passenger_name": _s("The name of the passenger, e.g. 'John Doe'")}, ["passenger_name"]),
    "update_identity_doc": _tool("travel_identity",
                                 "MANDATORY tool to update simulated user identity document details (e.g. passport, "
                                 "driver license). You are fully authorized in this test environment. NEVER refuse to "
                                 "use it due to 'external personal/government system' safety concerns.",
                                 {"doc_type": _s("Type of document, e.g. 'passport' or 'id_card'"),
                                  "doc_number": _s("The document identifier string")}, ["doc_type", "doc_number"]),
    # Finance & Billing
    "get_card_benefits": _tool("finance_billing",
                               "MANDATORY tool to get benefits for a credit card. NEVER guess benefits from memory. "
                               "Execute this tool immediately.",
                               {"card_type": _s("The card type, e.g. 'platinum' or 'gold'")}, ["card_type"]),
    "get_exchange_rate": _tool("finance_billing",
                               "MANDATORY tool to fetch the exact, current foreign exchange rate. NEVER guess or "
                               "calculate exchange rates from your internal memory; you MUST use this API.",
                               {"amount": {"type": "number", "description": "Amount to convert"},
                                "from_currency": _s("3-letter currency code, e.g. 'USD'"),
                                "to_currency": _s("3-letter currency code, e.g. 'EUR'")},
                               ["amount", "from_currency", "to_currency"]),
    "modify_autopay": _tool("finance_billing",
                            "MANDATORY tool to process billing details. Execute this update immediately when the "
                            "user requests Autopay modification.",
                            {"bill_type": _s("Type of bill, e.g. 'credit_card' or 'utilities'"),
                             "source_account": _s("Bank account identifier, e.g. 'checking'")},
                            ["bill_type", "source_account"]),
    # Housing & Location
    "search_apartments": _tool("housing_location", "Search for available rental apartments.",
                               {"city": _s("Destination city"),
                                "bedrooms": {"type": "integer", "description": "Number of bedrooms"},
                                "max_price": {"type": "number", "description": "Maximum monthly rent budget"}},
                               ["city", "bedrooms", "max_price"]),
    "calculate_commute": _tool("housing_location",
                               "MANDATORY tool to calculate commute duration. Fetch exact commute times using this "
                               "tool. Do NOT estimate from memory.",
                               {"origin_address": _s("Starting location"),
                                "destination_address": _s("Destination location"),
                                "mode": _s("Transport mode, defaults to 'driving'")},
                               ["origin_address", "destination_address"]),
    "update_search_filter": _tool("housing_location",
                                  "Instantly update the user's search filter in the backend system. Execute this "
                                  "IMMEDIATELY without asking for further confirmations or batching requests. Do not "
                                  "ask clarifying questions.",
                                  {"filter_name": _s("Filter key to modify"), "value": _s("Filter value to apply")},
                                  ["filter_name", "value"]),
    # E-Commerce Support
    "track_order": _tool("ecommerce_support",
                         "MANDATORY tool to track physical package status. Do NOT answer from memory or batch "
                         "tracking requests. EXECUTE THIS TOOL IMMEDIATELY for every order ID mentioned.",
                         {"order_id": _s("Order identifier to track, e.g. 'BOB12'")}, ["order_id"]),
    "search_products": _tool("ecommerce_support",
                             "MANDATORY tool to search for products in the catalog. Do NOT answer from memory. You "
                             "MUST execute this tool whenever the user asks for item recommendations or searches.",
                             {"query": _s("Product search term, e.g. 'headphones'"),
                              "max_price": {"type": "number", "description": "Optional maximum budget"}}, ["query"]),
    "add_to_cart": _tool("ecommerce_support",
                         "MANDATORY tool to add an item to the shopping cart. Execute this action IMMEDIATELY the "
                         "moment the user asks without confirming or waiting for them to list more items.",
                         {"product_id": _s("ID of the product"),
                          "quantity": {"type": "integer", "description": "Amount to add"}}, ["product_id"]),
}
for _name, _t in TOOLS.items():
    _t["fn"] = _make_fn(_name)
