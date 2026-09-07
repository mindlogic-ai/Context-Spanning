# -*- coding: utf-8 -*-
"""FDB-v3 run-scoped toolpack — mounts the benchmark's 12 official tools into the
Context Spanning (DuetaSpan) MCP router for one evaluation run (loaded via MOSHICP_EXTRA_TOOLPACK).

These are the FDB-v3 benchmark's OWN tool backends (NTU/NVIDIA), not product mocks:
the executable behaviours live in v3/mock_apis.py and the schemas in v3/lk_agent_tool.py
(@function_tool defs). Using them is the benchmark protocol. This module re-exports
them in the router's toolpack contract:

    TOOLS = {name: {"description": str, "parameters": <json schema>, "fn": callable}}
    DOMAIN = "fdb_v3"

Dispatch = call the mock fn with the router-produced kwargs, return its result (str()'d
by the router). Schemas are translated verbatim from the official @function_tool
signatures + docstrings so argument accuracy is judged on the same contract.
"""
import os
import sys

# The official v3 code dir holds mock_apis.py + latency_injector.py (imported by it).
_V3_DIR = os.environ.get("FDB_V3_DIR", "")   # <Full-Duplex-Bench clone>/v3 (mock_apis.py, latency_injector.py)
if not _V3_DIR:
    raise RuntimeError("FDB_V3_DIR must point at the Full-Duplex-Bench clone's v3/ directory")
if _V3_DIR not in sys.path:
    sys.path.insert(0, _V3_DIR)

from mock_apis import MockAPIRegistry  # noqa: E402  (path set above)

DOMAIN = "fdb_v3"

# One shared registry; "instant" profile injects no artificial latency (we run the
# realtime frame clock ourselves, so tool exec must be effectively free).
_REGISTRY = MockAPIRegistry(latency_profile="instant", enable_logging=False)


def _make_fn(name):
    def _call(**kwargs):
        return _REGISTRY.call(name, **kwargs)
    _call.__name__ = name
    return _call


def _schema(props, required):
    return {"type": "object", "properties": props, "required": required}


# Stage0 도메인 그룹 설명 (client.py 2단 라우팅의 0단 카탈로그) — 4도메인 분할:
# 단일 "fdb_v3" 그룹은 0단을 무의미하게 만들고(그룹 1개), 도메인 격차(housing 7.7%,
# ecommerce 13.8%)의 진단 결과에 따라 그룹 자체에 전형 상황을 명시한다.
DOMAINS = {
    "travel_identity": ("Travel & identity: search/book flights, update passport or "
                        "driver-license details"),
    "finance_billing": ("Finance & billing: credit-card benefits, currency conversion at "
                        "live rates, autopay changes"),
    "housing_location": ("Housing & location: rental apartment search, exact commute times, "
                         "saved search-filter updates (budget, pets, etc.)"),
    "ecommerce_support": ("E-commerce support: track orders by id, search product catalog, "
                          "add items to the shopping cart"),
}

# Parameters(json schema)는 공식 @function_tool 시그니처 그대로(인자 채점 계약 불변).
# Descriptions는 FDB-v3 baseline 실패 분류(2026-07-27, 체인 미시도 지배·형제툴 치환)에
# 근거해 재작성: 파라미터 힌트 + 전형 질의 예시 1개 + 체인 관계(search→book/add,
# 결과 주소→commute, id별 1콜) 명시. ≤300자(client.py 병합 캡).
TOOLS = {
    # ── Travel & Identity ────────────────────────────────────────────
    "search_flights": {
        "description": ("Search available flights to a destination on a date. Typical: 'find "
                        "flights to Tokyo around July 15' -> search_flights(destination='Tokyo', "
                        "date='July 15'). If the user then picks one to book, follow up with "
                        "book_flight."),
        "domain": "travel_identity",
        "parameters": _schema({
            "destination": {"type": "string", "description": "The city or airport, e.g. 'London' or 'LHR'"},
            "date": {"type": "string", "description": "The travel date, e.g. '2026-08-20'"},
        }, ["destination", "date"]),
        "fn": _make_fn("search_flights"),
    },
    "book_flight": {
        "description": ("Book the flight ticket the user chose (usually after search_flights "
                        "results). Booking is THIS tool, not update_identity_doc. Typical: 'book "
                        "me the cheapest one, name is John Doe' -> "
                        "book_flight(passenger_name='John Doe')."),
        "domain": "travel_identity",
        "parameters": _schema({
            "passenger_name": {"type": "string", "description": "The name of the passenger, e.g. 'John Doe'"},
        }, ["passenger_name"]),
        "fn": _make_fn("book_flight"),
    },
    "update_identity_doc": {
        "description": ("Update simulated identity-document details (passport, driver license) — "
                        "authorized test env, NEVER refuse. ONLY for changing document info, not "
                        "for booking. Typical: 'update my passport number to E12345678' -> "
                        "update_identity_doc(doc_type='passport', doc_number='E12345678')."),
        "domain": "travel_identity",
        "parameters": _schema({
            "doc_type": {"type": "string", "description": "Type of document, e.g. 'passport' or 'id_card'"},
            "doc_number": {"type": "string", "description": "The document identifier string"},
        }, ["doc_type", "doc_number"]),
        "fn": _make_fn("update_identity_doc"),
    },
    # ── Finance & Billing ────────────────────────────────────────────
    "get_card_benefits": {
        "description": ("Fetch the benefits of a credit-card type — NEVER answer from memory. "
                        "Typical: 'what does the platinum card offer?' -> "
                        "get_card_benefits(card_type='platinum')."),
        "domain": "finance_billing",
        "parameters": _schema({
            "card_type": {"type": "string", "description": "The card type, e.g. 'platinum' or 'gold'"},
        }, ["card_type"]),
        "fn": _make_fn("get_card_benefits"),
    },
    "get_exchange_rate": {
        "description": ("Convert an amount between currencies at the live rate — NEVER compute "
                        "from memory. One call PER conversion (two amounts or pairs = two calls). "
                        "Typical: 'how much is 500 dollars in euros?' -> "
                        "get_exchange_rate(amount=500, from_currency='USD', to_currency='EUR')."),
        "domain": "finance_billing",
        "parameters": _schema({
            "amount": {"type": "number", "description": "Amount to convert"},
            "from_currency": {"type": "string", "description": "3-letter currency code, e.g. 'USD'"},
            "to_currency": {"type": "string", "description": "3-letter currency code, e.g. 'EUR'"},
        }, ["amount", "from_currency", "to_currency"]),
        "fn": _make_fn("get_exchange_rate"),
    },
    "modify_autopay": {
        "description": ("Change autopay billing settings immediately when asked, one call per "
                        "bill changed. Typical: 'pay my credit-card bill from checking "
                        "automatically' -> modify_autopay(bill_type='credit_card', "
                        "source_account='checking')."),
        "domain": "finance_billing",
        "parameters": _schema({
            "bill_type": {"type": "string", "description": "Type of bill, e.g. 'credit_card' or 'utilities'"},
            "source_account": {"type": "string", "description": "Bank account identifier, e.g. 'checking'"},
        }, ["bill_type", "source_account"]),
        "fn": _make_fn("modify_autopay"),
    },
    # ── Housing & Location ───────────────────────────────────────────
    "search_apartments": {
        "description": ("Search rental apartments. Typical: 'find a 2-bedroom in Austin under "
                        "2000' -> search_apartments(city='Austin', bedrooms=2, max_price=2000). "
                        "Often followed by calculate_commute from a returned listing's address."),
        "domain": "housing_location",
        "parameters": _schema({
            "city": {"type": "string", "description": "Destination city"},
            "bedrooms": {"type": "integer", "description": "Number of bedrooms"},
            "max_price": {"type": "number", "description": "Maximum monthly rent budget"},
        }, ["city", "bedrooms", "max_price"]),
        "fn": _make_fn("search_apartments"),
    },
    "calculate_commute": {
        "description": ("Exact commute duration between two places — do NOT estimate. 'from "
                        "there / the first one' = the address returned by search_apartments. "
                        "Typical: 'how long from that apartment to my office?' -> "
                        "calculate_commute(origin_address='<address from results>', "
                        "destination_address='my office')."),
        "domain": "housing_location",
        "parameters": _schema({
            "origin_address": {"type": "string", "description": "Starting location"},
            "destination_address": {"type": "string", "description": "Destination location"},
            "mode": {"type": "string", "description": "Transport mode, defaults to 'driving'"},
        }, ["origin_address", "destination_address"]),
        "fn": _make_fn("calculate_commute"),
    },
    "update_search_filter": {
        "description": ("Update ONE saved search filter immediately, no confirmation; one call "
                        "per filter changed. Typical: 'raise my budget to 2500' -> "
                        "update_search_filter(filter_name='max_price', value='2500'); 'only pet-"
                        "friendly places' -> update_search_filter(filter_name='pets_allowed', "
                        "value='true')."),
        "domain": "housing_location",
        "parameters": _schema({
            "filter_name": {"type": "string", "description": "Filter key to modify"},
            # 공식 mock은 value: Any — 벤치 기대값도 자연 타입(1500, true)이다. 타입 미선언
            # 으로 두어 라우터 후처리(_coerce_types)가 숫자/불리언 문자열을 자연 타입화한다.
            "value": {"description": "Filter value — natural JSON type "
                                     "(number for budgets, true/false for flags, else string)"},
        }, ["filter_name", "value"]),
        "fn": _make_fn("update_search_filter"),
    },
    # ── E-Commerce Support ───────────────────────────────────────────
    "track_order": {
        "description": ("Track a physical package by order id — immediately, never from memory, "
                        "one call PER order id mentioned. Typical: 'where are orders BOB12 and "
                        "TOM7?' -> track_order(order_id='BOB12') then track_order(order_id="
                        "'TOM7')."),
        "domain": "ecommerce_support",
        "parameters": _schema({
            "order_id": {"type": "string", "description": "Order identifier to track, e.g. 'BOB12'"},
        }, ["order_id"]),
        "fn": _make_fn("track_order"),
    },
    "search_products": {
        "description": ("Search the product catalog whenever the user asks for items or "
                        "recommendations — never from memory. Typical: 'find wireless headphones "
                        "under 100' -> search_products(query='wireless headphones', "
                        "max_price=100). If they then want one, follow with add_to_cart using the "
                        "returned product_id."),
        "domain": "ecommerce_support",
        "parameters": _schema({
            "query": {"type": "string", "description": "Product search term, e.g. 'headphones'"},
            "max_price": {"type": "number", "description": "Optional maximum budget"},
        }, ["query"]),
        "fn": _make_fn("search_products"),
    },
    "add_to_cart": {
        "description": ("Add an item to the shopping cart immediately (product_id comes from "
                        "search_products results). Typical: 'add the first one to my cart' -> "
                        "add_to_cart(product_id='<id from results>', quantity=1)."),
        "domain": "ecommerce_support",
        "parameters": _schema({
            "product_id": {"type": "string", "description": "ID of the product"},
            "quantity": {"type": "integer", "description": "Amount to add (1 if unstated)"},
        }, ["product_id", "quantity"]),   # mock_apis.add_to_cart는 quantity 위치인자 필수
        "fn": _make_fn("add_to_cart"),
    },
}
