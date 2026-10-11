"""Synthetic structured-extraction cases (category ``json``).

Each template renders a short natural-language record that contains every target value
plus distractors (irrelevant numbers, names and dates), and the expected JSON object is
generated together with the text, so answer keys are correct by construction. Cases are
grouped by template so held-out splits test *unseen* templates, not paraphrases of
training ones.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from typing import Any

FIRST = [
    "Maya",
    "Ravi",
    "Elena",
    "Tomás",
    "Aisha",
    "Jonas",
    "Priya",
    "Kenji",
    "Lucía",
    "Omar",
    "Grace",
    "Nikolai",
    "Zara",
    "Wei",
    "Fatima",
    "Liam",
]
LAST = [
    "Okafor",
    "Lindqvist",
    "Patel",
    "Moreau",
    "Nakamura",
    "Haddad",
    "Silva",
    "Kowalski",
    "Reyes",
    "Chen",
    "Mensah",
    "Fischer",
    "Ivanova",
    "Brennan",
]
CITIES = [
    "Lisbon",
    "Nairobi",
    "Osaka",
    "Denver",
    "Tallinn",
    "Montreal",
    "Pune",
    "Valparaíso",
    "Krakow",
    "Auckland",
    "Ghent",
    "Austin",
    "Hanoi",
    "Bergen",
]
JOBS = [
    "data engineer",
    "nurse",
    "architect",
    "pastry chef",
    "civil engineer",
    "teacher",
    "pilot",
    "pharmacist",
    "UX researcher",
    "electrician",
]
PRODUCTS = [
    "noise-cancelling headphones",
    "standing desk",
    "espresso grinder",
    "trail running shoes",
    "4K monitor",
    "mechanical keyboard",
    "camping stove",
    "e-reader",
]
CATEGORIES = ["electronics", "furniture", "kitchen", "sports", "outdoors", "books"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD"]
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
AIRPORTS = ["LIS", "NBO", "KIX", "DEN", "TLL", "YUL", "PNQ", "SCL", "KRK", "AKL", "BRU", "AUS"]
DISHES = [
    "lentil soup",
    "mushroom risotto",
    "chicken tagine",
    "vegetable curry",
    "banana bread",
    "shakshuka",
]
INGREDIENTS = [
    "onion",
    "garlic",
    "olive oil",
    "rice",
    "lentils",
    "cumin",
    "tomato",
    "butter",
    "flour",
    "eggs",
    "spinach",
    "lemon",
    "paprika",
    "ginger",
]
COMPANIES = [
    "Northwind Labs",
    "Bluefin Analytics",
    "Cedar Health",
    "Orbital Freight",
    "Kite Robotics",
    "Lumen Retail",
]
GENRES = ["mystery", "science fiction", "history", "romance", "fantasy", "biography", "thriller"]
CONDITIONS = ["sunny", "overcast", "light rain", "snow", "thunderstorms", "fog"]
MAKES = [
    ("Toyota", "Corolla"),
    ("Honda", "Civic"),
    ("Ford", "Focus"),
    ("Volkswagen", "Golf"),
    ("Hyundai", "Elantra"),
    ("Mazda", "CX-5"),
]


def _name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def _date(rng: random.Random) -> tuple[str, str]:
    y, m, d = rng.randint(2024, 2027), rng.randint(1, 12), rng.randint(1, 28)
    return f"{y:04d}-{m:02d}-{d:02d}", f"{MONTHS[m - 1]} {d}, {y}"


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required or list(props),
        "additionalProperties": False,
    }


STR, INT, NUM, BOOL = (
    {"type": "string"},
    {"type": "integer"},
    {"type": "number"},
    {"type": "boolean"},
)


def t_person(rng: random.Random) -> tuple[str, dict[str, Any], dict[str, Any]]:
    name, age, city, job = _name(rng), rng.randint(21, 70), rng.choice(CITIES), rng.choice(JOBS)
    email = f"{name.split()[0].lower()}.{rng.randint(10, 99)}@example.org"
    other = _name(rng)
    text = (
        f"At the {rng.randint(3, 9)}th annual meetup, {name} introduced themself as a {job} who moved to {city} "
        f"{rng.randint(2, 15)} years ago. {name.split()[0]} is {age} and can be reached at {email}; their colleague "
        f"{other} (age {age + rng.randint(1, 9)}) joined later."
    )
    exp = {"name": name, "age": age, "city": city, "occupation": job, "email": email}
    return text, exp, _obj({"name": STR, "age": INT, "city": STR, "occupation": STR, "email": STR})


def t_product(rng: random.Random) -> tuple[str, dict[str, Any], dict[str, Any]]:
    prod, cat, cur = rng.choice(PRODUCTS), rng.choice(CATEGORIES), rng.choice(CURRENCIES)
    price = round(rng.uniform(19, 900), 2)
    stock = rng.random() < 0.5
    text = (
        f"Listing #{rng.randint(1000, 9999)}: {prod} in our {cat} section, now {price} {cur} "
        f"(was {round(price * 1.2, 2)} {cur}). "
        + (
            "Ships today - currently in stock."
            if stock
            else "Currently out of stock; restock expected next month."
        )
    )
    exp = {
        "product_name": prod,
        "category": cat,
        "price": price,
        "currency": cur,
        "in_stock": stock,
    }
    return (
        text,
        exp,
        _obj(
            {"product_name": STR, "category": STR, "price": NUM, "currency": STR, "in_stock": BOOL}
        ),
    )


def t_event(rng: random.Random) -> tuple[str, dict[str, Any], dict[str, Any]]:
    iso, human = _date(rng)
    _, other_human = _date(rng)
    title = rng.choice(
        [
            "Open Data Summit",
            "Harbor Jazz Night",
            "Robotics Expo",
            "Founders Breakfast",
            "Climate Hackathon",
        ]
    )
    city, n = rng.choice(CITIES), rng.randint(40, 5000)
    text = (
        f"Registration for {title} closes on {other_human}. The event itself takes place on {human} in {city}, "
        f"and organisers expect {n} attendees across {rng.randint(2, 6)} stages."
    )
    exp = {"title": title, "date": iso, "location": city, "expected_attendees": n}
    schema = _obj(
        {
            "title": STR,
            "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
            "location": STR,
            "expected_attendees": INT,
        }
    )
    return text, exp, schema


def t_flight(rng: random.Random) -> tuple[str, dict[str, Any], dict[str, Any]]:
    o, d = rng.sample(AIRPORTS, 2)
    fn = f"{rng.choice(['TP', 'KQ', 'NH', 'UA', 'AY', 'AC'])}{rng.randint(100, 2999)}"
    seat = f"{rng.randint(1, 45)}{rng.choice('ABCDEF')}"
    pax = _name(rng)
    text = (
        f"Booking confirmed for passenger {pax} on flight {fn}, departing {o} at {rng.randint(5, 22):02d}:{rng.choice(['05', '20', '45'])} "
        f"and arriving in {d}. Seat {seat} has been assigned; gate opens {rng.randint(30, 60)} minutes before departure."
    )
    exp = {"passenger": pax, "flight_number": fn, "origin": o, "destination": d, "seat": seat}
    return (
        text,
        exp,
        _obj(
            {"passenger": STR, "flight_number": STR, "origin": STR, "destination": STR, "seat": STR}
        ),
    )


def t_invoice(rng: random.Random) -> tuple[str, dict[str, Any], dict[str, Any]]:
    items = [
        {"name": n, "quantity": rng.randint(1, 6)}
        for n in rng.sample(
            ["widgets", "cables", "brackets", "sensors", "adapters", "panels"], rng.randint(2, 3)
        )
    ]
    total = round(rng.uniform(50, 4000), 2)
    inv = f"INV-{rng.randint(10000, 99999)}"
    cust = rng.choice(COMPANIES)
    lines = ", ".join(f"{it['quantity']} {it['name']}" for it in items)
    text = f"Invoice {inv} was issued to {cust} for {lines}. Amount due: {total} (tax already included, net 30 days, PO {rng.randint(100, 999)})."
    exp = {"invoice_id": inv, "customer": cust, "total": total, "items": items}
    item_schema = _obj({"name": STR, "quantity": INT})
    return (
        text,
        exp,
        _obj(
            {
                "invoice_id": STR,
                "customer": STR,
                "total": NUM,
                "items": {"type": "array", "items": item_schema},
            }
        ),
    )


def t_recipe(rng: random.Random) -> tuple[str, dict[str, Any], dict[str, Any]]:
    dish, serv = rng.choice(DISHES), rng.randint(2, 8)
    ings = rng.sample(INGREDIENTS, rng.randint(3, 5))
    text = (
        f"This {dish} serves {serv} and takes about {rng.randint(20, 90)} minutes. "
        f"You will need, in order of use: {', '.join(ings)}."
    )
    exp = {"dish": dish, "servings": serv, "ingredients": ings}
    return (
        text,
        exp,
        _obj({"dish": STR, "servings": INT, "ingredients": {"type": "array", "items": STR}}),
    )


def t_job(rng: random.Random) -> tuple[str, dict[str, Any], dict[str, Any]]:
    title, comp = (
        rng.choice(["Backend Engineer", "Data Analyst", "Product Designer", "SRE", "ML Engineer"]),
        rng.choice(COMPANIES),
    )
    lo = rng.randrange(60, 150) * 1000
    hi = lo + rng.randrange(10, 60) * 1000
    remote = rng.random() < 0.5
    text = (
        f"{comp} is hiring a {title} ({rng.randint(2, 8)}+ years of experience). Salary range {lo} to {hi} per year. "
        + ("Fully remote within the EU." if remote else "This role is on-site five days a week.")
    )
    exp = {
        "job_title": title,
        "company": comp,
        "salary_min": lo,
        "salary_max": hi,
        "remote": remote,
    }
    return (
        text,
        exp,
        _obj(
            {"job_title": STR, "company": STR, "salary_min": INT, "salary_max": INT, "remote": BOOL}
        ),
    )


def t_book(rng: random.Random) -> tuple[str, dict[str, Any], dict[str, Any]]:
    title = rng.choice(
        [
            "The Glass Orchard",
            "Tides of Meridian",
            "A Map of Small Hours",
            "Copper Season",
            "The Last Cartographer",
        ]
    )
    author, year = _name(rng), rng.randint(1950, 2025)
    genres = rng.sample(GENRES, 2)
    text = (
        f"First published in {year} and reissued in {year + rng.randint(5, 30)}, '{title}' by {author} "
        f"blends {genres[0]} with {genres[1]}. It runs {rng.randint(180, 620)} pages."
    )
    exp = {"title": title, "author": author, "first_published": year, "genres": genres}
    return (
        text,
        exp,
        _obj(
            {
                "title": STR,
                "author": STR,
                "first_published": INT,
                "genres": {"type": "array", "items": STR},
            }
        ),
    )


def t_weather(rng: random.Random) -> tuple[str, dict[str, Any], dict[str, Any]]:
    city, t, cond, hum = (
        rng.choice(CITIES),
        rng.randint(-15, 38),
        rng.choice(CONDITIONS),
        rng.randint(15, 98),
    )
    text = (
        f"Forecast update for {city}: {cond} with a high of {t} degrees Celsius (yesterday reached {t - rng.randint(1, 6)}). "
        f"Relative humidity around {hum} percent, winds {rng.randint(5, 40)} km/h."
    )
    exp = {"city": city, "temperature_c": t, "condition": cond, "humidity_pct": hum}
    return (
        text,
        exp,
        _obj({"city": STR, "temperature_c": INT, "condition": STR, "humidity_pct": INT}),
    )


def t_ticket(rng: random.Random) -> tuple[str, dict[str, Any], dict[str, Any]]:
    tid, prio = f"SUP-{rng.randint(1000, 9999)}", rng.choice(["low", "medium", "high"])
    email = f"{rng.choice(FIRST).lower()}@{rng.choice(['acme', 'globex', 'initech'])}.com"
    prod = rng.choice(["billing portal", "mobile app", "API gateway", "data export"])
    text = (
        f"Ticket {tid} opened by {email} about the {prod}. Triage marked it {prio} priority after {rng.randint(2, 9)} similar "
        f"reports this week."
    )
    exp = {"ticket_id": tid, "customer_email": email, "product": prod, "priority": prio}
    schema = _obj(
        {
            "ticket_id": STR,
            "customer_email": STR,
            "product": STR,
            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        }
    )
    return text, exp, schema


def t_car(rng: random.Random) -> tuple[str, dict[str, Any], dict[str, Any]]:
    make, model = rng.choice(MAKES)
    year, km, price = (
        rng.randint(2008, 2024),
        rng.randrange(5, 220) * 1000,
        rng.randrange(30, 400) * 100,
    )
    text = (
        f"For sale: {year} {make} {model}, {km} km on the odometer, one previous owner, service done at {km - rng.randint(1, 4) * 1000} km. "
        f"Asking {price} or best offer."
    )
    exp = {"make": make, "model": model, "year": year, "mileage_km": km, "price": price}
    return (
        text,
        exp,
        _obj({"make": STR, "model": STR, "year": INT, "mileage_km": INT, "price": INT}),
    )


def t_meeting(rng: random.Random) -> tuple[str, dict[str, Any], dict[str, Any]]:
    org = _name(rng)
    parts = [_name(rng) for _ in range(rng.randint(2, 3))]
    start = f"{rng.randint(8, 17):02d}:{rng.choice(['00', '15', '30', '45'])}"
    dur = rng.choice([15, 30, 45, 60, 90])
    text = (
        f"{org} scheduled a {dur}-minute planning session starting at {start} with {', '.join(parts[:-1])} and {parts[-1]}. "
        f"Room {rng.randint(100, 499)} is booked."
    )
    exp = {"organizer": org, "participants": parts, "start_time": start, "duration_minutes": dur}
    schema = _obj(
        {
            "organizer": STR,
            "participants": {"type": "array", "items": STR},
            "start_time": {"type": "string", "pattern": r"^\d{2}:\d{2}$"},
            "duration_minutes": INT,
        }
    )
    return text, exp, schema


TEMPLATES: dict[str, Callable[[random.Random], tuple[str, dict[str, Any], dict[str, Any]]]] = {
    "person": t_person,
    "product": t_product,
    "event": t_event,
    "flight": t_flight,
    "invoice": t_invoice,
    "recipe": t_recipe,
    "job": t_job,
    "book": t_book,
    "weather": t_weather,
    "ticket": t_ticket,
    "car": t_car,
    "meeting": t_meeting,
}

SYSTEM = (
    "Extract the requested fields from the text into a JSON object that conforms exactly to the given JSON Schema. "
    "Use the exact values from the text (dates as YYYY-MM-DD, numbers as numbers). Return only the JSON object."
)


def generate_json_cases(n: int, seed: int) -> list[dict[str, Any]]:
    per = max(1, n // len(TEMPLATES))
    out: list[dict[str, Any]] = []
    for name, fn in TEMPLATES.items():
        rng = random.Random(f"{seed}:{name}")
        for i in range(per):
            text, expected, schema = fn(rng)
            out.append(
                {
                    "id": f"json-{name}-{i:03d}",
                    "category": "json",
                    "source": f"synthetic:json_templates:{name}",
                    "license": "MIT",
                    "provenance": "synthetic_generated",
                    "group_id": f"json-template-{name}",
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {
                            "role": "user",
                            "content": f"JSON Schema:\n{json.dumps(schema)}\n\nText:\n{text}",
                        },
                    ],
                    "max_tokens": 300,
                    "response_format": {"type": "json_object"},
                    "scorer": "json_fields",
                    "reference": {"schema": schema, "expected": expected},
                }
            )
    return out
