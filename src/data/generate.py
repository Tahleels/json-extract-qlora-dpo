# src/data/generate.py
from __future__ import annotations
import random
from faker import Faker
from src.schema import PersonRecord

# Natural-language templates: {name},{age},{job_title},{company},{city},{email}
TEMPLATES = [
    "Hi, I'm {name}, {age}. I work as a {job_title} at {company} in {city}. Reach me at {email}.",
    "{name} ({age}) — {job_title} @ {company}, based in {city}. Email: {email}",
    "Please add {name}, aged {age}, a {job_title} for {company} located in {city} ({email}).",
    "Contact: {name} | Role: {job_title} | Company: {company} | City: {city} | Age: {age} | {email}",
    "This is {name} from {company} in {city}. I'm {age} and my role is {job_title}. {email} is my email.",
    "Meet {name}, our new {job_title} at {company}. She's {age}, lives in {city}, and uses {email}.",
    "Registration — Name: {name}, Age: {age}, Title: {job_title}, Org: {company}, Location: {city}, Mail: {email}",
    "{name} just joined {company} as a {job_title}. Age {age}, from {city}. Contact {email}.",
]


def generate_pairs(n: int, seed: int = 42) -> list[dict]:
    """Return list of {'text': str, 'json': PersonRecord-as-dict}."""
    fake = Faker()
    Faker.seed(seed)
    random.seed(seed)

    rows = []
    for _ in range(n):
        name = fake.name()
        rec = PersonRecord(
            name=name,
            age=random.randint(18, 65),
            job_title=fake.job(),
            company=fake.company(),
            city=fake.city(),
            email=(name.lower().replace(" ", ".").replace("'", "")
                   + "@" + fake.domain_name()),
        )
        text = random.choice(TEMPLATES).format(
            name=rec.name, age=rec.age, job_title=rec.job_title,
            company=rec.company, city=rec.city, email=rec.email,
        )
        rows.append({"text": text, "json": rec.model_dump()})
    return rows


if __name__ == "__main__":
    from src.config import load_config
    cfg = load_config()
    pairs = generate_pairs(cfg.data["n_samples"], cfg.seed)
    print(f"Generated {len(pairs)} pairs. Example:\n{pairs[0]}")
