import re

used = set()
for mod in ("config.py", "app.py", "agent.py", "retrieval.py", "ingest.py",
            "sessions.py", "main.py", "store.py"):
    src = open(mod, encoding="utf-8").read()
    used |= set(re.findall(r'os\.getenv\([\"\x27]([A-Z0-9_]+)', src))

example = open(".env.example", encoding="utf-8").read()
have = set(re.findall(r"^([A-Z0-9_]+)=", example, re.M))

readme = open("README.md", encoding="utf-8").read()
readme_vars = set(re.findall(r"`([A-Z][A-Z0-9_]{2,})`", readme))

print("USED(config/app/agent/retrieval/ingest/sessions/main/store):")
print("  ", sorted(used))
print("MISSING from .env.example:", sorted(used - have) or "нет")
print("EXTRA in .env.example:", sorted(have - used) or "нет")

missing_readme = used - readme_vars
print("Used-in-code but not documented in README:", sorted(missing_readme) or "нет")
