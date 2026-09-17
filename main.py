# %%
"""Консольный клиент агента (альтернатива HTTP API)."""

import json
import os

import agent
import config

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.HISTORY_FILE)
SESSION_ID = "console"


def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def save_history(history: list) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False)


def show_history():
    for i, e in enumerate(load_history()[-5:], 1):
        print(f"{i}. [{e.get('role', '?')}] {e.get('content', '')[:100]}")


def main():
    config.init_console_utf8()
    history = load_history()
    while True:
        user_input = input("\nВопрос (/history, /clear, /quit): ")
        cmd = user_input.strip().lower()
        if cmd == "/quit":
            break
        if cmd == "/history":
            show_history()
            continue
        if cmd == "/clear":
            history = []
            print("История очищена.")
            continue

        history.append({"role": "user", "content": user_input})
        result = agent.run_my_agent_logic(user_input, SESSION_ID, history)
        answer = result.get("answer", "")
        print("\n" + answer)
        if result.get("sources"):
            print("Источники: " + ", ".join(result["sources"]))
        history.append({"role": "assistant", "content": answer})
        save_history(history)


if __name__ == "__main__":
    main()