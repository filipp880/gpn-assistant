# %%
from agent import agent, load_history, save_history

def show_history():
    for i, e in enumerate(load_history()[-5:],1):
        print(f"{i}. [{e['timestamp'][:16]}]{e['query']}")
        print(f" Ответ: {e['response'][:100]}...")
        print(f" Источники: {', '.join(e['sources']) or 'нет'}")

while True:
    user_input = input("\nВопрос (/history, /clear, /quit): ")
    cmd = user_input.strip().lower()
    if cmd == '/quit':
        break
    elif cmd == '/history':
        show_history()
    elif cmd == '/clear':
        save_history([])
        print("История очищена.")
    else:
        agent(user_input)

    


