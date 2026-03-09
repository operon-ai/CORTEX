import re
import os

cortex_path = r'c:\Users\udita\Desktop\CORTEX\cua_agents\v1\agents\cortex.py'
app_ts_path = r'c:\Users\udita\Desktop\CORTEX\gui\renderer\app.ts'
ws_server_path = r'c:\Users\udita\Desktop\CORTEX\cua_agents\v1\agents\ws_server.py'

emojis_to_remove = ["🧠", "🖱️", "🔧", "💻", "🔩", "🛠️", "⏹️", "⏰", "📷", "📋", "▶️", "🏁", "📝", "💬"]

def remove_emojis_from_file(path):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Remove specific emojis
    for e in emojis_to_remove:
        text = text.replace(e, "")
    
    # Specifically for _log_fn, remove ✅ and ❌ only if they are the last argument
    # e.g. _log_fn(f"❌ Step...", "warning", "❌") -> _log_fn(f"❌ Step...", "warning", "")
    text = re.sub(r'(_log_fn\([^,]+,\s*"[^"]+",\s*")[❌✅]("\))', r'\1\2', text)
    
    # Fix CodeAgent.execute stop_flag
    text = re.sub(r'^\s*stop_flag=_stop_flag,?\s*\r?\n', '', text, flags=re.MULTILINE)
    
    # Write back
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

remove_emojis_from_file(cortex_path)
remove_emojis_from_file(app_ts_path)
remove_emojis_from_file(ws_server_path)
