"""
EchoNote - local AI voice notes.

Start the server and open the web UI in the default browser.
"""
import threading
import webbrowser

import uvicorn

from server.config import Config, ensure_dirs


def main():
    ensure_dirs()
    config = Config()
    url = f"http://{config.host}:{config.port}"
    print(f"EchoNote starting at {url}")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    from server.app import create_app
    uvicorn.run(create_app(config), host=config.host, port=config.port, log_level="warning")


if __name__ == "__main__":
    main()
