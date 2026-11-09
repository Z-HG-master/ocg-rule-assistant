# -*- coding: utf-8 -*-
"""启动测试台：python run.py"""
from app.webui import app
from app import config

if __name__ == "__main__":
    print("配置来源:", config.load().get("_source"))
    print("Embedding:", config.get("embedding.model"), "@", config.get("embedding.base_url"))
    print("Chat:", config.get("chat.model"), "@", config.get("chat.base_url"))
    app.run(host=config.get("server.host", "127.0.0.1"),
            port=int(config.get("server.port", 7871)), threaded=True)
