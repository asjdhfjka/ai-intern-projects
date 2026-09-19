import httpx

# 你的本地接口地址
url = "http://127.0.0.1:8000/chat/stream"
# 测试参数
params = {"user_input": "你好，请写一首只有四句的短诗"}

print("开始接收流式响应：\n")

# 发送流式请求
with httpx.stream("POST", url, params=params) as response:
    for chunk in response.iter_text():
        if chunk:
            # 注意这里用了 flush=True，才能看到打字机效果
            print(chunk, end="", flush=True)

print("\n\n[流式结束]")