# Serving Qwen-3 with SGLang
This repo is the implementation for the Microsoft internship assignment.

## Delpoyment

> **TO-DO**: Use a Python script instead of bash script.

Run the following shell to start the backend:
```bash
bash sgl.sh
```

Now the Qwen3-0.6B has been deployed at `http://127.0.0.1:30000`. For simple chatting usage, just run the following command:
```bash
curl --location --request POST 'http://127.0.0.1:30000/v1/chat/completions' --header 'Content-Type: application/json' --data-raw '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Damn, I finally deployed you successfully"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

The response should be like:
```json
{"id":"12730b4f37924164a9d051d8d8c65d43","object":"chat.completion","created":1756301670,"model":"Qwen/Qwen3-0.6B","choices":[{"index":0,"message":{"role":"assistant","content":"Great job! 🎉 I'm really glad you did it. If you have any questions or need further assistance, feel free to ask! 😊","reasoning_content":"Okay, the user said \"Damn, I finally deployed you successfully.\" Let me think about how to respond. First, I need to acknowledge their success. Maybe start with something like, \"Great job!\" to show excitement. Then, add something helpful, like confirming that the deployment was successful and offer further assistance. Keep the tone positive and open for questions. Avoid any negative language, so just focus on celebration and support. Make sure the response is friendly and engaging.\n","tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":15,"total_tokens":145,"completion_tokens":130,"prompt_tokens_details":null}}
```