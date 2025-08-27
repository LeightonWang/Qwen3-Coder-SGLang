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
    "model": "Qwen/Qwen3-32B",
    "messages": [
      {"role": "user", "content": "Damn, I finally deployed you successfully"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```