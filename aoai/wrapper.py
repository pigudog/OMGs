# aoai/wrapper.py
from openai import OpenAI
import time
from datetime import datetime
from .logger import DBLogger


class OpenAIWrapper:
    def __init__(self, api_key, base_url=None, db_path="api_trace.db"):
        """
        Initialize wrapper around OpenAI / Azure OpenAI client.
        
        Parameters:
        - api_key: API key for OpenAI or Azure OpenAI.
        - base_url: Optional. Required for Azure OpenAI such as:
                    https://<resource>.openai.azure.com/openai/v1/
        - db_path: SQLite database path for request tracing and logging.
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,   # Can be None for OpenAI public endpoint
        )
        self.logger = DBLogger(db_path)

    def chat_completion(self, *, model="gpt-5-mini", messages=None, **kwargs):
        """
        A complete replacement of client.chat.completions.create()
        Automatically logs request/response metadata to SQLite.
        
        Parameters:
        - model: deployment name or model ID
        - messages: list of {"role": "...", "content": "..."}
        - kwargs: any other OpenAI ChatCompletion parameters
        
        Returns:
        - resp: raw API response
        """
        print(f"Calling model: {model}")
        start = time.time()

        # Save raw request for logging
        raw_request = {
            "model": model,
            "messages": messages,
            **kwargs
        }

        print("Sending request to OpenAI/Azure OpenAI...")

        # ---- Real API call ----
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )

        # Compute latency
        latency_ms = (time.time() - start) * 1000

        # Extract output text
        output_text = resp.choices[0].message.content

        # Extract token usage (OpenAI SDK v1 uses object attributes)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
        else:
            input_tokens = output_tokens = total_tokens = None

        # ---- Log to SQLite ----
        self.logger.log(
            timestamp=str(datetime.now()),
            model=model,
            temperature=kwargs.get("temperature"),
            input_text=str(messages),
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            raw_request=raw_request,
            raw_response=resp.model_dump(),  # pydantic BaseModel
            latency_ms=latency_ms
        )

        # Return normal OpenAI response
        return resp
