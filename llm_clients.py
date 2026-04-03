"""
DeepAssist - LLM 클라이언트
Ollama, Gemini API: 단순 채팅 모드용 클라이언트
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Generator, List

import requests

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


class BaseLLMClient(ABC):
    """LLM Provider 공통 인터페이스"""

    @abstractmethod
    def check_connection(self) -> tuple[bool, str]:
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict],
        tools: List[Dict] = None,
        enable_thinking: bool = False,
        stream_to_terminal: bool = False,
    ) -> Dict:
        pass

    def stream_chat(
        self,
        messages: List[Dict],
        enable_thinking: bool = False,
    ) -> Generator[str, None, None]:
        """UI 스트리밍용 제너레이터. 기본 구현은 한 번에 반환."""
        resp = self.chat(messages=messages, enable_thinking=enable_thinking)
        content = resp.get("message", {}).get("content", "") if "error" not in resp else f"❌ 오류: {resp['error']}"
        yield content


class OllamaClient(BaseLLMClient):
    """Ollama API 클라이언트"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3:8b"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.options = {
            "num_ctx": 32768,
            "temperature": 0.3,
        }

    def check_connection(self) -> tuple[bool, str]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                model_base = self.model.split(":")[0] if ":" in self.model else self.model
                found = any(model_base in m for m in models)
                if found:
                    return True, f"연결됨 (모델: {self.model})"
                return False, f"Ollama 실행 중이나 모델 '{self.model}'을 찾을 수 없습니다.\n사용 가능: {', '.join(models[:10])}"
            return False, "Ollama 응답 오류"
        except requests.ConnectionError:
            return False, "Ollama에 연결할 수 없습니다. ollama serve 실행 여부를 확인하세요."
        except Exception as e:
            return False, f"연결 오류: {e}"

    def chat(
        self,
        messages: List[Dict],
        tools: List[Dict] = None,
        enable_thinking: bool = False,
        stream_to_terminal: bool = False,
    ) -> Dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream_to_terminal,
            "options": self.options,
        }
        if tools:
            payload["tools"] = tools

        if not enable_thinking:
            if messages and messages[0]["role"] == "system":
                if "/nothink" not in messages[0]["content"]:
                    messages[0] = messages[0].copy()
                    messages[0]["content"] += "\n/nothink"

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300,
                stream=stream_to_terminal
            )
            resp.raise_for_status()

            if stream_to_terminal:
                full_message = {"role": "assistant", "content": ""}
                tool_calls = []
                for line in resp.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            msg_chunk = chunk.get("message", {})
                            if "content" in msg_chunk and msg_chunk["content"]:
                                c = msg_chunk["content"]
                                full_message["content"] += c
                                print(c, end="", flush=True)
                            if "tool_calls" in msg_chunk:
                                for tc in msg_chunk["tool_calls"]:
                                    if tc not in tool_calls:
                                        tool_calls.append(tc)
                        except json.JSONDecodeError:
                            pass

                if tool_calls:
                    full_message["tool_calls"] = tool_calls
                print()
                return {"message": full_message}
            else:
                return resp.json()

        except requests.Timeout:
            return {"error": "API 호출 타임아웃 (300초)"}
        except requests.ConnectionError:
            return {"error": "API 연결 끊김"}
        except Exception as e:
            return {"error": f"API 오류: {e}"}


class GeminiClient(BaseLLMClient):
    """Gemini API 클라이언트 (google-genai SDK 사용)"""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        if genai is None:
            raise ImportError("google-genai 패키지가 설치되어 있지 않습니다. 'pip install google-genai' 를 실행해주세요.")
        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=self.api_key)

    def check_connection(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "Gemini API Key가 설정되지 않았습니다."
        try:
            self.client.models.generate_content(
                model=self.model,
                contents="hi",
                config=types.GenerateContentConfig(max_output_tokens=1)
            )
            return True, f"연결됨 (모델: {self.model})"
        except Exception as e:
            return False, f"연결 오류: {e}"

    def chat(
        self,
        messages: List[Dict],
        tools: List[Dict] = None,
        enable_thinking: bool = False,
        stream_to_terminal: bool = False,
    ) -> Dict:
        gemini_contents = []
        system_instruction = None

        for m in messages:
            role = m["role"]
            content = m.get("content", "")

            if role == "system":
                if system_instruction:
                    system_instruction += f"\n{content}"
                else:
                    system_instruction = content
                continue

            parts = []
            if role == "assistant":
                gemini_role = "model"
                if content:
                    parts.append(types.Part.from_text(text=content))
                if "tool_calls" in m:
                    for tc in m["tool_calls"]:
                        try:
                            args = json.loads(tc["function"]["arguments"])
                        except Exception:
                            args = {}
                        parts.append(types.Part.from_function_call(
                            name=tc["function"]["name"],
                            args=args
                        ))
            elif role == "tool":
                gemini_role = "user"
                parts.append(types.Part.from_function_response(
                    name=m.get("name", "unknown_tool"),
                    response={"result": content}
                ))
            else:  # user
                gemini_role = "user"
                if content:
                    parts.append(types.Part.from_text(text=content))

            if parts:
                gemini_contents.append(types.Content(role=gemini_role, parts=parts))

        config_kwargs = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if tools:
            func_decls = []
            for t in tools:
                if t.get("type") == "function":
                    func = t["function"]
                    func_decls.append(types.FunctionDeclaration(
                        name=func["name"],
                        description=func.get("description", ""),
                        parameters=func.get("parameters", {})
                    ))
            if func_decls:
                config_kwargs["tools"] = [types.Tool(function_declarations=func_decls)]

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        try:
            if stream_to_terminal:
                response = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=gemini_contents,
                    config=config
                )

                full_message = {"role": "assistant", "content": ""}
                tool_calls = []

                for chunk in response:
                    if chunk.text:
                        full_message["content"] += chunk.text
                        print(chunk.text, end="", flush=True)

                    if chunk.function_calls:
                        for fc in chunk.function_calls:
                            tc = {
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {
                                    "name": fc.name,
                                    "arguments": json.dumps(fc.args) if fc.args else "{}"
                                }
                            }
                            tool_calls.append(tc)

                if tool_calls:
                    full_message["tool_calls"] = tool_calls
                print()
                return {"message": full_message}
            else:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=gemini_contents,
                    config=config
                )

                full_message = {"role": "assistant", "content": response.text or ""}
                if response.function_calls:
                    tool_calls = []
                    for fc in response.function_calls:
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": json.dumps(fc.args) if fc.args else "{}"
                            }
                        })
                    full_message["tool_calls"] = tool_calls

                return {"message": full_message}

        except Exception as e:
            return {"error": f"API 오류: {str(e)}"}



