from fastapi import FastAPI
from fastapi.responses import StreamingResponse, RedirectResponse
from pydantic import BaseModel
import json
import asyncio
from typing import AsyncIterator, Dict, Any, List, Optional
import httpx
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


app = FastAPI(title="Azure OpenAI Streaming API")

class ChatRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o-mini"
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
GITHUB_OAUTH_URL = os.getenv("GITHUB_OAUTH_URL")
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

# Tool definitions
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit for temperature"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# Tool implementations
async def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
    """Mock weather function - replace with actual weather API call"""
    await asyncio.sleep(0.5)  # Simulate API delay
    return f"The weather in {location} is 72°{unit[0].upper()} and sunny."

async def get_current_time() -> str:
    """Get current time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Tool function mapping
TOOL_FUNCTIONS = {
    "get_current_weather": get_current_weather,
    "get_current_time": get_current_time
}

async def execute_tool_call(tool_call: Dict[str, Any]) -> str:
    """Execute a tool call and return the result"""
    function_name = tool_call["function"]["name"]
    function_args = json.loads(tool_call["function"]["arguments"])
    
    print(f"🔧 EXECUTING TOOL: {function_name}")
    print(f"📋 Tool Arguments: {function_args}")
    
    if function_name in TOOL_FUNCTIONS:
        try:
            result = await TOOL_FUNCTIONS[function_name](**function_args)
            print(f"✅ Tool Result: {result}")
            return str(result)
        except Exception as e:
            error_msg = f"Error executing {function_name}: {str(e)}"
            print(f"❌ Tool Error: {error_msg}")
            return error_msg
    else:
        error_msg = f"Unknown function: {function_name}"
        print(f"❌ Tool Error: {error_msg}")
        return error_msg

async def stream_azure_openai_response(
    messages: List[Dict[str, str]], 
    model: str,
    max_tokens: int,
    temperature: float
) -> AsyncIterator[str]:
    """Stream response from Azure OpenAI with tool call support"""
    
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        print("❌ Azure OpenAI credentials not configured")
        yield f"data: {json.dumps({'error': 'Azure OpenAI credentials not configured'})}\n\n"
        return
    
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{model}/chat/completions"
    print(f"🌐 Making request to: {url}")
    
    headers = {
        "api-key": AZURE_OPENAI_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Initial request payload
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "tools": AVAILABLE_TOOLS,
        "tool_choice": "auto"
    }
    
    print(f"🔧 Available tools: {[tool['function']['name'] for tool in AVAILABLE_TOOLS]}")
    print(f"📤 Sending {len(messages)} messages to OpenAI")
    
    params = {"api-version": AZURE_OPENAI_API_VERSION}
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # First request to check for tool calls
            async with client.stream(
                "POST", url, json=payload, headers=headers, params=params
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    print(f"❌ Azure OpenAI API error: {response.status_code} - {error_text.decode()}")
                    yield f"data: {json.dumps({'error': f'Azure OpenAI API error: {response.status_code} - {error_text.decode()}'})}\n\n"
                    return
                
                print("📡 Streaming response from OpenAI...")
                
                # Collect the complete response to check for tool calls
                complete_response = ""
                tool_calls = []
                current_tool_call = None
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        if data_str.strip() == "[DONE]":
                            print("✅ OpenAI stream completed")
                            break
                            
                        try:
                            data = json.loads(data_str)
                            
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                
                                # Handle content streaming
                                if "content" in delta and delta["content"]:
                                    content = delta["content"]
                                    complete_response += content
                                    yield f"data: {json.dumps({'content': content})}\n\n"
                                
                                # Handle tool calls
                                if "tool_calls" in delta:
                                    if not tool_calls:
                                        print("🔧 OpenAI is requesting tool calls!")
                                    
                                    for tool_call_delta in delta["tool_calls"]:
                                        index = tool_call_delta.get("index", 0)
                                        
                                        # Initialize tool call if new
                                        if index >= len(tool_calls):
                                            tool_calls.extend([None] * (index + 1 - len(tool_calls)))
                                        
                                        if tool_calls[index] is None:
                                            tool_calls[index] = {
                                                "id": tool_call_delta.get("id", ""),
                                                "type": "function",
                                                "function": {
                                                    "name": "",
                                                    "arguments": ""
                                                }
                                            }
                                            print(f"🆕 New tool call initiated (index {index})")
                                        
                                        # Update tool call data
                                        if "id" in tool_call_delta:
                                            tool_calls[index]["id"] = tool_call_delta["id"]
                                        
                                        if "function" in tool_call_delta:
                                            function_delta = tool_call_delta["function"]
                                            if "name" in function_delta:
                                                tool_calls[index]["function"]["name"] += function_delta["name"]
                                                if function_delta["name"]:  # Only print if there's actual content
                                                    print(f"🏷️ Tool name fragment: '{function_delta['name']}'")
                                            if "arguments" in function_delta:
                                                tool_calls[index]["function"]["arguments"] += function_delta["arguments"]
                                                if function_delta["arguments"]:  # Only print if there's actual content
                                                    print(f"📝 Arguments fragment: '{function_delta['arguments']}'")
                                
                        except json.JSONDecodeError:
                            continue
                
                # If tool calls were requested, execute them and continue the conversation
                if tool_calls and any(tc for tc in tool_calls if tc is not None):
                    valid_tool_calls = [tc for tc in tool_calls if tc is not None]
                    print(f"\n🔧 TOOL CALLS DETECTED: {len(valid_tool_calls)} tools requested")
                    
                    for i, tool_call in enumerate(valid_tool_calls):
                        print(f"Tool {i+1}: {tool_call['function']['name']} with args: {tool_call['function']['arguments']}")
                    
                    yield f"data: {json.dumps({'tool_calls_detected': True})}\n\n"
                    
                    # Execute tool calls
                    tool_messages = []
                    for tool_call in valid_tool_calls:
                        print(f"\n🏃 Executing tool: {tool_call['function']['name']}")
                        yield f"data: {json.dumps({'executing_tool': tool_call['function']['name']})}\n\n"
                        result = await execute_tool_call(tool_call)
                        tool_messages.append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "content": result
                        })
                    
                    print(f"\n✅ All {len(valid_tool_calls)} tools executed successfully")
                    
                    # Add assistant message with tool calls
                    messages.append({
                        "role": "assistant",
                        "content": complete_response or "",
                        "tool_calls": valid_tool_calls
                    })
                    
                    # Add tool responses
                    messages.extend(tool_messages)
                    
                    print(f"📤 Sending tool results back to OpenAI ({len(messages)} total messages)")
                    
                    # Make second request with tool results
                    payload["messages"] = messages
                    
                    async with client.stream(
                        "POST", url, json=payload, headers=headers, params=params
                    ) as second_response:
                        if second_response.status_code != 200:
                            error_text = await second_response.aread()
                            print(f"❌ Azure OpenAI API error on tool response: {second_response.status_code} - {error_text.decode()}")
                            yield f"data: {json.dumps({'error': f'Azure OpenAI API error on tool response: {second_response.status_code} - {error_text.decode()}'})}\n\n"
                            return
                        
                        print("📡 Streaming final response with tool results...")
                        yield f"data: {json.dumps({'tool_results_processed': True})}\n\n"
                        
                        async for line in second_response.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:]
                                
                                if data_str.strip() == "[DONE]":
                                    print("✅ Final response stream completed")
                                    break
                                    
                                try:
                                    data = json.loads(data_str)
                                    
                                    if "choices" in data and len(data["choices"]) > 0:
                                        delta = data["choices"][0].get("delta", {})
                                        
                                        if "content" in delta and delta["content"]:
                                            content = delta["content"]
                                            yield f"data: {json.dumps({'content': content})}\n\n"
                                            
                                except json.JSONDecodeError:
                                    continue
                else:
                    print("ℹ️ No tool calls requested by OpenAI")
                
        except Exception as e:
            print(f"❌ Request failed: {str(e)}")
            yield f"data: {json.dumps({'error': f'Request failed: {str(e)}'})}\n\n"
    
    print("🏁 Stream completed\n")
    yield "data: [DONE]\n\n"

@app.get("/github-login")
async def github_login():
    """
    Initiates the GitHub login process.
    """
    # Redirect the user to the GitHub OAuth authorization URL
    return RedirectResponse(url=f"{GITHUB_OAUTH_URL}{GITHUB_CLIENT_ID}", status_code=302)

@app.get("/github-callback")
async def github_callback(code: str):
    """
    Handles the GitHub OAuth callback.
    """
    # Exchange the code for an access token

    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code
            }
        )

    if token_response.status_code == 200:
        token_data = token_response.json()
        access_token = token_data.get("access_token")

        if access_token:
            # Use the access token to fetch user information
            async with httpx.AsyncClient() as client:
                user_response = await client.get(
                        "https://api.github.com/user",
                        headers={"Authorization": f"Bearer {access_token}",
                                 "Accept": "application/json"}
                    )

            if user_response.status_code == 200:
                user_data = user_response.json()
                return {"user": user_data}
            else:
                return {"error": "Failed to fetch user information"}
        else:
            return {"error": "Failed to obtain access token"}
    else:
        return {"error": "Failed to exchange code for access token"}

@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """
    Stream chat completions from Azure OpenAI with tool call support
    
    The response will be streamed as Server-Sent Events (SSE) with the following format:
    - data: {"content": "chunk of text"} - For regular content
    - data: {"tool_calls_detected": true} - When tool calls are detected
    - data: {"executing_tool": "function_name"} - When executing a tool
    - data: {"tool_results_processed": true} - When tool results are processed
    - data: {"error": "error message"} - For errors
    - data: [DONE] - When streaming is complete
    """
    
    print(f"\n🚀 NEW CHAT REQUEST")
    print(f"📝 User Prompt: {request.prompt}")
    print(f"🤖 Model: {request.model}")
    print(f"🌡️ Temperature: {request.temperature}")
    print(f"📏 Max Tokens: {request.max_tokens}")
    
    messages = [{"role": "user", "content": request.prompt}]
    
    return StreamingResponse(
        stream_azure_openai_response(
            messages=messages,
            model=request.model,
            max_tokens=request.max_tokens or 1000,
            temperature=request.temperature or 0.7
        ),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "azure_openai_configured": bool(AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY),
        "available_tools": [tool["function"]["name"] for tool in AVAILABLE_TOOLS]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)