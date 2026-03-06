import logging
import json
import time
from typing import Dict, List, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from cua_agents.v1.agents.prompts import MCP_WORKER_SYSTEM_PROMPT

logger = logging.getLogger("cortex.mcp_agent")

class MCPAgent:
    """A looping agent that uses MCP tools with a step budget."""

    def __init__(self, llm, budget: int = 10):
        self.llm = llm
        self.budget = budget

    async def execute(self, instruction: str, tm, openai_tools: List[Dict]) -> str:
        """Execute the instruction, looping through tool calls as needed."""
        messages = [
            SystemMessage(content=MCP_WORKER_SYSTEM_PROMPT),
            HumanMessage(content=instruction),
        ]
        
        llm_with_tools = self.llm.bind(tools=openai_tools)
        
        step = 0
        final_response = ""
        
        while step < self.budget:
            logger.info(f"MCP Step {step + 1}/{self.budget}")
            
            # Invoke LLM
            # Note: We use ainvoke for async execution
            response = await llm_with_tools.ainvoke(messages)
            
            # Add assistant response to history
            messages.append(response)
            
            # Check for tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tc in response.tool_calls:
                    func_name = tc["name"]
                    func_args = tc["args"]
                    call_id = tc["id"]
                    
                    if "__" in func_name:
                        server, tool_name = func_name.split("__", 1)
                    else:
                        tool_result = f"Error: Invalid tool format: {func_name}"
                        messages.append(ToolMessage(content=tool_result, tool_call_id=call_id))
                        continue
                        
                    logger.info("🔧 Calling %s.%s(%s)", server, tool_name, json.dumps(func_args, default=str)[:200])
                    print(f"[{time.strftime('%H:%M:%S')}]   └─ 🔧 {server}.{tool_name}", flush=True)
                    
                    try:
                        result = await tm.call_tool(server, tool_name, func_args)
                        tool_result = str(result)
                    except Exception as e:
                        tool_result = f"Error: {e}"
                        
                    messages.append(ToolMessage(content=tool_result, tool_call_id=call_id))
                
                step += 1
            else:
                # No tool calls, this is the final answer
                final_response = response.content
                break
        
        if not final_response:
            final_response = "Budget exhausted. Last response: " + (response.content if response.content else "Pending tool calls.")
            
        return final_response
