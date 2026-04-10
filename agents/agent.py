from typing import Any

class ToolPolicy:
    pass
class Sandbox:
    pass
class ApprovalConfig:
    pass
class ApprovalHandler:
    pass
class ReasoningLevel:
    pass



class AgentConfig:
    provider: str   #例如openai、anthropic、google等
    model: str      #与provider匹配，例如gpt-5.4、claude-opus-4.6等
    apiKey: str
    baseUrl: str
    tools: list[Any]
    toolPolicy: ToolPolicy
    agent_id: str
    sandbox: Sandbox
    approval: ApprovalConfig
    approval_handler: ApprovalHandler
    reasoning_level: ReasoningLevel   #推理级别
    max_turn: int     #最大循环次数
    session_store_dir: str  #会话存储路径
    work_dir: str      #工作路径
    memory_dir: str    #记忆存储路径
    enable_memory: bool #记忆存储开关
    enable_context: bool #上下文开关
    enable_skills: bool #skill开关
    enable_heartbeat: bool #是否启用主动唤醒
    heartbeat_interval: int #心跳检测间隔
    context_token_size: int #上下文窗口大小
    max_concurrent_runs: int #最大并发数



class RunResult:
    run_id: str
    text: str   #最终文本
    turns: int  #总循环次
    tool_calls: int #工具调用次数
    skill_triggered: bool #是否触发了skill
    triggered_skills: list[str]  #触发的技能
    memories_used: int #使用的memory条数

class StreamFunction:
    pass

class Agent:
    stream_function: StreamFunction



