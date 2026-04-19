import os
from typing import Any, Callable

from agents.agent_event import AgentEvent
from agents.builtin_tools import Tool, BUILTIN_TOOLS
from agents.context_loader import ContextLoader
from agents.heartbeat_manager import HeartbeatManager
from agents.memory_manager import MemoryManager
from agents.run_abort_controller import RunAbortController
from agents.session_manager import SessionManager
from agents.skills_manager import SkillsManager
from system_prompt import SYSTEM_PROMPT


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

class ModelDef:
    pass

class ThinkLevel:
    pass

class AgentConfig:
    provider: str   #例如openai、anthropic、google等
    model: str      #与provider匹配，例如gpt-5.4、claude-opus-4.6等
    model_def: ModelDef
    api_key: str
    base_url: str
    tools: list[Tool]
    tool_policy: ToolPolicy
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

class ModelDef:
    api_key: str
    base_url: str
    model_id: str
    privider: str
    pass
class AllowListManager:
    pass

class ToolResultGuard:
    pass


def install_session_tool_result_guard(sessions):
    pass


class Agent:
    stream_function: StreamFunction
    model_def: ModelDef
    api_key: str
    temperature: float
    reasoning_level: ReasoningLevel
    agent_id: str
    base_system_prompt: str
    tools: list[Tool]
    max_turn: int
    work_dir: str
    tool_policy: ToolPolicy
    approval: ApprovalConfig
    approval_handler: ApprovalHandler
    allow_list:AllowListManager
    context_token_size: int
    sandbox: Sandbox

    #大子系统
    sessions: SessionManager
    memory: MemoryManager
    context: ContextLoader
    skills: SkillsManager
    heartbeat: HeartbeatManager

    enable_memory: bool
    enable_context: bool
    enable_skills: bool
    enable_heartbeat: bool


    run_abort_controller: dict[str, RunAbortController]

    #用户在工具执行期间发送消息入队，会在每次工具执行完后检查，若非空则跳过工具执行。队列中的消息作为下一个user turn处理
    steering_queue: dict[str, list[str]]
    tool_result_guard:  ToolResultGuard

    listerns: set[Callable[[AgentEvent], None]]

    def __init__(self, agent_config: AgentConfig):
        self.agent_id = agent_config.agent_id or 'main'
        self.base_system_prompt = agent_config.base_system_prompt
        provider = agent_config.provider or 'openai'
        model_id = agent_config.model_id or ('gpt-4o' if provider == 'openai' else None)
        self.model_def = ModelDef()
        self.stream_function = agent_config.stream_function or StreamFunction()
        self.base_system_prompt = agent_config.base_system_prompt or SYSTEM_PROMPT

        self.tools = agent_config.tools or BUILTIN_TOOLS
        self.max_turn = agent_config.max_turn or 20
        self.work_dir = (agent_config.work_dir or

                         os.getcwd())
        self.api_key = agent_config.api_key
        self.temperature = agent_config.temperature
        self.reasoning_level = agent_config.reasoning_level or ReasoningLevel.MEDIUM
        self.tool_policy = agent_config.tool_policy
        self.approval = agent_config.approval
        self.approval_handler = agent_config.approval_handler
        self.allow_list = AllowListManager()
        self.context_token_size = agent_config.context_token_size or 20000
        self.sandbox = agent_config.sandbox


        #初始化子系统
        self.sessions = SessionManager(agent_config.session_store_dir)
        self.memory = MemoryManager(agent_config.memory_dir)
        self.context = ContextLoader(self.work_dir)
        self.skills = SkillsManager(self.work_dir)
        self.heartbeat = HeartbeatManager(self.work_dir, agent_config.heartbeat_interval)

        #功能开关
        self.enable_memory = agent_config.enable_memory or True
        self.enable_context = agent_config.enable_context or True
        self.enable_skills = agent_config.enable_skills or True
        self.enable_heartbeat = agent_config.enable_heartbeat or True
        self.tool_result_guard = install_session_tool_result_guard(self.sessions)