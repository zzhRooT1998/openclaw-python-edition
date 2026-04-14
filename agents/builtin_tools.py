class Tool:
    pass
class ReadTool(Tool):
    pass

class WriteTool(Tool):
    pass

class EditTool(Tool):
    pass
class ExecTool(Tool):
    pass

class ListTool(Tool):
    pass

class GrepTool(Tool):
    pass

class MemorySearchTool(Tool):
    pass

class MemoryGetTool(Tool):
    pass
class MemorySaveTool(Tool):
    pass

class SessionSpawnTool(Tool):
    pass

BUILTIN_TOOLS = [ReadTool, WriteTool, EditTool, ExecTool, ListTool, GrepTool, MemorySearchTool, MemoryGetTool, MemorySaveTool, SessionSpawnTool]