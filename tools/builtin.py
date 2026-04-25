import os
import textwrap
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional
from contextlib import redirect_stdout
import io

from memory.vector_store import VectorStore


@dataclass
class ToolResult:
    output: str
    data: Optional[Any] = None


class BaseTool:
    name: str = "base"
    description: str = ""

    def run(self, **kwargs) -> ToolResult:  # pragma: no cover - interface
        raise NotImplementedError


class PythonExecTool(BaseTool):
    name = "python_exec"
    description = "Execute short Python code snippets in a sandboxed scope"

    def run(self, code: str) -> ToolResult:
        buffer = io.StringIO()
        local_scope: Dict[str, Any] = {}
        try:
            with redirect_stdout(buffer):
                exec(code, {}, local_scope)
            result_obj = local_scope.get("result")
            output_text = buffer.getvalue()
            if result_obj is not None:
                output_text += f"\nresult={result_obj!r}"
            return ToolResult(output=output_text.strip(), data=result_obj)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc(limit=1)
            return ToolResult(output=f"error: {e}\n{tb}")


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read text content from a file path"

    def run(self, path: str, start: int = 0, end: Optional[int] = None) -> ToolResult:
        if not os.path.exists(path):
            return ToolResult(output=f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        snippet = data[start:end]
        return ToolResult(output=snippet, data=data)


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write text content to a file path"

    def run(self, path: str, content: str) -> ToolResult:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResult(output=f"wrote {len(content)} chars to {path}")


class AppendNoteTool(BaseTool):
    name = "append_note"
    description = "Append a note to episodic memory file"

    def __init__(self, memory_dir: str, vector_store: Optional[VectorStore] = None):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        self.vector_store = vector_store

    def run(self, note: str, session: str = "default") -> ToolResult:
        path = os.path.join(self.memory_dir, f"{session}.notes.txt")
        with open(path, "a", encoding="utf-8") as f:
            f.write(note.strip() + "\n")
        if self.vector_store:
            self.vector_store.add(note, metadata={"session": session, "source": "note"})
        return ToolResult(output=f"appended note to {path}")


class SearchMemoryTool(BaseTool):
    name = "search_memory"
    description = "Stub search over episodic notes"

    def __init__(self, memory_dir: str, vector_store: Optional[VectorStore] = None):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        self.vector_store = vector_store

    def run(self, query: str, session: str = "default", top_k: int = 3) -> ToolResult:
        # vector search if available
        if self.vector_store and self.vector_store.items:
            hits = self.vector_store.search(query, top_k=top_k)
            texts = [self.vector_store.get_text(vid) for vid, _ in hits]
            return ToolResult(output=" | ".join(texts) if texts else "no hits", data=texts)

        path = os.path.join(self.memory_dir, f"{session}.notes.txt")
        if not os.path.exists(path):
            return ToolResult(output="no memory yet", data=[])
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        matches = [ln.strip() for ln in lines if query.lower() in ln.lower()]
        top = matches[:top_k]
        return ToolResult(output="; ".join(top) if top else "no hits", data=top)


def get_builtin_tools(memory_dir: str, vector_store: Optional[VectorStore] = None) -> Dict[str, BaseTool]:
    tools: Dict[str, BaseTool] = {
        PythonExecTool.name: PythonExecTool(),
        ReadFileTool.name: ReadFileTool(),
        WriteFileTool.name: WriteFileTool(),
        AppendNoteTool.name: AppendNoteTool(memory_dir, vector_store=vector_store),
        SearchMemoryTool.name: SearchMemoryTool(memory_dir, vector_store=vector_store),
    }
    return tools
