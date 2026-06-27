from typing import List, Dict, Any

class MCPAgent:
    """
    轻量级 MCP 模型上下文工具集中管理器。
    将系统中核心功能包装为带有标准 JSON-Schema 的 Tools 供中台消费和调用。
    """
    def __init__(self, qa_system, resume_analyzer, ai_interviewer):
        self.qa_system = qa_system
        self.resume_analyzer = resume_analyzer
        self.ai_interviewer = ai_interviewer
        self.tools = {}
        self._register_core_tools()

    def _register_core_tools(self):
        # 1. 注册招聘 RAG 问答工具
        self.register_tool(
            name="query_recruiting_qa",
            description="查询招聘知识库，帮助回答候选人关于公司试用期、福利等方面的提问。",
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "候选人咨询的具体问题"}
                },
                "required": ["question"]
            },
            handler=lambda args: {"answer": self.qa_system.answer_question(args.get("question"))}
        )

        # 2. 注册简历自动筛选评估工具
        self.register_tool(
            name="analyze_resume",
            description="比对候选人简历与岗位JD，输出综合匹配度评分、缺失技能、匹配点及录用建议。",
            parameters={
                "type": "object",
                "properties": {
                    "resume_text": {"type": "string", "description": "简历原文"},
                    "job_description": {"type": "string", "description": "岗位职责描述"}
                },
                "required": ["resume_text", "job_description"]
            },
            handler=lambda args: self.resume_analyzer.analyze(
                args.get("resume_text"), args.get("job_description")
            )
        )

    def register_tool(self, name: str, description: str, parameters: dict, handler):
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "handler": handler
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """获取所有已向 MCP 系统注册的可用工具列表"""
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
            for tool in self.tools.values()
        ]

    def execute_tool(self, name: str, arguments: dict) -> Dict[str, Any]:
        """运行 MCP 工具并返回统一的数据格式"""
        if name not in self.tools:
            return {"status": "error", "message": f"未向中台注册工具: {name}"}
        try:
            result = self.tools[name]["handler"](arguments)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "message": f"中台工具运行异常: {str(e)}"}