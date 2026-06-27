import os
import uuid
from flask import Flask, request, jsonify, render_template
from modules.rag_qa import RecruitingQA
from modules.resume_analyzer import ResumeAnalyzer
from modules.ai_interviewer import AIInterviewer
from modules.mcp_agent import MCPAgent

app = Flask(__name__)

# 初始化后台各 AI 服务组件
qa_system = RecruitingQA()
resume_analyzer = ResumeAnalyzer()
ai_interviewer = AIInterviewer()

# 统一注册至 MCP 能力中台
mcp_agent = MCPAgent(
    qa_system=qa_system,
    resume_analyzer=resume_analyzer,
    ai_interviewer=ai_interviewer
)

# 全局异常捕获处理，确保代码异常时客户端获得结构化的错误反馈
@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({
        "error": str(e),
        "status": "fail",
        "message": "中台服务遇到内部异常，请检查后台控制台日志。"
    }), 500


# =====================================================================
# 0. 前台页面渲染路由
# =====================================================================
@app.route('/', methods=['GET'])
def index():
    """渲染并返回 templates/index.html 页面"""
    return render_template('index.html')


# =====================================================================
# 1. 招聘问答系统路由 (Module 3)
# =====================================================================
@app.route('/api/qa', methods=['POST'])
def handle_qa():
    data = request.json or {}
    question = data.get("question")
    if not question:
        return jsonify({"error": "参数 'question' 不能为空"}), 400
    
    answer = qa_system.answer_question(question)
    return jsonify({
        "question": question,
        "answer": answer
    })


# =====================================================================
# 2. 简历筛选评估路由 (Module 4)
# =====================================================================
@app.route('/api/resume/screen', methods=['POST'])
def handle_resume_screen():
    data = request.json or {}
    resume_text = data.get("resume_text")
    job_description = data.get("job_description")
    
    if not resume_text or not job_description:
        return jsonify({"error": "参数 'resume_text' 与 'job_description' 均不能为空"}), 400
    
    analysis = resume_analyzer.analyze(resume_text, job_description)
    return jsonify(analysis)


# =====================================================================
# 3. AI 模拟面试官路由 (Module 5)
# =====================================================================
@app.route('/api/interview/start', methods=['POST'])
def start_interview():
    data = request.json or {}
    candidate_name = data.get("candidate_name", "候选人")
    job_title = data.get("job_title", "通用软件开发工程师")
    
    session_id = str(uuid.uuid4())
    welcome_msg = ai_interviewer.start_session(session_id, candidate_name, job_title)
    
    return jsonify({
        "session_id": session_id,
        "message": welcome_msg
    })

@app.route('/api/interview/chat', methods=['POST'])
def chat_interview():
    data = request.json or {}
    session_id = data.get("session_id")
    user_message = data.get("message")
    
    if not session_id or not user_message:
        return jsonify({"error": "参数 'session_id' 与 'message' 均不能为空"}), 400
    
    response = ai_interviewer.chat(session_id, user_message)
    return jsonify({
        "session_id": session_id,
        "message": response
    })

@app.route('/api/interview/evaluate', methods=['POST'])
def evaluate_interview():
    data = request.json or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "参数 'session_id' 不能为空"}), 400
    
    evaluation = ai_interviewer.evaluate_session(session_id)
    return jsonify(evaluation)


# =====================================================================
# 4. MCP 平台统一中台能力管理 (Module 2)
# =====================================================================
@app.route('/api/mcp/tools', methods=['GET'])
def get_mcp_tools():
    """获取 MCP 服务中注册的可用工具清单与参数规范"""
    tools = mcp_agent.list_tools()
    return jsonify({"tools": tools})

@app.route('/api/mcp/execute', methods=['POST'])
def execute_mcp_tool():
    """动态调用指定的 MCP 协议工具"""
    data = request.json or {}
    tool_name = data.get("tool")
    parameters = data.get("arguments", {})
    
    if not tool_name:
        return jsonify({"error": "参数 'tool' 不能为空"}), 400
    
    result = mcp_agent.execute_tool(tool_name, parameters)
    return jsonify(result)


if __name__ == '__main__':
    # 启动 Flask 服务，支持本地 5000 端口访问
    app.run(host='0.0.0.0', port=5000, debug=True)
    