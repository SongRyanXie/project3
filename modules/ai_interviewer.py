from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from config import Config

class AIInterviewer:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.DEEPSEEK_MODEL,
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL,
            temperature=0.7  # 较高温度确保追问灵活性
        )
        # 用作多用户内存 Session 的缓存结构: { session_id: { "candidate_name": str, "job_title": str, "history": [] } }
        self.sessions = {}

    def start_session(self, session_id: str, candidate_name: str, job_title: str) -> str:
        # 面试开场白生成
        welcome_prompt = (
            f"你好，{candidate_name}！我是今天的 AI 面试官。感谢参加我们公司关于【{job_title}】岗位的面试。"
            "我们将通过简短的交流了解你的过往经历和技术积累。首先，请用 1-2 分钟做一个简短的自我介绍，"
            "并谈谈你为什么对该岗位感兴趣。"
        )
        
        self.sessions[session_id] = {
            "candidate_name": candidate_name,
            "job_title": job_title,
            "history": [AIMessage(content=welcome_prompt)]
        }
        return welcome_prompt

    def chat(self, session_id: str, user_message: str) -> str:
        if session_id not in self.sessions:
            return "会话未创建或已过期，请重新启动面试会话。"
        
        session = self.sessions[session_id]
        # 追加候选人的回答
        session["history"].append(HumanMessage(content=user_message))

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "你是一位资深、严谨而态度温和的技术面试官。当前正在进行【{job_title}】岗位的面试。 "
                "你需要仔细阅读聊天历史，根据候选人的上一次回答进行针对性地深入技术追问或过渡至下一环节问题。 "
                "一次仅提出一个问题，保持语气职业，不表现出过度敷衍或过分恭维。不要一次提多问。"
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        chain = prompt | self.llm
        
        # 排除最新一条尚未输入系统的对话作为历史，并调用模型
        response = chain.invoke({
            "job_title": session["job_title"],
            "chat_history": session["history"][:-1],
            "input": user_message
        })

        # 保存 AI 面试官提问
        session["history"].append(AIMessage(content=response.content))
        return response.content

    def evaluate_session(self, session_id: str) -> dict:
        if session_id not in self.sessions:
            return {"error": "未找到有效的面试记录，无法完成评估。"}
        
        session = self.sessions[session_id]
        history_text = ""
        for msg in session["history"]:
            role = "AI面试官" if isinstance(msg, AIMessage) else "候选人"
            history_text += f"{role}: {msg.content}\n"

        eval_prompt = ChatPromptTemplate.from_template("""
        你是一位资深的技术招聘总监。请阅读以下【{job_title}】岗位候选人【{candidate_name}】的模拟面试实录，对该候选人的面试表现、技能深度及沟通能力做出客观详细的考评。

        面试实录:
        {history_text}

        请严格根据以上内容评定并生成 JSON，不要附加说明、不要带有 markdown 反单引号前缀：
        {{
            "comprehensive_score": 综合考核得分(0-100),
            "pros": ["优势/亮点1", "优势/亮点2"],
            "cons": ["短板/改进点1", "短板/改进点2"],
            "summary": "最终客观评价总结",
            "decision": "录用决策建议，例如（强烈推荐 / 建议录用 / 待定 / 不予考虑）"
        }}
        """)

        parser = JsonOutputParser()
        chain = eval_prompt | self.llm | parser

        try:
            evaluation = chain.invoke({
                "job_title": session["job_title"],
                "candidate_name": session["candidate_name"],
                "history_text": history_text
            })
            return evaluation
        except Exception as e:
            return {
                "comprehensive_score": 0,
                "pros": [],
                "cons": [],
                "summary": f"生成最终面试评估报告出现异常: {str(e)}",
                "decision": "无法决策"
            }