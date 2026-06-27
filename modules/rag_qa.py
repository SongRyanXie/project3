from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from config import Config
from typing import List  # 增强向后版本兼容性
import random

# 定义 Mock 嵌入类，确保在无第三方 Embedding 密钥时程序仍能正常初始化运行
class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[random.uniform(-1, 1) for _ in range(1536)] for _ in texts]
    def embed_query(self, text: str) -> List[float]:
        return [random.uniform(-1, 1) for _ in range(1536)]

def format_docs(docs: List[Document]) -> str:
    """提取检索到的文档 page_content 并组合成文本段落"""
    return "\n\n".join(doc.page_content for doc in docs)

class RecruitingQA:
    def __init__(self):
        # 实例化 DeepSeek Chat 模型
        self.llm = ChatOpenAI(
            model=Config.DEEPSEEK_MODEL,
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL,
            temperature=0.2
        )
        
        # 选用合适的 Embedding 模型
        if Config.OPENAI_API_KEY:
            self.embeddings = OpenAIEmbeddings(
                api_key=Config.OPENAI_API_KEY,
                model=Config.EMBEDDING_MODEL
            )
        else:
            # 降级使用 Mock 嵌入，防止由于未配置 Embedding Key 导致程序直接报错
            self.embeddings = MockEmbeddings()

        self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        # 初始化样本企业问答知识库
        samples = [
            "Q: 公司的试用期是多久？福利怎么样？ A: 公司的法定试用期一般为3个月。试用期期间薪资按100%全额发放，并从入职第一天起足额缴纳五险一金。",
            "Q: 弹性工作制具体是怎样的？ A: 我们的核心工作时间是 10:00 - 17:00，其余时间实行弹性考勤，可以自由安排错峰上下班。",
            "Q: 晋升机制和调薪机会如何？ A: 公司每年提供两次（6月和12月）绩效评估及晋升提名机会。调薪取决于个人业绩与团队贡献表现。",
            "Q: 年终奖如何计算和发放？ A: 年终奖依据公司当年的整体业绩及员工的年度绩效等级进行评定，基准范围在2至4个月年薪，通常于春节前一次性随工资发放。"
        ]
        docs = [Document(page_content=text) for text in samples]
        self.vector_store.add_documents(docs)

    def answer_question(self, question: str) -> str:
        # 配置检索层获取最相关的 2 条记录
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        
        prompt = ChatPromptTemplate.from_template("""
        你是一位专业的企业人力资源（HR）助手。请根据以下参考信息，专业、礼貌地回答候选人关于招聘的问题。
        如果提供的参考信息不足以回答候选人的问题，请明确告知对方你目前无法解答，并表示后续会联系人工 HR 确认。

        参考信息:
        {context}

        候选人提问:
        {question}

        专业回答:
        """)

        # 引入 format_docs 处理机制，将 Document 转换为大模型更易理解的文本
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(question)