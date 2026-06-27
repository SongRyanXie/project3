from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import Config

# 定义输出评分和分析报告的 Pydantic 模型
class ResumeEvaluation(BaseModel):
    score: int = Field(description="候选人综合匹配评分，分值区间 0 - 100")
    matched_skills: List[str] = Field(description="候选人简历中匹配岗位要求的核心技术与通用能力列表")
    missing_skills: List[str] = Field(description="候选人缺失的关键核心技能或要求列表")
    suitability_analysis: str = Field(description="针对候选人与岗位契合度做出的多维度定量定性分析")
    recommended_position: str = Field(description="推荐的合适岗位方向或给候选人的发展建议")

class ResumeAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.DEEPSEEK_MODEL,
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL,
            temperature=0.1  # 低温度确保评分与分析的稳定性
        )
        self.parser = JsonOutputParser(pydantic_object=ResumeEvaluation)

    def analyze(self, resume_text: str, job_description: str) -> dict:
        prompt = ChatPromptTemplate.from_template("""
        你是一位经验丰富的人才甄选专家。请仔细比对岗位 JD 和候选人的简历，做出客观的打分和多维度匹配分析。

        岗位职责与要求:
        {job_description}

        候选人简历内容:
        {resume_text}

        请严格按照以下格式指南输出 JSON，不要在结果前、后附加多余的说明性汉字：
        {format_instructions}
        """)

        chain = prompt | self.llm | self.parser
        
        try:
            result = chain.invoke({
                "job_description": job_description,
                "resume_text": resume_text,
                "format_instructions": self.parser.get_format_instructions()
            })
            return result
        except Exception as e:
            # 基础降级异常捕获，保持接口一致性
            return {
                "score": 0,
                "matched_skills": [],
                "missing_skills": [],
                "suitability_analysis": f"深度分析执行失败: {str(e)}",
                "recommended_position": "解析异常"
            }