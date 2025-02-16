import os
from app.models import QwenModel
from smolagents import DuckDuckGoSearchTool, ToolCallingAgent, FinalAnswerTool

model = QwenModel(
    model_id="qwen-max-latest",
    token=os.environ.get("QWEN_TOKEN"),
)
agent = ToolCallingAgent(tools=[DuckDuckGoSearchTool(), FinalAnswerTool()], model=model)
agent.run("GDP of Vietnam in 2024.")
