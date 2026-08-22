from model_monitoring.agent import MonitoringAgent
from model_monitoring.rag.retrieval import PolicyRetriever

retriever = PolicyRetriever()
retriever.build_index()  # Setup operation outside the agent.

result = MonitoringAgent(retriever).run("M001", "2026-07")
print(result.recommendation.model_dump(mode="json"))
print([entry.model_dump(mode="json") for entry in result.tool_call_log])