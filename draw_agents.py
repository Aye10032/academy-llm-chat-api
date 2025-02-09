from langchain_core.runnables.graph import MermaidDrawMethod

from llm.core.agent import MainAgent, KnowledgeManageAgent, OptimizerAgent
from llm.core.model import load_gpt4o_mini


def main() -> None:
    llm = load_gpt4o_mini()

    main_agent = MainAgent(llm=llm).build()
    main_agent.get_graph().draw_mermaid_png(output_file_path='test/images/main.png', draw_method=MermaidDrawMethod.PYPPETEER)

    searcher = KnowledgeManageAgent(llm=llm, use_web=True).build()
    searcher.get_graph().draw_mermaid_png(output_file_path='test/images/searcher.png', draw_method=MermaidDrawMethod.PYPPETEER)

    optimizer = OptimizerAgent(llm=llm).build()
    optimizer.get_graph().draw_mermaid_png(output_file_path='test/images/optimizer.png', draw_method=MermaidDrawMethod.PYPPETEER)


if __name__ == '__main__':
    main()
