from lida import Manager, llm, TextGenerationConfig
from dotenv import load_dotenv
import os

load_dotenv()

lida = Manager(text_gen=llm("openai"))
textgen_config = TextGenerationConfig(
    n=1, temperature=0.5, model="gpt-3.5-turbo-16k", use_cache=True
)

summary = lida.summarize(
    "Titanic-Dataset.csv", summary_method=llm, textgen_config=textgen_config
)  # generate data summary

# print(summary)
goals = lida.goals(
    summary,
    n=5,
    persona="data analytics with an experties of 10+ years",
    textgen_config=textgen_config,
)  # generate goals


# generate charts (generate and execute visualization code)
charts = lida.visualize(
    summary=summary,
    goal=goals[0],
    library="seaborn",
    textgen_config=textgen_config,
)  # seaborn, ggplot ..

# print(charts[0].code)

# Natural language based visualization refinement
code = charts[0].code
recommended_charts = lida.recommend(
    code=code, summary=summary, n=2, textgen_config=textgen_config
)
print(f"Recommended {len(recommended_charts)} charts")
for chart in recommended_charts:
    print("*" * 10, chart.code)
