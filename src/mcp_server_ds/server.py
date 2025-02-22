from enum import Enum
import logging
from typing import Optional, List

## import mcp server
from mcp.server.models import InitializationOptions
from mcp.types import (
    TextContent,
    Tool,
    Resource,
    INTERNAL_ERROR,
    Prompt,
    PromptArgument,
    EmbeddedResource,
    GetPromptResult,
    PromptMessage,
)
from mcp.server import NotificationOptions, Server
from mcp.shared.exceptions import McpError
from pydantic import AnyUrl
import mcp.server.stdio
from pydantic import BaseModel

## import common data analysis libraries
import pandas as pd
import numpy as np
import scipy
import sklearn
import statsmodels.api as sm
from io import StringIO
import sys


logger = logging.getLogger(__name__)
logger.info("Starting mini data science exploration server")

### Prompt templates
class DataExplorationPrompts(str, Enum):
    EXPLORE_DATA = "explore-data"

class PromptArgs(str, Enum):
    CSV_PATH = "csv_path"
    TOPIC = "topic"

PROMPT_TEMPLATE = """
You are a professional Data Scientist tasked with performing exploratory data analysis on a dataset. Your goal is to provide insightful analysis while ensuring stability and manageable result sizes.

First, load the CSV file from the following path:

<csv_path>
{csv_path}
</csv_path>

Your analysis should focus on the following topic:

<analysis_topic>
{topic}
</analysis_topic>

You have access to the following tools for your analysis:
1. load_csv: Use this to load the CSV file.
2. run-script: Use this to execute Python scripts on the MCP server.

Please follow these steps carefully:

1. Load the CSV file using the load_csv tool.

2. Explore the dataset. Provide a brief summary of its structure, including the number of rows, columns, and data types. Wrap your exploration process in <dataset_exploration> tags, including:
   - List of key statistics about the dataset
   - Potential challenges you foresee in analyzing this data

3. Wrap your thought process in <analysis_planning> tags:
   Analyze the dataset size and complexity:
   - How many rows and columns does it have?
   - Are there any potential computational challenges based on the data types or volume?
   - What kind of questions would be appropriate given the dataset's characteristics and the analysis topic?
   - How can we ensure that our questions won't result in excessively large outputs?

   Based on this analysis:
   - List 10 potential questions related to the analysis topic
   - Evaluate each question against the following criteria:
     * Directly related to the analysis topic
     * Can be answered with reasonable computational effort
     * Will produce manageable result sizes
     * Provides meaningful insights into the data
   - Select the top 5 questions that best meet all criteria

4. List the 5 questions you've selected, ensuring they meet the criteria outlined above.

5. For each question, follow these steps:
   a. Wrap your thought process in <analysis_planning> tags:
      - How can I structure the Python script to efficiently answer this question?
      - What data preprocessing steps are necessary?
      - How can I limit the output size to ensure stability?
      - What type of visualization would best represent the results?
      - Outline the main steps the script will follow
   
   b. Write a Python script to answer the question. Include comments explaining your approach and any measures taken to limit output size.
   
   c. Use the run_script tool to execute your Python script on the MCP server.
   
   d. Render the results returned by the run-script tool as a chart using plotly.js (prefer loading from cdnjs.cloudflare.com). Do not use react or recharts, and do not read the original CSV file directly. Provide the plotly.js code to generate the chart.

6. After completing the analysis for all 5 questions, provide a brief summary of your findings and any overarching insights gained from the data.

Remember to prioritize stability and manageability in your analysis. If at any point you encounter potential issues with large result sets, adjust your approach accordingly.

Please begin your analysis by loading the CSV file and providing an initial exploration of the dataset.
"""


### Data Exploration Tools Description & Schema
class DataExplorationTools(str, Enum):
    LOAD_CSV = "load_csv"
    RUN_SCRIPT = "run_script"


LOAD_CSV_TOOL_DESCRIPTION = """
Load CSV File Tool

Purpose:
Load a local CSV file into a DataFrame.

Usage Notes:
	•	If a df_name is not provided, the tool will automatically assign names sequentially as df_1, df_2, and so on.
"""

class LoadCsv(BaseModel):
    csv_path: str
    df_name: Optional[str] = None



RUN_SCRIPT_TOOL_DESCRIPTION = """
Python Script Execution Tool

Purpose:
Execute Python scripts for specific data analytics tasks.

Allowed Actions
	1.	Print Results: Output will be displayed as the script’s stdout.
	2.	[Optional] Save DataFrames: Store DataFrames in memory for future use by specifying a save_to_memory name.

Prohibited Actions
	1.	Overwriting Original DataFrames: Do not modify existing DataFrames to preserve their integrity for future tasks.
	2.	Creating Charts: Chart generation is not permitted.
"""

class RunScript(BaseModel):
    script: str
    save_to_memory: Optional[List[str]] = None


### Python (Pandas, NumPy, SciPy) Script Runner
class ScriptRunner:
    def __init__(self):
        self.data = {}
        self.df_count = 0
        self.notes: list[str] = []

    def load_csv(self, csv_path: str, df_name:str = None):
        self.df_count += 1
        if not df_name:
            df_name = f"df_{self.df_count}"
        try:
            self.data[df_name] = pd.read_csv(csv_path)
            self.notes.append(f"Successfully loaded CSV into dataframe '{df_name}'")
            return [
                TextContent(type="text", text=f"Successfully loaded CSV into dataframe '{df_name}'")
            ]
        except Exception as e:
            raise McpError(
                INTERNAL_ERROR, f"Error loading CSV: {str(e)}"
            ) from e

    def safe_eval(self, script: str, save_to_memory: Optional[List[str]] = None):
        """safely run a script, return the result if valid, otherwise return the error message"""
        # first extract dataframes from the self.data
        local_dict = {
            **{df_name: df for df_name, df in self.data.items()},
        }
        # execute the script and return the result and if there is error, return the error message
        try:
            stdout_capture = StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout_capture
            self.notes.append(f"Running script: \n{script}")
            # pylint: disable=exec-used
            exec(script, \
                {'pd': pd, 'np': np, 'scipy': scipy, 'sklearn': sklearn, 'statsmodels': sm}, \
                local_dict)
            std_out_script = stdout_capture.getvalue()
        except Exception as e:
            raise McpError(INTERNAL_ERROR, f"Error running script: {str(e)}") from e

        # check if the result is a dataframe
        if save_to_memory:
            for df_name in save_to_memory:
                self.notes.append(f"Saving dataframe '{df_name}' to memory")
                self.data[df_name] = local_dict.get(df_name)

        output = std_out_script if std_out_script else "No output"
        self.notes.append(f"Result: {output}")
        return [
            TextContent(type="text", text=f"print out result: {output}")
        ]

### MCP Server Definition
async def main():
    script_runner = ScriptRunner()
    server = Server("local-mini-ds")

    @server.list_resources()
    async def handle_list_resources() -> list[Resource]:
        logger.debug("Handling list_resources request")
        return [
            Resource(
                uri="data-exploration://notes",
                name="Data Exploration Notes",
                description="Notes generated by the data exploration server",
                mimeType="text/plain",
            )
        ]

    @server.read_resource()
    async def handle_read_resource(uri: AnyUrl) -> str:
        logger.debug(f"Handling read_resource request for URI: {uri}")
        if uri == "data-exploration://notes":
            return "\n".join(script_runner.notes)
        else:
            raise ValueError(f"Unknown resource: {uri}")

    @server.list_prompts()
    async def handle_list_prompts() -> list[Prompt]:
        logger.debug("Handling list_prompts request")
        return [
            Prompt(
                name=DataExplorationPrompts.EXPLORE_DATA,
                description="A prompt to explore a csv dataset as a data scientist",
                arguments=[
                    PromptArgument(
                        name=PromptArgs.CSV_PATH,
                        description="The path to the csv file",
                        required=True,
                    ),
                    PromptArgument(
                        name=PromptArgs.TOPIC,
                        description="The topic the data exploration need to focus on",
                        required=False,
                    ),
                ],
            )
        ]

    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
        logger.debug(f"Handling get_prompt request for {name} with args {arguments}")
        if name != DataExplorationPrompts.EXPLORE_DATA:
            logger.error(f"Unknown prompt: {name}")
            raise ValueError(f"Unknown prompt: {name}")

        if not arguments or PromptArgs.CSV_PATH not in arguments:
            logger.error("Missing required argument: csv_path")
            raise ValueError("Missing required argument: csv_path")

        csv_path = arguments[PromptArgs.CSV_PATH]
        topic = arguments.get(PromptArgs.TOPIC)
        prompt = PROMPT_TEMPLATE.format(csv_path=csv_path, topic=topic)

        logger.debug(f"Generated prompt template for csv_path: {csv_path} and topic: {topic}")
        return GetPromptResult(
            description=f"Data exploration template for {topic}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=prompt.strip()),
                )
            ],
        )

    @server.list_tools()
    async def handle_list_tools() -> list[Tool]:
        logger.debug("Handling list_tools request")
        return [
            Tool(
                name = DataExplorationTools.LOAD_CSV,
                description = LOAD_CSV_TOOL_DESCRIPTION,
                inputSchema = LoadCsv.model_json_schema(),
            ),
            Tool(
                name=DataExplorationTools.RUN_SCRIPT,
                description=RUN_SCRIPT_TOOL_DESCRIPTION,
                inputSchema=RunScript.model_json_schema(),
            )
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> list[TextContent | EmbeddedResource]:
        logger.debug(f"Handling call_tool request for {name} with args {arguments}")
        if name == DataExplorationTools.LOAD_CSV:
            csv_path = arguments.get("csv_path")
            df_name = arguments.get("df_name")
            return script_runner.load_csv(csv_path, df_name)
        elif name == DataExplorationTools.RUN_SCRIPT:
            script = arguments.get("script")
            df_name = arguments.get("df_name")
            return script_runner.safe_eval(script, df_name)
        else:
            raise McpError(INTERNAL_ERROR, f"Unknown tool: {name}")
        return None

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.debug("Server running with stdio transport")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="data-exploration-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
