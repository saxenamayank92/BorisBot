import asyncio
import typer
from .runtime import Runtime

app = typer.Typer()

@app.command()
def main(
    agent_id: str = typer.Option(..., help="The unique ID of the agent"),
    supervisor_url: str = typer.Option(..., help=" The URL of the supervisor"),
):
    """
    Start the agent worker process.
    """
    runtime = Runtime(agent_id=agent_id, supervisor_url=supervisor_url)
    asyncio.run(runtime.start())

if __name__ == "__main__":
    app()
