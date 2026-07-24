import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "ingest" / "05_load_clickhouse.py"
SPEC = importlib.util.spec_from_file_location("load_clickhouse", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ensure_schema_sends_one_http_command_per_create_table():
    class Client:
        def __init__(self):
            self.commands = []

        def command(self, sql):
            self.commands.append(sql)

    client = Client()
    MODULE.ensure_schema(client, Path(__file__).parents[1] / "schema.sql")

    assert len(client.commands) == 2
    assert client.commands[0].startswith("CREATE TABLE IF NOT EXISTS clips_xclip")
    assert client.commands[1].startswith("CREATE TABLE IF NOT EXISTS clips_siglip2")
