"""notebooks/*.ipynb icin ortak insa+calistir yardimcisi. Notebook'lar bu
depoda interaktif bir Jupyter oturumundan degil, bu fonksiyon araciligiyla
nbclient ile GERCEKTEN calistirilarak uretiliyor - hucre ciktilari gercek
calistirmadan geliyor, elle yazilmiyor."""
import pathlib

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def build_and_execute(cells: list, out_path: pathlib.Path, timeout: int = 1800) -> nbformat.NotebookNode:
    """cells: [("md", "..."), ("code", "...")] siralı liste. Hucreler
    SIRAYLA, TEK kernel oturumunda calistirilir (bir onceki hucrenin
    degiskenleri sonrakine gecer) - gercek notebook davranisi."""
    nb = new_notebook()
    for kind, source in cells:
        if kind == "md":
            nb.cells.append(new_markdown_cell(source))
        elif kind == "code":
            nb.cells.append(new_code_cell(source))
        else:
            raise ValueError(f"bilinmeyen hucre turu: {kind}")

    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(REPO_ROOT)}})
    client.execute()

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_path)
    return nb


def cell_outputs_text(nb: nbformat.NotebookNode, cell_index: int) -> str:
    """Bir kod hucresinin tum stdout/text ciktisini birlestirir (rapor
    uretiminde hucre ciktisini programatik okumak icin)."""
    cell = nb.cells[cell_index]
    parts = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            parts.append(out.get("text", ""))
        elif out.get("output_type") in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "text/plain" in data:
                parts.append(data["text/plain"])
    return "".join(parts)


__all__ = ["build_and_execute", "cell_outputs_text", "REPO_ROOT"]
