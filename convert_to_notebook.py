import nbformat
import json

def convert_to_notebook(md_file, py_file, out_ipynb):
    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()
    
    with open(py_file, "r", encoding="utf-8") as f:
        py_content = f.read()

    nb = nbformat.v4.new_notebook()

    # Split markdown by sections (h2)
    import re
    sections = re.split(r'\n(?=## )', md_content)
    
    for section in sections:
        if section.strip():
            nb.cells.append(nbformat.v4.new_markdown_cell(section.strip()))

    # Add code as a single cell
    nb.cells.append(nbformat.v4.new_markdown_cell("## Complete Implementation Code"))
    nb.cells.append(nbformat.v4.new_code_cell(py_content))

    with open(out_ipynb, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

convert_to_notebook("plan_and_execute_pattern.md", "17_plan_and_execute.py", "17_plan_and_execute.ipynb")
convert_to_notebook("self_reflection_pattern.md", "16_self_reflection.py", "16_self_reflection.ipynb")
convert_to_notebook("parallel_map_reduce_pattern.md", "22_parallel_map_reduce.py", "22_parallel_map_reduce.ipynb")
convert_to_notebook("supervisor_as_tools_pattern.md", "23_supervisor_as_tools.py", "23_supervisor_as_tools.ipynb")
convert_to_notebook("hitl_pattern.md", "14_hitl.py", "14_hitl.ipynb")
convert_to_notebook("hierarchical_rag_pattern.md", "24_hierarchical_rag.py", "24_hierarchical_rag.ipynb")
print("Successfully created notebooks")
