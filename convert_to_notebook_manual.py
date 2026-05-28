import json
import re

def create_notebook(md_file, py_file, out_ipynb):
    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()
    
    with open(py_file, "r", encoding="utf-8") as f:
        py_content = f.read()

    cells = []
    
    # Split markdown by sections (## )
    sections = re.split(r'\n(?=## )', md_content)
    
    for section in sections:
        if section.strip():
            # Create a markdown cell
            cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in section.strip().split("\n")]
            }
            # Remove trailing newline from last line
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n")
            cells.append(cell)

    # Add code as a single cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Complete Implementation Code"]
    })
    
    py_lines = [line + "\n" for line in py_content.split("\n")]
    if py_lines:
        py_lines[-1] = py_lines[-1].rstrip("\n")
        
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": py_lines
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(out_ipynb, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

create_notebook("plan_and_execute_pattern.md", "17_plan_and_execute.py", "17_plan_and_execute.ipynb")
create_notebook("self_reflection_pattern.md", "16_self_reflection.py", "16_self_reflection.ipynb")
create_notebook("supervisor_as_tools_pattern.md", "23_supervisor_as_tools.py", "23_supervisor_as_tools.ipynb")
create_notebook("hitl_pattern.md", "14_hitl.py", "14_hitl.ipynb")
print("Successfully created notebooks!")
