# PoE_Small Repository

This repository contains the code and experimental results for the paper **"PoE_Small"**.

## Repository Structure

### `PoE_Small/`
This folder contains the implementation of our multi-agent system.  
Detailed instructions on how to run the system can be found in the **Appendix** of the paper.

#### `evaluation_scripts/`
- Contains the evaluation scripts used to extract and normalize answers.
- **Important:** Before running the evaluation scripts, ensure that the file paths are correctly set up according to your system.

### `results/`

- Contains the raw experimental results.
- Results are organized by dataset, with each dataset folder containing subfolders corresponding to different **description frameworks**.
- Each description framework folder includes the following files:

  - **`args_dict.json`** – Contains the experimental settings.
  - **`experts-list.json`** – Specifies the expertise fields of the expert agents.
  - **`experts.json`** – Stores the descriptions of all expert agents.
  - **`final_decisor.json`** – Provides the description of the **Final Decision Maker** agent.
  - **`projectmanager.json`** – Contains the description of the **Project Manager** agent.
  - **`psychologist.json`** – Describes the **Psychologist** agent.
  - **`queries_answers.json`** – Records the responses of the agents and the Final Decision Maker for each query.

For further details, refer to the `Suplementary Material.pdf`.


Implementation details can be found in `ImplementaionDetails.pdf`