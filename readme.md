# Code Structure

The main entry point of the project is _main.py_


This script orchestrates the execution of the pipeline and connects all core components of the project.

---

# Word2Vec Models

We provide pre-trained **Word2Vec** models located in the `./models/` folder.  
These models were trained on the following dataset splits:

- `train`
- `train_val`
- `train_val_test`

**Important:**  
The models were trained **including the evaluation dataset**.

---

# Project Modules

The project is organized into modular components. Some internal utility modules are included in `./utils/`

## Utils
     
These modules contain supporting functions and internal logic required for the pipeline to run correctly. While they are not the main entry point, they are essential dependencies of the project.



## Data Folder

All datasets must be placed inside the following directory: `./data/`


Make sure the expected files and subfolders are available before running the project. Missing data may cause the pipeline to fail.

---

# Repository

GitHub repository:  
👉 https://github.com/Gianbattistabsn/Winter-project-DSMLL-2025-2026-ImpChi-and-W2V

