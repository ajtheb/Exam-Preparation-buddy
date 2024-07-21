**Exam Preparation Buddy** is a Retrieval Augmentation Generation **(RAG)** based python application which uses **PDFs given by the user** for grounded generation of answers of the user's queries. It is deployed using **streamlit** framework.

**Format** supported for subject material: PDFs

**LLM** used: Gemma 2b for now(constraint by GPU memory)

**Chunking strategy**: CharacterTextsplitter ( 800 characters per chunk and 100 characters overlap)




How to run?
1. Setup the conda environment:-

   Command:  conda env create -f environment.yml
2. Run the command : streamlit run app.py
3. Click on the link of hosted application server.

![image](https://github.com/user-attachments/assets/cd4f0725-3916-4936-91e1-b0b220aff4e1)
