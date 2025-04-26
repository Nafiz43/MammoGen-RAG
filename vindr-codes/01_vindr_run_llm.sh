# This file contains some example run commands
# --reports_to_process=-1 means it will process all the reports in the dataset; 
# provide a valid number to process that many reports

# conda activate fine_tune

# python3 00_populate_db_for_rag.py 

# python3 01_vindr_run_llm.py --model_name=llava:latest 
python3 01_vindr_run_llm.py --model_name=mistral:7b-instruct 
python3 01_vindr_run_llm.py --model_name=qwen2.5:latest 


python3 01_vindr_run_llm_CoT.py --model_name=llava:latest 
python3 01_vindr_run_llm_CoT.py --model_name=mistral:7b-instruct 
python3 01_vindr_run_llm_CoT.py --model_name=qwen2.5:latest 


python3 01_vindr_run_llm_nshot.py --model_name=llava:latest 
python3 01_vindr_run_llm_nshot.py --model_name=mistral:7b-instruct 
python3 01_vindr_run_llm_nshot.py --model_name=qwen2.5:latest 


python3 01_vindr_run_llm_rag_nshot.py --model_name=llava:latest 
python3 01_vindr_run_llm_rag_nshot.py --model_name=mistral:7b-instruct 
python3 01_vindr_run_llm_rag_nshot.py --model_name=qwen2.5:latest 

