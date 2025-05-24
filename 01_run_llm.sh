# This file contains some example run commands
# --reports_to_process=-1 means it will process all the reports in the dataset; 
# provide a valid number to process that many reports

# conda activate fine_tune

python3 01_run_llm.py --model_name=llava:latest --reports_to_process=-1
# python3 01_run_llm.py --model_name=qwen2.5:latest --reports_to_process=-1
# python3 01_run_llm.py --model_name=mistral:latest --reports_to_process=-1


# python3 01_run_llm.py --model_name=mixtral:8x7b-instruct-v0.1-q4_K_M --reports_to_process=-1


# python3 01_run_llm_CoT.py --model_name=mistral:latest --reports_to_process=-1
# python3 01_run_llm_CoT.py --model_name=qwen2.5:latest --reports_to_process=-1
python3 01_run_llm_CoT.py --model_name=llava:latest --reports_to_process=-1


# python3 01_run_llm_nshot.py --model_name=mistral:latest --reports_to_process=-1
# python3 01_run_llm_nshot.py --model_name=qwen2.5:latest --reports_to_process=-1
# python3 01_run_llm_nshot.py --model_name=llava:latest --reports_to_process=-1


# python3 02_run_llm_rag_nshot.py --model_name=mistral:latest --reports_to_process=-1
# python3 02_run_llm_rag_nshot.py --model_name=qwen2.5:latest --reports_to_process=-1
# python3 02_run_llm_rag_nshot.py --model_name=llava:latest --reports_to_process=-1