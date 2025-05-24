from _constant import *


# source_file_dir = '/mnt/data1/raiyan/breast_cancer/datasets/dmid/png_images/all_images/IMG'

# source_file_dir =  '/mnt/data1/raiyan/breast_cancer/datasets/dmid/pixel_level_annotations/png_images/IMG'
source_file_dir =  '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/images_png'
# saving_dir = '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/evaluated/llava_base/'

img_files = list_png_files(source_file_dir)

temp = 0
prompt_technique = "base"
prompt_template = """
I will provide you with a mammogram image. Your task is to analyze the image and extract key diagnostic information, including breast composition, BIRADS category, and any significant findings. Present the output in a structured JSON format with the following keys: IMG_ID, Breast_Composition, BIRADS, and Findings. Ensure the response is precise, medically relevant, and well-organized.
Please follow the below given JSON format for your response
{
    "IMG-ID" "<Image_Filename>",
    "BREAST-COMPOSITION" "<Description of breast tissue composition>",
    "BIRADS": "<BIRADS category; any values between 1 to 6. BI-RADS category is a standardized classification for breast imaging findings, ranging from 1 to 6, where: BI-RADS 1 indicates a negative result with no abnormalities; BI-RADS 2 signifies benign findings with no suspicion of cancer; BI-RADS 3 suggests a benign lesion, requiring short-term follow-up to confirm stability; BI-RADS 4 represents a suspicious abnormality needing biopsy, further divided into 4A (low suspicion), 4B (moderate suspicion), and 4C (high suspicion); BI-RADS 5 is highly suggestive of malignancy with a high probability of cancer; and BI-RADS 6 confirms a known malignancy with a biopsy-proven cancer diagnosis.",
    "FINDINGS": "<Summary of any abnormalities, calcifications, or other observations>"
}

"""

@click.command()
@click.option(
    "--model_name",
    default="llama3.1:latest",
    type=click.Choice(allowable_models),
    help="name of the model to be used for processing",
)
@click.option(
    "--reports_to_process", 
    default=-1,  # Default value
    type=int, 
    help="An extra integer to be passed via command line"
)

def main(model_name, reports_to_process):
    print(f"Received model_name: {model_name}")
    print(f"Received value for reports_to_process: {reports_to_process}")

    global data 

    if(reports_to_process > 0):
        # data = data.head(reports_to_process)
        print(f"Processing only {reports_to_process} reports")
    
    if(reports_to_process == -1):
        reports_to_process = len(img_files)


    # Your existing logic to handle logging
    # log_dir, log_file = "local_chat_history", f"{model_name+datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.csv"
    
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)

    # log_path = os.path.join(log_dir, log_file)

    # if not os.path.isfile(log_path):
    #     with open(log_path, mode="w", newline="", encoding="utf-8") as file:
    #         writer = csv.writer(file)
    #         writer.writerow(["timestamp", "question", "answer","reason","model_name"])

    # cnt = 0
    # print(questions)


    for report in range(0, reports_to_process):
        # report_id = source_file_dir + str(report+1).zfill(3)+'.png'
        report_id = source_file_dir + img_files[report]

        print(report_id)
        # image_id = 'IMG'+ str(report+1).zfill(3)
        image_id =  img_files[report].replace('.png', '')

        
        # query = 'image ID: ' + report_id
        query = prompt_template+ 'image ID: '+  report_id

        print("QUERY: ", query)

        ollama = Ollama(model=model_name, temperature=temp)
        logging.getLogger().setLevel(logging.ERROR)  # Suppress INFO logs
        response = ollama.invoke(query)
        print("RESPONSE: ",response)

        ###the following is a dummy response for testing ###

        # dummy_data = {
        #     "IMG_ID": "image_001.jpg",
        #     "Breast_Composition": "Dense tissue with scattered fibroglandular elements",
        #     "BIRADS": "2",
        #     "Findings": "No significant abnormalities or calcifications. Normal breast tissue."
        # }
        # dummy_data_str = json.dumps(dummy_data, indent=4)

        # response =dummy_data_str+"abdc"
        ### dummy response processing ENDS ###

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match in [None, ""]:
            json_match = {"IMG_ID": "NA", "Breast_Composition": "NA", "BIRADS": "NA", "Findings": "NA"}
        else:
            json_match = fix_json(json_match.group(0))
        
        print(json_match)
        # global saving_dir
        #constructing the saving dir here
        saving_dir = 'evaluated-vindr/'+model_name+'_/'
        print(saving_dir)

        image_saving_dir = saving_dir +image_id + '.json'

        os.makedirs(os.path.dirname(image_saving_dir), exist_ok=True)
        with open(image_saving_dir, 'w') as json_file:
            json.dump(json_match, json_file, indent=4)
        

        print("Data has been written to", image_saving_dir)

    print("\nTotal Reports Processed", reports_to_process)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()