from _constant import *


# source_file_dir = '/mnt/data1/raiyan/breast_cancer/datasets/dmid/png_images/all_images/IMG'

# source_file_dir =  '/mnt/data1/raiyan/breast_cancer/datasets/dmid/pixel_level_annotations/png_images/IMG'
source_file_dir =  '/mnt/data1/raiyan/breast_cancer/datasets/vindr/images_png'
# saving_dir = '/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/evaluated/llava_base/'

img_files = list_png_files(source_file_dir)

temp = 0
prompt_technique = "base"
prompt_template = """
I am providing you a mammogram image. Your task is to analyze the image and extract key diagnostic information, including breast composition, any significant findings, and assign the BIRADS score. Present the output in a structured JSON format with the following keys: IMG_ID, Breast_Composition, Findings, and BIRADS. Ensure the response is precise, medically relevant, and well-organized.
Please follow the below given JSON format for your response and only output a valid json object:
{
    "IMG-ID" "<Image_Filename>",
    "BREAST-COMPOSITION" "<Provide the tissue density in ACR format where ACR A is almost entirely fatty, ACR B is scattered fibroglandular densities, ACR C is heterogeneously dense, and ACR D is extremely dense>",
    "FINDINGS": "<Summary of any abnormalities, calcifications, or other observations>",
    "BIRADS": "<BIRADS category; any values between 1 to 6. BI-RADS category is a standardized classification for breast imaging findings, ranging from 1 to 6, where: BI-RADS 1 indicates a negative result with no abnormalities; BI-RADS 2 signifies benign findings with no suspicion of cancer; BI-RADS 3 suggests a benign lesion, requiring short-term follow-up to confirm stability; BI-RADS 4 represents a suspicious abnormality needing biopsy, further divided into 4A (low suspicion), 4B (moderate suspicion), and 4C (high suspicion); BI-RADS 5 is highly suggestive of malignancy with a high probability of cancer; and BI-RADS 6 confirms a known malignancy with a biopsy-proven cancer diagnosis.",
}

Here are some examples of doctor annotated reports to guide you:
Example 1:
{
    "Image": "/mnt/data1/raiyan/breast_cancer/datasets/vindr/images_png/f89214e649de29a9e050564b894f4036/0a0c8b8a32ff32b0feb0d76e93b8dcad.png",
    "Breast Composition": "C",
    "Findings": "['No Finding']",
    "BIRADS": "1"
}

Example 2:
{
    "Image": "/mnt/data1/raiyan/breast_cancer/datasets/vindr/images_png/50a8bfda39adc93f22a174153632b823/0a6f0eac805822cd00f3c25ba99569e8.png",
    "Breast Composition": "C",
    "Findings": "['Mass']",
    "BIRADS": "5"
}

Example 3:
{
    "Image": "/mnt/data1/raiyan/breast_cancer/datasets/vindr/images_png/1a5c56c6f83b913488686f7fca265194/0a4004a005e1d213aed0691106d8772f.png",
    "Breast Composition": "B",
    "Findings": "['No Finding']",
    "BIRADS": "2"
}

Example 4:
{
    "Image": "/mnt/data1/raiyan/breast_cancer/datasets/vindr/images_png/7c51789da6c462e55bcb95c2a7d437ee/f581ef53bb7e61f4575db33eceac8ff8.png",
    "Breast Composition": "C",
    "Findings": "['Nipple Retraction', 'Mass']",
    "BIRADS": "4"
}

Example 5:
{
    "Image": "/mnt/data1/raiyan/breast_cancer/datasets/vindr/images_png/2ad573bf0a419ab289c3a23c6a6db18f/d6db56d5cb90391e234dca76c4a344e4.png",
    "Breast Composition": "A",
    "Findings": "['Global Asymmetry']",
    "BIRADS": "3"
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
        print(f"Processing only {reports_to_process} reports")

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

    cnt=0
    for report in range(0, reports_to_process):
        # report_id = source_file_dir + str(report+1).zfill(3)+'.png'
        report_id = source_file_dir + img_files[report]

        # print(report_id)
        # image_id = 'IMG'+ str(report+1).zfill(3)
        image_id =  img_files[report].replace('.png', '')

        
        # query = 'image ID: ' + report_id
        query = prompt_template+ 'image ID: '+  report_id

        # print("QUERY: ", query)

        ollama = Ollama(model=model_name, temperature=temp)
        logging.getLogger().setLevel(logging.ERROR)  # Suppress INFO logs
        response = ollama.invoke(query)
        # print("RESPONSE: ",response)

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
        
        response = re.sub(r'\\_', '_',response)

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match in [None, ""]:
            json_match = {"IMG_ID": "NA", "Breast_Composition": "NA", "BIRADS": "NA", "Findings": "NA"}
        else:
            json_match = fix_json(json_match.group(0))
        
        

        # print(json_match)
        # global saving_dir
        #constructing the saving dir here
        saving_dir = 'evaluated-vindr/'+model_name+'_n-shot/'
        # print(saving_dir)
        
        
        image_id = image_id.rsplit('/', 1)[-1] 
        image_saving_dir = saving_dir +image_id + '.json'

        os.makedirs(os.path.dirname(image_saving_dir), exist_ok=True)
        with open(image_saving_dir, 'w') as json_file:
            json.dump(json_match, json_file, indent=4)
        
        cnt+=1
        if(cnt%20==0):
            print("Processed", cnt, "reports with model", model_name)

        # print("Data has been written to", image_saving_dir)

    print("\nTotal Reports Processed", reports_to_process)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()