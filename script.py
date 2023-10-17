import gradio as gr
import re
import os
from pathlib import Path
import json
from peft import PeftModel
import modules.shared as shared
from modules.LoRA import add_lora_autogptq, add_lora_exllama
import torch
from datetime import datetime

try:
    from peft.config import PeftConfig
    print("NEW PEFT is installed")
except ImportError:
        print("Error: you are using an old PEFT version. LORA merging will not work. You need to update to the latest version")
        from peft.utils.config import PeftConfig

params = {
        "display_name": "Virtual Lora",
        "is_tab": True,
    }

g_print_twice = False

folder_tree = {}

struct_params = {
    "edit": True,
    "root_SEL": "",
    "folders_SEL": "None",
    "subfolders_SEL": "None",
    "selected_template": "Latest",
    "sort_by_date": False,
}

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"

BYDATE = "[All By Month]"
BYDATE2 = "[Last 10 dates]"

refresh_symbol = '\U0001f504'  # 🔄


def get_file_path(filename):
    basepath = "extensions/VirtualLora/"+filename
   
    #if os.path.exists(basepath):
    #    return basepath
    #else:
    #    return None

    return basepath

def save_folder_file(string,filename):
    path = get_file_path("Templates/"+filename+".txt")
    try:
        with open(path, 'w', encoding='utf-8') as file:
            file.write(string)
        print(f"Tree saved {path}")
    except Exception as e:
        print("Error occurred while saving string to file:", str(e))

def load_folder_file(filename):

    if filename==BYDATE:
        file_content = create_Folders_byDate(False)
        return file_content
    
    if filename==BYDATE2:
        file_content = create_Folders_byDate(True)
        return file_content
    
    path = get_file_path("Templates/"+filename+".txt")

    try:
        with open(path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            print(f"File loaded from: {path}")
            return file_content
    except FileNotFoundError:
        print(f"The file '{path}' does not exist.")
        return ""

def create_folder_tree(input_string):
    global folder_tree
    folder_tree = {}
    lines = input_string.split('\n')
    current_folder = None
    for line in lines:
        if line.startswith('+'):
            if current_folder is not None:
                newline = line[1:]
                newline = newline.strip()
                folder_tree[current_folder].append(newline)
        else:
            line = line.strip()
            if line != '':
                current_folder = line
                folder_tree[current_folder] = []


def get_root_list():
    global folder_tree
    root_folders = []
    for folder in folder_tree:
        root_folders.append(folder)

    return root_folders


def create_Folders_byDate(last_five):

# Define the root folder
    rootFolder = shared.args.lora_dir
    if not rootFolder.endswith('/'):
        rootFolder += '/'
    # Create a list to store subfolder information
    subfolder_info = []
    index = 0
    print(f"Looking and sorting folders:")
    # Walk through the root folder to get subfolders

    name_list = os.listdir(rootFolder)
    full_list = [os.path.join(rootFolder, i) for i in name_list]

    for foldername in full_list:

        fimename = f"{foldername}"+"/adapter_config.json"
        
        if os.path.exists(fimename):
            try:
                file_timestamp = os.path.getmtime(fimename)
                formatted_date = datetime.fromtimestamp(file_timestamp).strftime("%Y-%m-%d-%H-%M")
                if last_five:
                    formatted_date_min = datetime.fromtimestamp(file_timestamp).strftime("%Y-%m-%d")
                else:
                    formatted_date_min = datetime.fromtimestamp(file_timestamp).strftime("%Y-%m")

                # Create a dictionary for the subfolder
            except:
                formatted_date = "1999-00-00-00-00"
                formatted_date_min = "Unknown"

            subfolder_name = os.path.basename(foldername)
            subfolder_dict = {
                "subfolder": subfolder_name,
                "date": formatted_date,
                "date_min": formatted_date_min
                }

            subfolder_info.append(subfolder_dict)
            index = index +1

    # Sort the subfolder_info list based on the "date" key
    subfolder_info.sort(key=lambda x: x["date"], reverse=True)

    print(f" Found: {index} folders")

    # Create a string to group subfolders by date
    grouped_subfolders_string = ""
    current_date = None
    idx = 0
    for subfolder in subfolder_info:
        subfrom = subfolder["date_min"]
        if subfrom != current_date:
           
            if last_five and idx>9:
                return grouped_subfolders_string

            if current_date:
                grouped_subfolders_string += "\n"
            current_date = subfrom
            grouped_subfolders_string += current_date + "\n"
            idx = idx+1


        grouped_subfolders_string += "+ " + subfolder["subfolder"] + "\n"

    return grouped_subfolders_string


def get_folder_list(selected_root_folder):
    global folder_tree
    if selected_root_folder in folder_tree:
        subfolders = list(folder_tree[selected_root_folder])
        return subfolders
    else:
        return ["None"]


def atoi(text):
    return int(text) if text.isdigit() else text.lower()

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def string_to_name_list(input_string):
    # Split the input string by commas and remove leading/trailing spaces
    name_list = [name.strip() for name in input_string.split(',')]
    return name_list

def name_list_to_string(name_list):
    # Join the list of names with commas
    return ', '.join(name_list)

def get_available_templates():
    templpath = get_file_path("Templates")
    paths = (x for x in Path(templpath).iterdir() if x.suffix in ('.txt'))
    sortedlist = sorted(set((k.stem for k in paths)), key=natural_keys)
    if len(sortedlist)==0:
        sortedlist = ['Latest']
    
    sortedlist.insert(0, BYDATE)
    sortedlist.insert(0, BYDATE2)

    return sortedlist

def list_Folders_byDate(directory):

    if not directory.endswith('/'):
        directory += '/'
    subfolders = []
    path = directory
    name_list = os.listdir(path)
    full_list = [os.path.join(path, i) for i in name_list]

    time_sorted_list = sorted(full_list, key=os.path.getmtime , reverse=True)

    for entry in time_sorted_list:
        if os.path.isdir(entry):
            entry_str = f"{entry}"  # Convert entry to a string
            full_path = entry_str
            entry_str = entry_str.replace('\\','/')
            entry_str = entry_str.replace(f"{directory}", "")  # Remove directory part
            entry_str = entry_str.strip()
            entry_str = "+ "+ entry_str
            subfolders.append(entry_str)

    return subfolders

def list_Folders_byAlpha(directory):

    if not directory.endswith('/'):
        directory += '/'

    subfolders = []
    path = directory
    name_list = os.listdir(path)
    full_list = [os.path.join(path, i) for i in name_list]

    time_sorted_list = sorted(full_list, key=natural_keys, reverse=False)

    for entry in time_sorted_list:
        if os.path.isdir(entry):
            entry_str = f"{entry}"  # Convert entry to a string
            full_path = entry_str
            entry_str = entry_str.replace('\\','/')
            entry_str = entry_str.replace(f"{directory}", "")  # Remove directory part
            entry_str = entry_str.strip()
            entry_str = "+ "+ entry_str
            subfolders.append(entry_str)

    return subfolders

#sort in natural order reverse
def list_subfolders(directory):
    subfolders = []
    
    if os.path.isdir(directory):
        
        subfolders.append('Final')

        for entry in os.scandir(directory):
            if entry.is_dir() and entry.name != 'runs':
                subfolders.append(entry.name)

    return sorted(subfolders, key=natural_keys, reverse=True)

def save_pickle():
    global struct_params
    file_nameJSON = get_file_path("params.json")
    try:
        with open(file_nameJSON, 'w') as json_file:
            json.dump(struct_params, json_file,indent=2)
            print(f"Saved: {file_nameJSON}")
    except IOError as e:
        print(f"An error occurred while saving the file: {e}")  


def path_to_LORA(selectlora , selectsub):

    if selectsub=='':
        selectsub = 'Final'

    if selectsub and selectsub!='Final':
        return f"{selectlora}/{selectsub}"
    
    return f"{selectlora}"

def load_pickle():
    global struct_params
    file_nameJSON = get_file_path("params.json")
    try:
        with open(file_nameJSON, 'r') as json_file:
            new_params = json.load(json_file)
            for item in new_params:
                struct_params[item] = new_params[item]
    except FileNotFoundError:
        print(f"Default values, the file '{file_nameJSON}' does not exist.")

def load_note():
    selected_lora_main = struct_params["folders_SEL"]
    if selected_lora_main=='':
        return ""
    
    path = path_to_LORA(selected_lora_main,"Final")
    full_path = Path(f"{shared.args.lora_dir}/{path}/notes.txt")

    note = f'<h3 style="color: orange;">{selected_lora_main}</h3>'
    if full_path.is_file():
        try:
            with open(full_path, 'r', encoding='utf-8') as file:
                note = note+file.read()
        except:
            pass

    return note        

def load_log():

    selected_lora_main = struct_params["folders_SEL"]
    selected_lora_sub = struct_params['subfolders_SEL'] 
    
    if selected_lora_main=='':
        return "None","Select LoRA"

    adapter_params = None
    new_params = None

    path = path_to_LORA(selected_lora_main,selected_lora_sub)
    full_path = Path(f"{shared.args.lora_dir}/{path}/training_log.json")
    full_pathAda = Path(f"{shared.args.lora_dir}/{path}/adapter_config.json")
    try:
        with open(full_pathAda, 'r') as json_file:
            adapter_params = json.load(json_file)

    except:
        pass

    str_noteline = ''

    table_html = '<table>'

    try:
        with open(full_path, 'r') as json_file:
            new_params = json.load(json_file)
    except FileNotFoundError: 
        pass

    row_one = '<tr style="text-align: center;">'
    row_two = '<tr style="text-align: center;">'


    if new_params:    
        keys_to_include = ['base_model_name', 'loss', 'learning_rate', 'epoch', 'current_steps', 'projections', 'epoch_adjusted']

        epoch_str = ''

        for key, value in new_params.items():

            if key=='note':
                str_noteline = f"\nNote: {value}"

            if key=="base_model_name":
                base_model = f"Base: {value}"

            if key =="epoch":
                epoch_str = f'{value:.2}'

            if key in keys_to_include:
                # Create the first row with keys
                valid = True

                if key == "epoch_adjusted":
                    value2 = new_params.get(key, '')
                    epoch_str2 = f'{value2:.2}'
                    
                    if epoch_str==epoch_str2:
                        valid = False

                if valid:
                    row_one += f'<th style="border: 1px solid gray; padding: 8px; text-align: center; background-color: #233958; color: white;">{key}</th>'
                    value = new_params.get(key, '')
                    if isinstance(value, float) and value < 1:
                        value = f'{value:.1e}'
                    elif isinstance(value, float):
                        value = f'{value:.2}'

                    row_two += f'<td style="border: 1px solid gray; padding: 8px; text-align: center;">{value}</td>'
                    

  
    if new_params and adapter_params:
        keys_to_include = ['r', 'lora_alpha']
        for key, value in adapter_params.items():
            if key in keys_to_include:
                row_one += f'<th style="border: 1px solid gray; padding: 8px; text-align: center; background-color: #235358; color: white;">{key}</th>'
                value = adapter_params.get(key, '')
                if isinstance(value, float) and value < 1:
                    value = f'{value:.1e}'
                elif isinstance(value, float):
                    value = f'{value:.2}'
                row_two += f'<td style="border: 1px solid gray; padding: 8px; text-align: center;">{value}</td>'
               
    if new_params==None and adapter_params:
        keys_to_include = ['base_model_name_or_path','r', 'lora_alpha', 'target_modules']
        row_one += f'<th style="border: 1px solid gray; padding: 8px; text-align: center; background-color: #8E2438; color: white;">No log file</th>'
        row_two += f'<td style="border: 1px solid gray; padding: 8px; text-align: center;">training_log.json</td>'
        for key, value in adapter_params.items():
            if key in keys_to_include:
                row_one += f'<th style="border: 1px solid gray; padding: 8px; text-align: center; background-color: #235358; color: white;">{key}</th>'
                value = adapter_params.get(key, '')
                if isinstance(value, float) and value < 1:
                    value = f'{value:.1e}'
                elif isinstance(value, float):
                    value = f'{value:.2}'
                row_two += f'<td style="border: 1px solid gray; padding: 8px; text-align: center;">{value}</td>'
                

    row_one += '</tr>'        
    row_two += '</tr>'
    table_html += row_one + row_two + '</table>'     


    return table_html+str_noteline,"Selection changed, Press Load LORA"

def get_loaded_adapters():
    prior_set = []
    if hasattr(shared.model,'peft_config'):
        for adapter_name in shared.model.peft_config.items():
            prior_set.append(adapter_name[0])
    return prior_set      

def get_available_adapters_ui():

    #print (f"Scaling {shared.model.base_model.scaling}")

    prior_set = ['None']
    
    if shared.model:
        if hasattr(shared.model,'peft_config'):
            print(RED+"List of available adapters in model:"+RESET)
            index = 1
            for adapter_name in shared.model.peft_config.items():
                print(f"  {GREEN}{index}:{RESET} {adapter_name[0]}")
                index = index+1
                prior_set.append(adapter_name[0])
            
            if index == 1:
                print(RED+"  [None]"+RESET)
        else:
            print(f'({shared.model_name} is not PEFT model)')

    else:
        print('(no model loaded yet)')

    return prior_set      

def add_lora_to_model(lora_name):
    if 'GPTQForCausalLM' in shared.model.__class__.__name__ or shared.args.loader == 'AutoGPTQ':
        add_lora_autogptq([lora_name])
    elif shared.model.__class__.__name__ in ['ExllamaModel', 'ExllamaHF'] or shared.args.loader == 'ExLlama':
        add_lora_exllama([lora_name])
    else:
        
        params = {}
        if not shared.args.cpu:
            if shared.args.load_in_4bit or shared.args.load_in_8bit:
                params['peft_type'] = shared.model.dtype
            else:
                params['dtype'] = shared.model.dtype
                if hasattr(shared.model, "hf_device_map"):
                    params['device_map'] = {"base_model.model." + k: v for k, v in shared.model.hf_device_map.items()}

        print(f"Applying the following LoRAs to {shared.model_name}: {lora_name}")

        lora_path = Path(f"{shared.args.lora_dir}/{lora_name}")
        
        shared.model = PeftModel.from_pretrained(shared.model,  lora_path, adapter_name=lora_name, **params)
        

        if not shared.args.load_in_8bit and not shared.args.cpu:
            shared.model.half()
            if not hasattr(shared.model, "hf_device_map"):
                if torch.backends.mps.is_available():
                    device = torch.device('mps')
                    shared.model = shared.model.to(device)
                else:
                    shared.model = shared.model.cuda()


def Load_and_apply_lora():

    selected_lora_main = struct_params["folders_SEL"]
    selected_lora_sub = struct_params['subfolders_SEL'] 
    path = path_to_LORA(selected_lora_main, selected_lora_sub)

    lora_path = Path(f"{shared.args.lora_dir}/{path}")
    selected_lora_main_sub = path

    if os.path.isdir(lora_path):
            
        if shared.model_name!='None' and shared.model_name!='':
            yield (f"Applying the following LoRAs to {shared.model_name} : {selected_lora_main_sub}")

            shared.lora_names = []
            loras_before = get_loaded_adapters()


            if 'GPTQForCausalLM' in shared.model.__class__.__name__:
                print("LORA -> AutoGPTQ")
            elif shared.model.__class__.__name__ == 'ExllamaModel':
                print("LORA -> Exllama")
            else:
                        # shared.model may no longer be PeftModel
                print("LORA -> Transformers") 


                if hasattr(shared.model, 'disable_adapter'):
                    print (RED+"Disable PEFT adapter"+RESET)  
                    shared.model.disable_adapter()
                    #delete_All_ButFirst()
                    #shared.model = shared.model.base_model.model.unload()
                    
                modeltype = shared.model.__class__.__name__

                if hasattr(shared.model.base_model, 'model'):
                    modelbasetype = shared.model.base_model.model.__class__.__name__

                    print(f"Returning  model {YELLOW}{modeltype}{RESET} back to {YELLOW}{modelbasetype}{RESET}") 
                    shared.model = shared.model.base_model.model
                else:
                    print(f"Starting from {YELLOW}clean{RESET} model {YELLOW}{modeltype}{RESET}") 


            modeltype = shared.model.__class__.__name__

            print(f"Creating {RED}PEFT{RESET} model for {YELLOW}{modeltype}{RESET}")

            #if len(loras_before) == 0:
            add_lora_to_model(selected_lora_main_sub)
            modeltype = shared.model.__class__.__name__

            if hasattr(shared.model.base_model, 'model'):
                modelbasetype = shared.model.base_model.model.__class__.__name__
                print(f"{GREEN}[OK] {RESET} Model {YELLOW}{modeltype}{RESET} created on top of {YELLOW}{modelbasetype}{RESET} with {GREEN}{selected_lora_main_sub}{RESET}")
            else: 
                print(f"{RED}Error - no PEFT model created for{RESET} {YELLOW}{modeltype}{RESET}")

            #Select_last_lora()

            #else: 
            #    reload_model()
            #    add_lora_to_model(selected_lora_main_sub)

            
            #delete_All_loraButActive()

            loras_after =  get_loaded_adapters()
            
            if loras_before == loras_after:
                yield "Nothing changed..." 
            else:
                yield f"Successfuly applied new adapter: {selected_lora_main_sub}"   
                
        else:
            print("you have no model loaded yet!")
            yield 'No Model loaded...' 
        
        adapter_name = getattr(shared.model,'active_adapter','None')
        print (f"{YELLOW}Active adapter:{RESET} {adapter_name}")      

def add_lora_to_PEFT():

    selected_lora_main = struct_params["folders_SEL"]
    selected_lora_sub = struct_params['subfolders_SEL'] 
    path = path_to_LORA(selected_lora_main, selected_lora_sub)

    lora_path = Path(f"{shared.args.lora_dir}/{path}")

    
    selected_lora_main_sub = path

    print(f"{YELLOW}Adding Lora from:{RESET} {lora_path}")

    if os.path.isdir(lora_path):
        
        loras_before = get_loaded_adapters()
        if len(loras_before) == 0:
                yield (f"First lora needs to be loaded with Load Lora")     
        else:    
            if shared.model_name!='None' and shared.model_name!='':
                yield (f"Adding the following LoRAs to {shared.model_name} : {selected_lora_main_sub}")

                newkey = selected_lora_main_sub

                shared.model.load_adapter(lora_path, newkey)
                loras_after =  get_loaded_adapters()
                if loras_before == loras_after:
                    print("No Lora Added")
                    yield 'No Lora added...' 
                else:
                    # get last item of loras_after
                    last_lora = loras_after[-1]
                    print (f"{GREEN}Added Lora: {RESET} {last_lora}")
                    yield (f"Added Lora {last_lora}")
        
        Select_last_lora()
        adapter_name = getattr(shared.model,'active_adapter','None')
        print (f"Active adapter: {adapter_name}")


def set_adapter(item):

    if shared.model == None:
        print(f"No Model loaded")    
        return
    
    print(RED+ 'SET LORA:'+RESET)
    if hasattr(shared.model, 'set_adapter') and hasattr(shared.model, 'active_adapter'):
        #if prior_set:

        if hasattr(shared.model.base_model, 'model'):
            modelbasetype = shared.model.base_model.__class__.__name__
        else:
            modelbasetype = 'None'

        modeltype = shared.model.__class__.__name__
        if not hasattr(shared.model.base_model, 'disable_adapter_layers'):
            
            print(f"{RED} ERROR {RESET} {YELLOW}{modeltype}{RESET} ({modelbasetype}) is not PEFT model (PeftModelForCausalLM). You need to Load Lora first.")
            
            return


        if (item =='None' or item == None or item == ''):
            shared.model.base_model.disable_adapter_layers()
            print (f"{RED} [Disable]{RESET} Adapters in  {YELLOW}{modeltype}{RESET} ({modelbasetype})")   
        else:
            adapters = get_loaded_adapters()
            
            if item in adapters:
                shared.model.set_adapter(item)
                if hasattr(shared.model.base_model, 'enable_adapter_layers'):
                    shared.model.base_model.enable_adapter_layers()
                    print (f"{GREEN} [Enable]{RESET} {shared.model.active_adapter} in {YELLOW}{modeltype}{RESET} ({modelbasetype})")
                else:
                     print(f"{RED} ERROR {RESET} {YELLOW}{modeltype}{RESET} with base {YELLOW}{modelbasetype}{RESET} is not correct PEFT model.")

            else:
                print (f"No or unknown Adapter {item} in {adapters}")
                
                shared.model.base_model.disable_adapter_layers()
                print (f"{RED} [Disable]{RESET} Adapters in {YELLOW}{modeltype}{RESET} ({modelbasetype})")   
        
    else:
        print(f" Wrong model {shared.model.__class__.__name__}, it has no support for adapters")                
          
        

def Select_last_lora():
    loras_before = get_loaded_adapters()
    last_element = loras_before[-1]
    set_adapter(last_element)


def ui():
    global struct_params
    global folder_tree

    load_pickle()

    text_file = load_folder_file(struct_params['selected_template'])

    create_folder_tree(text_file)


    list_fold = get_folder_list(struct_params['root_SEL'])
    list_checkpoints = []

    selected_lora_main = struct_params["folders_SEL"]
    if selected_lora_main !='' and selected_lora_main!="None":
        model_dir = f"{shared.args.lora_dir}/{selected_lora_main}"  # Update with the appropriate directory path
        list_checkpoints = list_subfolders(model_dir)
    
    html_text,status_txt = load_log()
    html_note = load_note()

    if shared.model:
        model_name = str(getattr(shared.model,'active_adapter','None'))
    else:
        model_name = str('None')


    with gr.Tab('Lora'):
        with gr.Row():
            gr_Loralmenu = gr.Radio(choices=get_available_adapters_ui(), value=model_name, label='Activate adapter', interactive=True)
            gr_Loralmenu_refresh = gr.Button(value=refresh_symbol, elem_classes='refresh-button')


        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            para_templates_drop = gr.Dropdown(choices=get_available_templates(), label='Collection Set', value=struct_params['selected_template'])
                        with gr.Row():
                            gr.Markdown(' ')    
                    with gr.Column(scale=3):
                        gr_displayLine2 = gr.HTML(html_note)
            with gr.Column():                    
                gr_displayLine = gr.HTML(html_text)

        with gr.Row():
            with gr.Column():
                with gr.Row():    
                    with gr.Column(scale = 1):
                        gr_ROOT_radio = gr.Radio(choices=get_root_list(), value=struct_params['root_SEL'], label='Collection',
                                                interactive=True, elem_classes='checkboxgroup-table')
                    with gr.Column(scale = 3):
                        gr_FOLDER_radio = gr.Radio(choices=list_fold, value=struct_params['folders_SEL'], label='Folders', interactive=True, elem_classes='checkboxgroup-table')
                        gr_EditNameLora = gr.Textbox(value='',lines=4,visible=False,variant="primary")
                        gr_EditNoteSaveLora = gr.Button(value='Save Note',visible=False,variant="primary")
                        gr_EditNameCancelLora = gr.Button(value='Cancel',visible=False)
            with gr.Column():
                with gr.Row():    
                    with gr.Column(scale = 3):
                        gr_SUBFOLDER_radio = gr.Radio(choices=list_checkpoints, value=struct_params['subfolders_SEL'], label='Checkpoints', interactive=True, elem_classes='checkboxgroup-table')
                        
                        gr_EditName = gr.Text(value='',visible=False,label='Edit')
                        gr_EditNameSave = gr.Button(value='Rename',visible=False,variant="primary")
                        gr_EditNoteSave = gr.Button(value='Save Note',visible=False,variant="primary")
                        gr_EditNameCancel = gr.Button(value='Cancel',visible=False)

                    with gr.Column(scale = 1):
                        lora_Load = gr.Button(value='Load LoRA', variant="primary")
                        lora_Add = gr.Button(value='+ Add LoRA')
                        #lora_Disable = gr.Button(value='Disable Lora',variant="stop")
                        lora_Rename = gr.Button(value='Rename CHKP')
                        lora_Note = gr.Button(value='Edit CHKP Note')
                        lora_all_Note = gr.Button(value='Edit Lora Note')

                   
    with gr.Tab('Setup'):
        with gr.Row():
            gr_setup_templName = gr.Textbox(label="Set Name", value=struct_params['selected_template'], lines=1)
            gr_setup_templName2 = gr.Textbox(label="Lora Folder", value="/loras/", lines=1)
        with gr.Row():
            gr_setup = gr.Textbox(label="Set Definition", value=text_file, lines=15, elem_classes='textbox')
            gr_setup_folders = gr.Textbox(label="Folders", value='', lines=15, elem_classes=['textbox', 'add_scrollbar'])
        with gr.Row():  
            gr.Markdown('')
            with gr.Row():
                gr_setup_search = gr.Text(label='Must Include string', value='')
                gr_setup_byDate = gr.Checkbox(label = 'Sort by Date', value =struct_params['sort_by_date'])  
        with gr.Row():
            gr_setup_APPLY = gr.Button("Save", variant='primary')
            gr_setup_REFRESH = gr.Button("Refresh")

    with gr.Row():
        status_text = gr.Markdown(value='Ready')

    def update_folders():
        value = struct_params['root_SEL']
        list_fold = get_folder_list(value)
        return gr.Radio.update(choices=list_fold, value='')

    def save_template(setup_text, templatename):
        global struct_params
        save_folder_file(setup_text,templatename)
        struct_params['selected_template'] = templatename

    def reload_tree():
        global struct_params
        textfile = load_folder_file(struct_params['selected_template'])
        create_folder_tree(textfile)
        struct_params['folders_SEL'] = ""
        struct_params['subfolders_SEL'] = ""
        choices = get_root_list()

        return gr.Radio.update(choices=choices, value=''), gr.Radio.update(choices=["None"], value=''), gr.Radio.update(choices=[], value=''),textfile,struct_params['selected_template']

    def update_dropdown():
        templates = get_available_templates()
        return gr.Dropdown.update(choices=templates, value=struct_params['selected_template'])

    gr_setup_APPLY.click(save_template, [gr_setup,gr_setup_templName], None).then(
        reload_tree, None, [gr_ROOT_radio, gr_FOLDER_radio, gr_SUBFOLDER_radio,gr_setup,gr_setup_templName]).then(update_dropdown,None,para_templates_drop)

    def refresh_Lorafolders(must_include):
        model_dir = shared.args.lora_dir 
        if struct_params['sort_by_date']:
            folder = list_Folders_byDate(model_dir)
        else:
            folder = list_Folders_byAlpha(model_dir)
        must_include = must_include.strip()
        
        if must_include!='':
            new_list = []
            must_include = must_include.lower() 
            for item in folder:
                if must_include in item.lower():
                    new_list.append(item)

            return '\n'.join(new_list)        

        return '\n'.join(folder)


    gr_setup_REFRESH.click(refresh_Lorafolders,gr_setup_search, gr_setup_folders)

    para_templates_drop.change(lambda x: struct_params.update({"selected_template": x}), para_templates_drop, None).then(
        reload_tree, None, [gr_ROOT_radio, gr_FOLDER_radio, gr_SUBFOLDER_radio,gr_setup,gr_setup_templName])

    def update_lotra_subs():
        global struct_params
        selected_lora_main = struct_params["folders_SEL"]
        if selected_lora_main !='' and selected_lora_main!="None":
            model_dir = f"{shared.args.lora_dir}/{selected_lora_main}"  # Update with the appropriate directory path
            subfolders = list_subfolders(model_dir)
            struct_params['subfolders_SEL'] = "Final"
            return gr.Radio.update(choices=subfolders, value ='Final') 

        return gr.Radio.update(choices=[], value ='')    

    gr_ROOT_radio.change(lambda x: struct_params.update({"root_SEL": x}), gr_ROOT_radio, None).then(update_folders, None, gr_FOLDER_radio, show_progress=False)

    gr_FOLDER_radio.change(lambda x: struct_params.update({"folders_SEL": x}), gr_FOLDER_radio, None).then(
        update_lotra_subs, None, gr_SUBFOLDER_radio, show_progress=False).then(
        load_note,None,gr_displayLine2, show_progress=False).then(
        load_log,None,[gr_displayLine,status_text], show_progress=False)
        

    gr_SUBFOLDER_radio.change(lambda x: struct_params.update({"subfolders_SEL": x}), gr_SUBFOLDER_radio, None).then(load_log,None,[gr_displayLine,status_text], show_progress=False)

    def show_edit_rename():
        selected_lora_sub = struct_params['subfolders_SEL'] 
        note = selected_lora_sub
        if note == '' or note =='Final':
            note = 'Rename sub only!'
            return gr.Textbox.update(value = note, interactive= True, visible=True),gr.Button.update(visible=False),gr.Button.update(visible=True)

        return gr.Textbox.update(value = note, interactive= True, visible=True),gr.Button.update(visible=True),gr.Button.update(visible=True)

    def show_edit_Note():
        selected_lora_main = struct_params["folders_SEL"]
        selected_lora_sub = struct_params['subfolders_SEL'] 
        path = path_to_LORA(selected_lora_main, selected_lora_sub)

        full_path = Path(f"{shared.args.lora_dir}/{path}/training_log.json")

        note = 'Write a note here...'
        try:
            with open(full_path, 'r') as json_file:
                new_params = json.load(json_file)
                
                for key, value in new_params.items():
                    if key=='note':
                        note = f"{value}"
        except FileNotFoundError:
            pass 

        return gr.Textbox.update(value = note, interactive= True, visible=True),gr.Button.update(visible=True),gr.Button.update(visible=True)


    def show_cancel():

        return gr.Textbox.update(visible=False),gr.Button.update(visible=False),gr.Button.update(visible=False),gr.Button.update(visible=False)

    def rename_chkp(line):
       
        selected_lora_main = struct_params["folders_SEL"]
        selected_lora_sub = struct_params['subfolders_SEL'] 
        path = path_to_LORA(selected_lora_main, selected_lora_sub)
   
        full_path = Path(f"{shared.args.lora_dir}/{path}")
        newpath = path_to_LORA( selected_lora_main, line)
        full_newpath = Path(f"{shared.args.lora_dir}/{newpath}")
        note = 'Rename Failed'
        try:
            # Rename the subfolder
            os.rename(full_path, full_newpath)
            print(f"Renamed '{selected_lora_sub}' to '{line}'")
            note = f"Renamed '{selected_lora_sub}' to '{line}'"
        except FileNotFoundError:
            print(f"Error: The '{full_path}' does not exist.")
        except PermissionError:
            print(f"Error: Permission denied. You may not have the necessary permissions.")
        except OSError as e:
            print(f"Error: An OS error occurred: {e}")    

            # reload again
        
        return gr.Textbox.update(value = note, interactive= True, visible=True),gr.Textbox.update(visible=False),gr.Button.update(visible=False),gr.Button.update(visible=False)

    def save_note(line):
        note = ''
        selected_lora_main = struct_params["folders_SEL"]
        selected_lora_sub = struct_params['subfolders_SEL'] 
 
        path = path_to_LORA(selected_lora_main,selected_lora_sub)
        

        full_path = Path(f"{shared.args.lora_dir}/{path}/training_log.json")
            
        #load log
        resave_new = {}
        try:
            with open(full_path, 'r') as json_file:
                new_params = json.load(json_file)
                
                for item in new_params:
                    resave_new[item] = new_params[item]

        except FileNotFoundError:
            note = f"Error loading {full_path}"
            pass 


        line_str = f"{line}"        
        if line_str != 'Write a note here...':
            resave_new.update({"note": line_str})        
            #save    
            if len(resave_new)>0:     
                try:
                    with open(full_path, 'w') as json_file:
                        json.dump(resave_new, json_file,indent=2)
                        print(f"Saved: {full_path}")
                        note = f"Saved: {full_path}"
                except IOError as e:
                    print(f"An error occurred while saving the file: {e}")  
                    note = f"Error resaving {full_path}"
        
        return gr.Textbox.update(value = note, interactive= True, visible=True),gr.Textbox.update(visible=False),gr.Button.update(visible=False),gr.Button.update(visible=False)

    gr_EditNameCancel.click(show_cancel,None,[gr_EditName,gr_EditNameSave,gr_EditNameCancel,gr_EditNoteSave])
    gr_EditNameSave.click(rename_chkp,gr_EditName,[gr_displayLine,gr_EditName,gr_EditNameSave,gr_EditNameCancel]).then(update_lotra_subs, None, gr_SUBFOLDER_radio, show_progress=False).then(load_log,None,[gr_displayLine,status_text], show_progress=False)
    lora_Rename.click(show_edit_rename,None,[gr_EditName,gr_EditNameSave,gr_EditNameCancel])


    gr_EditNoteSave.click(save_note,gr_EditName,[gr_displayLine,gr_EditName,gr_EditNoteSave,gr_EditNameCancel]).then(update_lotra_subs, None, gr_SUBFOLDER_radio, show_progress=False).then(load_log,None,[gr_displayLine,status_text], show_progress=False)
    lora_Note.click(show_edit_Note,None,[gr_EditName,gr_EditNoteSave,gr_EditNameCancel])


    def show_edit_NoteLora():
        selected_lora_main = struct_params["folders_SEL"]
        path = path_to_LORA(selected_lora_main, "Final")

        full_path = Path(f"{shared.args.lora_dir}/{path}/notes.txt")

        note = 'Write a note here...'
        try:
            with open(full_path, 'r', encoding='utf-8') as file:
                note = file.read()
        except:
            pass

        return gr.Textbox.update(value = note, interactive= True, visible=True),gr.Button.update(visible=True),gr.Button.update(visible=True)

    def save_note_LORA(line):
        selected_lora_main = struct_params["folders_SEL"]
        path = path_to_LORA(selected_lora_main, "Final")
        full_path = Path(f"{shared.args.lora_dir}/{path}/notes.txt")


        line_str = f"{line}"        
        if line_str != 'Write a note here...':
            try:
                with open(full_path, 'w', encoding='utf-8') as file:
                    file.write(line_str)
            except:
                pass

    lora_all_Note.click(show_edit_NoteLora,None,[gr_EditNameLora,gr_EditNoteSaveLora,gr_EditNameCancelLora])
    gr_EditNameCancelLora.click(show_cancel,None,[gr_EditNameLora,gr_EditNoteSaveLora,gr_EditNameCancelLora,gr_EditNoteSave])
    gr_EditNoteSaveLora.click(save_note_LORA,gr_EditNameLora, None).then(
        show_cancel,None,[gr_EditNameLora,gr_EditNoteSaveLora,gr_EditNameCancelLora,gr_EditNoteSave]).then(load_note,None,gr_displayLine2)

    def update_activeAdapters():
        choice = get_available_adapters_ui()
 
        cur_adapt = getattr(shared.model, 'active_adapter', 'None')
        if cur_adapt not in choice:
            cur_adapt = 'None'

        return gr.Radio.update(choices=choice, value= cur_adapt)


    lora_Load.click(Load_and_apply_lora,None,status_text).then(save_pickle,None,None).then(update_activeAdapters,None, gr_Loralmenu) 

    lora_Add.click(add_lora_to_PEFT,None,status_text).then(save_pickle,None,None).then(update_activeAdapters,None, gr_Loralmenu)

    gr_Loralmenu_refresh.click(update_activeAdapters,None, gr_Loralmenu)


    gr_Loralmenu.change(set_adapter,gr_Loralmenu,None)   

    gr_setup_byDate.change(lambda x: struct_params.update({"sort_by_date": x}), gr_setup_byDate, None) 



    #gr_set_ROOT.change(lambda x: struct_params.update({"root_folders": x}), gr_set_ROOT, None)

    