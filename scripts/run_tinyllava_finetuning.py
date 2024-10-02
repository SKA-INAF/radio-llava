import tokenizers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor

from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module

import argparse

def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio


# - Configure wandb
#os.environ["WANDB_PROJECT"]= "tinyllava"
#os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

###########################
##     CUSTOM ARGS
###########################
def parse_custom_args(args_list):
    """This function parses and return arguments passed in"""
    parser = argparse.ArgumentParser(description="Parse args.")
    
    parser.add_argument('-vision_model_name_or_path','--vision_model_name_or_path', dest='vision_model_name_or_path', required=False, type=str, default='', help='vision_model_name_or_path. If empty, use default specified in model config file. This arg is to override default vision model with another pretrained one (NB: it must be the same vision model architecture).')
    parser.add_argument('--reset_imgnorm', dest='reset_imgnorm', action='store_true',help='Reset vision model image normalization to mean=0, std=1 (default=false)')	
    parser.set_defaults(reset_imgnorm=False)
	
    args = parser.parse_args(args_list)
    
    return args

###########################
##     TRAIN
###########################
def train():
    
    # - Load arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments, custom_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    logger_setting(getattr(training_arguments, 'output_dir', None))
    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments) 
    load_settings(model_arguments, data_arguments, training_arguments)
    
    print("custom_args")
    print(custom_args)
    
    # - Custom arguments
    custom_arguments= parse_custom_args(custom_args)
    
    # - Load pretrained checkpoint
    print("training_arguments.pretrained_model_path")
    print(training_arguments.pretrained_model_path)

    model = AutoModelForCausalLM.from_pretrained(training_arguments.pretrained_model_path, trust_remote_code=True)
    config = model.config
    tokenizer = AutoTokenizer.from_pretrained(training_arguments.pretrained_model_path, use_fast=False, model_max_length = config.tokenizer_model_max_length,padding_side = config.tokenizer_padding_side)
    model.tokenizer = tokenizer
    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    
    
    vision_model_name_or_path= config.vision_model_name_or_path
    if custom_arguments.vision_model_name_or_path!="":
        vision_model_name_or_path= custom_arguments.vision_model_name_or_path
        
    #data_arguments.image_processor = AutoImageProcessor.from_pretrained(config.vision_model_name_or_path)
    data_arguments.image_processor = AutoImageProcessor.from_pretrained(vision_model_name_or_path)
    
    if custom_arguments.reset_imgnorm:
        data_arguments.image_processor.image_mean= [0.,0.,0.]
        data_arguments.image_processor.image_std= [1.,1.,1.]
        
    print("config")
    print(config)
    print("config.vision_model_name_or_path")
    print(config.vision_model_name_or_path)
    print("vision_model_name_or_path")
    print(vision_model_name_or_path)

    print("image_processor size/mean/std")
    print((data_arguments.image_processor.size["height"], data_arguments.image_processor.size["width"]))
    print(data_arguments.image_processor.image_mean)
    print(data_arguments.image_processor.image_std)

    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)
    log_trainable_params(model)  # not work well with zero3
    trainer = LLaVATrainer(model=model, #does not require model.to(device), huggingface/deepspeed does it for you?
                           tokenizer=tokenizer,
                           args=training_arguments,
                           **data_module)
    
    trainer.train()
    
    training_recipe.save(model, trainer)

if __name__ == "__main__":
    train()
