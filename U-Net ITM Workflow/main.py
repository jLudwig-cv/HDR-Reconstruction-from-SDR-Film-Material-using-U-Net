import argparse
import os

from train import train_model
from test import test_model
from utils import *
from save_images import npy_to_bwtiff, process_bw_tiff_folder_to_colour_avif, process_colour_tiff_folder_to_colour_avif

def main():
    # Dynamically set dataset paths based on the selected dataset option
    dataset_paths = get_dataset_paths(data_dir, args.dataset)
    
    if args.train:
        model_save_path = create_folder_model(model_base_dir, args.model)
        save_args_to_txt(args, os.path.dirname(model_save_path))

        if args.model == 1:
            convert_images_to_luminance_train(dataset_paths['dataset_dir'])
            train_model(dataset_paths['sdr_luminance_dir'], dataset_paths['hdr_luminance_dir'], model_save_path, 
                        model=args.model, batch_size=args.batch_size, num_epochs=args.num_epochs, 
                        patience=args.patience, lr=args.lr, loss_function=args.loss_function, max_nits=args.max_nits)
        elif args.model == 2:
            train_model(dataset_paths['sdr_colour_dir'], dataset_paths['hdr_colour_dir'], model_save_path, 
                        model=args.model, batch_size=args.batch_size, num_epochs=args.num_epochs, 
                        patience=args.patience, lr=args.lr, loss_function=args.loss_function, max_nits=args.max_nits)

    if args.test:
        model_path = get_latest_file(model_base_dir, args.model)
        test_output_dir = create_folder_test(test_base_dir, model_path, args.model)
        save_args_to_txt(args, test_output_dir)
        
        if args.model == 1:
            convert_images_to_luminance_test(dataset_paths['dataset_dir'])
            test_model(dataset_paths['sdr_test_luminance_dir'], model_path, test_output_dir, 
                       dataset_paths['hdr_test_colour_tiff_dir'], model_type=args.model, 
                       batch_size=args.batch_size, max_nits=args.max_nits)
            output_dir = get_latest_folder(test_base_dir, args.model)
            npy_to_bwtiff(output_dir, dataset_paths['hdr_test_luminance_dir'], max_nits=args.max_nits)
        elif args.model == 2:
            test_model(dataset_paths['sdr_test_colour_tiff_dir'], model_path, test_output_dir, 
                       dataset_paths['hdr_test_colour_tiff_dir'], model_type=args.model, 
                       batch_size=args.batch_size, max_nits=args.max_nits)

    if args.save_avifs:
        output_dir = get_latest_folder(test_base_dir, args.model)
        if args.model == 1:
            process_bw_tiff_folder_to_colour_avif(dataset_paths['hdr_test_colour_tiff_dir'], 
                                                  dataset_paths['sdr_test_colour_tiff_dir'], 
                                                  output_dir, max_nits=args.max_nits)
        elif args.model == 2:
            process_colour_tiff_folder_to_colour_avif(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test SDR to HDR conversion model")
    
    # General arguments
    parser.add_argument('--model',          type=int,    default=2,      help="1: luminance model; 2: color model")
    parser.add_argument('--train',          type=bool,   default=True,   help="Set to True to train the model")
    parser.add_argument('--test',           type=bool,   default=True,   help="Set to True to test the model")
    parser.add_argument('--save_avifs',     type=bool,   default=True,   help="Set to True to save AVIF images")
    
    # Training arguments
    parser.add_argument('--batch_size',     type=int,    default=2,      help="Size of Batch")
    parser.add_argument('--num_epochs',     type=int,    default=500,    help="Number of training epochs")
    parser.add_argument('--patience',       type=int,    default=100,    help="Patience for early stopping")
    parser.add_argument('--lr',             type=float,  default=1e-4,   help="Learning rate for the optimizer")
    parser.add_argument('--loss_function',  type=int,    default=2,      help="1: mse, 2: SSIM, 3: log, 4: mae")
    parser.add_argument('--max_nits',       type=int,    default=4000,   help="Peak Luminance Value of HDR Data")
    
    # Dataset selection argument
    parser.add_argument('--dataset',        type=int,    default=4,      help="1 = digital, 2 = analog , 3 = komplett, 4 = Hable digital, 5 = Reinhard digital, 6 = Möbius digital")
    
    args = parser.parse_args()

    # Base directory for data
    data_dir = r"U-Net_Upmapping_Workflow\data"
    
    # Base directories for models and outputs
    model_base_dir = r"U-Net_Upmapping_Workflow\model_output"
    test_base_dir = r"U-Net_Upmapping_Workflow\image_output"
    
    main()
