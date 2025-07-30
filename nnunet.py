#%%
print("executing 07_nnunet.py", __name__)

import os
run = 0
rabbit_data_path = "/mnt/data2/joe/data/data experimental/ex-vivo-rabbit/"
base = rabbit_data_path + 'nnunet/'
nnUNet_raw = base + "nnUNet_raw/"
nnUNet_preprocessed = base + "nnUNet_preprocessed/"
nnUNet_results = base + f"nnUNet_models_run{run:02}/"
os.environ["nnUNet_raw"] = nnUNet_raw
os.environ["nnUNet_preprocessed"] = nnUNet_preprocessed
os.environ["nnUNet_results"] = nnUNet_results
os.environ["nnUNet_compile"] = '0'

import torch
import numpy as np
import shutil, time, sys, subprocess, re, json, re, pprint, os
import SimpleITK as sitk

# from torch.utils.tensorboard import SummaryWriter
# from torch.fx import symbolic_trace
from torchview import draw_graph  # pip install torchview
# from torchviz import make_dot  # pip install torchviz
from torch.fx import symbolic_trace
from torch._dynamo import explain
from torchinfo import summary
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from pathlib import Path
from generate_mix_data_experimental_synthetic_v1 import augment_with_exclude_date
from matplotlib.colors import colorConverter
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgenerators.utilities.file_and_folder_operations import load_json
from batchgenerators.utilities.file_and_folder_operations import join
from support import *
from calculations import *
from transforms import *
print("Imports successfully imported!")


def main():
    global run
    
   # paths
    nnUNet_tests = base + "nnUNet_tests/"
    nnUNet_evaluation_metrics = base + "nnUNet_metrics/"
    nnUNet_cross_sections = base + "nnUNet_cross_sections/"
    nnUNet_videos = base + "nnUNet_videos/"
    nnUNet_videos_predictions = base + "nnUNet_videos_predictions/"
    nnUNet_mp4s = base + "nnUNet_mp4s/"
    nnUNet_gif_grid = base + "nnUNet_gifs_grid/"
    nnUNet_tests_npy = base + "nnUNet_tests_npy/"
    nnUNet_metrics_new = base + "nnUNet_metrics_new/"
    print(nnUNet_results, nnUNet_preprocessed)

    # Parameters for augmented data (If choose not to use NNUNET TRAINER with DA5 Augmentations)
    scaling_factors = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    buffer = 21 
    dimension = 64
    # Number of samples
    n_samples = 2
    n_exp_images = 2
    # ns = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900,
    #     1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    # ]


    # binary classification folders for dataset creation
    foldername_experimental = f'/mnt/data2/joe/data/data experimental/ex-vivo-rabbit/'


    # VALIDATION SCHEME: 100% old data, 87.5 %, 75%, ... , 0% old data

    # 100 % OLD DATA
    # validation_dates = sorted(["2023-11-21"])
    # segmentations_folder = rabbit_data_path + "segmentations npy (edited)/" 

    # 87.5% old data
    validation_dates = sorted(["2023-11-21"])
    segmentations_folder = "/mnt/data2/joe/data/data experimental/ex-vivo-rabbit/nnunet/opto-acoustic-replacement-sets/87.5% old data/segmentations/" # 87.5% old data

    segmentations_test_folder = foldername_experimental + "segmentations npy (original with optical aligned)/"




    # segmentations_folder = foldername_experimental + "segmentations (warped)/threshold_0.5/"

    ultrasound_videos_path = foldername_experimental + "ultrasound videos/"
    ultrasound_videos = get_npy_files_in_folder(ultrasound_videos_path)


    # Validation dates list
    # validation_dates = sorted(["2024-08-14", "2024-08-06", "2024-06-26", "2024-06-25", "2023-12-08", "2023-12-12"])
    # validation_dates = sorted(["2024-06-25"])
    # validation_dates = sorted(["2025-04-22"])
    # validation_dates = sorted(["None"])
    # validation_dates = sorted(["2024-08-14", "2024-08-13", "2024-08-06", "2024-06-26", "2024-06-25", "2023-12-08"])
    # validation_dates = sorted(["2023-11-21", "2024-08-14", "2024-08-06", "2024-06-25", "2023-12-08", "2023-12-12"])
    # validation_dates = get_unique_dates_in_folder(segmentations_folder)
    # validation_dates = sorted(["2024-06-26", "2024-08-06", "2025-04-22", "2025-05-15"])
    # validation_dates = sorted(["2023-11-21"])
    pprint.pprint(validation_dates)
    # sys.exit()

    # trinary classification folders
    # foldername_experimental = f"/mnt/data2/joe/data/data experimental/exs-vivo-rabbit-cavity/"
    # segmentations_folder = foldername_experimental + "segmentations trinary class/"

    # synthetic data for binary classification folders
    data_set_mat = "gt-us-2"
    foldername_matlab = f'/mnt/data1/joe/data/data synthetic/matlab-hearts/{data_set_mat}/'
    save_drive = "data2"
    foldernames = [foldername_experimental, foldername_matlab]

    # indicate which dataset you want to start the loop on


    # creating ultrasound videos
    create_video=False
    save_videos_as = "NIIGZ"
    convert_using = ".nii.gz" # stkio uses: .nii.gz, .nrrd, .mha

    # dataset creation
    check_segmentations_unique_binary_values(segmentations_folder)
    check_segmentations_unique_binary_values(segmentations_test_folder)


    # Dataset creation
    create_dataset=0
    start_dataset_num = 9999
    end_dataset_num = start_dataset_num
    edit_testing_dataset_only=0
    manual_augmentation=False
    # use_synthetic_images=True


    validate=0 # validation of the dataset (make sure it is good for training)
    plan_experiment_only=0


    # Training
    train=1 # training
    fine_tune=0 # fine tune the model
    predict=0 # get the test image predictions

    # evaluate metrics and predict entire video which will be based on metrics
    evaluate=0
    save_metrics=0
    save_plot=1
    predict_video=1 # Parameters for vtk movies and monochrome video
    monochrome_videos=1
    vtk_movies=0
    calibrate_vtk_camera=0
    frames = 200
    fps=25



    # Create Dataset
    target_dataset_id = start_dataset_num
    unique_dates = get_unique_dates_in_folder(rabbit_data_path + 'ultrasound videos/')

    if create_video:
        
        for us_file in ultrasound_videos:
            videos_path = nnUNet_videos + save_videos_as + "/" + us_file.replace(".npy", "/")
            print(videos_path)
            if not os.path.exists(videos_path):
                os.makedirs(videos_path)

            us_video = np.load(ultrasound_videos_path + us_file)
            us_video_length = len(us_video)

            for us_frame in range(us_video_length):
                us = us_video[us_frame]
                image = sitk.GetImageFromArray(us) # Convert the NumPy array to a SimpleITK image
                save_frame_path = videos_path + us_file.replace("_US", "-rec") + f"-frame{us_frame:03.0f}_0000{convert_using}"
                save_frame_path = save_frame_path.replace(".npy", "")
                sitk.WriteImage(image, save_frame_path)

    if create_dataset:

        if target_dataset_id > end_dataset_num:
            sys.exit()

        # Create a dataset that has one heart that you want to exclude
        if len(validation_dates) == 1:
            exclude = validation_dates[0]

        # or multiple hearts
        elif len(validation_dates) > 1:
            exclude = "("
            for date in validation_dates[:-1]:
                exclude += f"{date}_"
            exclude += validation_dates[-1]
            exclude += ")"

        # or if there are no hearts
        else:
            x = input("There are no hearts that will be validated during the training for the creation of the dataset. Proceed [y/n]")
            if x == "y":
                exclude = None
            else:
                print("terminating.")
                sys.exit()

        # create the dataset names and identifier tags
        if manual_augmentation:
            target_dataset_animal = f"RabbitHeart_exclude-{exclude}_{n_exp_images:05.0f}-{n_samples-n_exp_images:05.0f}mix"
            target_dataset_name = f'Dataset{target_dataset_id:03.0f}_{target_dataset_animal}'
        else:
            target_dataset_animal = f"RabbitHeart_exclude-{exclude}_{len(os.listdir(segmentations_folder)):05.0f}-{0:05.0f}mix"
            target_dataset_name = f'Dataset{target_dataset_id:03.0f}_{target_dataset_animal}'

        # Check to see if a dataset already exists
        datasets = sorted([nnUNet_raw + f for f in os.listdir(nnUNet_raw)])
        for dataset in datasets:
            if f"Dataset{target_dataset_id:03.0f}" in dataset:
                if not edit_testing_dataset_only:
                    confirm = input(f"Do you want to overwrite dataset {target_dataset_id:03.0f}? [y/n]")
                    if confirm == "y":
                        # Removing Raw
                        try:
                            shutil.rmtree(f"{dataset}/")
                            print(f"Removed directory          '{dataset}/'")
                        except FileNotFoundError as e:
                            print(f"Could not remove directory '{dataset}/'")

                        # Removing Preprocessed
                        try:
                            dataset = dataset.replace("nnUNet_raw", "nnUNet_preprocessed")
                            shutil.rmtree(f"{dataset}/")
                            print(f"Removed directory          '{dataset}/'")
                        except FileNotFoundError as e:
                            print(f"Could not remove directory '{dataset}/'")

                        # Removing results
                        try:
                            directory = nnUNet_results + f"{target_dataset_name}"
                            shutil.rmtree(f"{directory}/")
                            print(f"Removed directory          '{directory}/'")
                        except FileNotFoundError as e:
                            print(f"Could not remove directory '{directory}/'")


                        # Removing tests
                        try:
                            directory = nnUNet_tests + f"{target_dataset_name}"
                            shutil.rmtree(f"{directory}/")
                            print(f"Removed directory          '{directory}/'")
                        except FileNotFoundError as e:
                            print(f"Could not remove directory '{directory}/'")

                        # Removing metrics
                        # try:
                        #     dataset = dataset.replace("nnUNet_raw", "nnUNet_metrics")
                        #     shutil.rmtree(f"{dataset}/")
                        #     print(f"Removed directory          '{dataset}/'")
                        # except FileNotFoundError as e:
                        #     print(f"Could not remove directory '{dataset}/'")
                        

                        print()
                    else:
                        raise ValueError("Maybe think before you want to overwrite.")
                else:
                    x = input(f"Edit the testing dataset for dataset {target_dataset_id}? [y/n]")
                    if x.lower() != "y":
                        print("terminating")
                        sys.exit()

        print(f"Target dataset id: {target_dataset_id:03.0f}")
        print(f"Target dataset animal: {target_dataset_animal}")
        print(f"Target dataset name: {target_dataset_name}")
        print(f"Validation dates: {validation_dates}")
        # print("Exclude", exclude, exclude[1:-1].split("_"))

        if len(validation_dates) > 1:
            assert validation_dates == exclude[1:-1].split("_")
        maybe_mkdir_p(join(nnUNet_raw, target_dataset_name))
        imagesTr = join(nnUNet_raw, target_dataset_name, 'imagesTr/')
        labelsTr = join(nnUNet_raw, target_dataset_name, 'labelsTr/')
        imagesTs = join(nnUNet_raw, target_dataset_name, 'imagesTs/')
        labelsTs = join(nnUNet_raw, target_dataset_name, 'labelsTs/')
        labelsTs_figures = join(nnUNet_raw, target_dataset_name, 'labelsTs_figures/')
        maybe_mkdir_p(imagesTr)
        maybe_mkdir_p(labelsTr)
        maybe_mkdir_p(imagesTs)
        maybe_mkdir_p(labelsTs)
        maybe_mkdir_p(labelsTs_figures)
            
        # save path for the test figures, and the description of the dataset
        save_path = f"/mnt/{save_drive}/joe/data/data experimental/ex-vivo-rabbit/nnunet/nnUNet_raw/{target_dataset_name}/"


        # GENERATE THE TRAINING DATA

        # Manual augmentation
        if manual_augmentation:
            augment_with_exclude_date(foldernames, n_samples, 
                    dimension=dimension, 
                    scaling_factors=scaling_factors, 
                    buffer=buffer, 
                    rotation=False, 
                    noising=False, 
                    swap=False, 
                    limits=[n_exp_images, n_samples-n_exp_images],
                    imagesTr=imagesTr, labelsTr=labelsTr,
                    save_file_type=".nii.gz",
                    save_path=save_path, 
                    excluded_list=validation_dates, # the dates excluded from the hearts are those for the validation
                    segmentations_folder=segmentations_folder,
                    binary_close_sg=False,
                    normalize_us_min_max=True,
                    normalize_sg_min_max=False)
            # remove temporary files
            shutil.rmtree(nnUNet_raw + f"{target_dataset_name}/tmp/")
            
        # Not manual augmentation
        else:
            us_npy_path = rabbit_data_path + "ultrasound/"
            us_files = sorted([f for f in os.listdir(us_npy_path)])
            sg_files = sorted([f for f in os.listdir(segmentations_folder)])
            named_files = check_same_corresponding_names_in_two_folders(us_npy_path, segmentations_folder, return_names=True)
            print("NAMED FILES", named_files)
            
            # create the training images
            if not edit_testing_dataset_only:
                for us in us_files: 
                    if us in named_files:
                        if len(validation_dates) > 0:
                            if get_date_rec_frame_for_file_name_convention(us)[0] not in validation_dates:
                                convert(us_npy_path + us, imagesTr + f"{us}".replace(".npy", f"_0000{convert_using}"))
                        else:
                            convert(us_npy_path + us, imagesTr + f"{us}".replace(".npy", f"_0000{convert_using}"))

            # create the training labels
                for sg in sg_files: 
                    
                    if sg in named_files:
                        
                        if len(validation_dates) > 0:
                            if get_date_rec_frame_for_file_name_convention(sg)[0] not in validation_dates:
                                convert(segmentations_folder + sg, labelsTr + f"{sg}".replace(".npy", f"{convert_using}"))
                        else:
                            convert(segmentations_folder + sg, labelsTr + f"{sg}".replace(".npy", f"{convert_using}"))

            # print(len(get_files_in_folder(labelsTr)) == len(get_files_in_folder(imagesTr)))
            assert len(get_files_in_folder(labelsTr)) == len(get_files_in_folder(imagesTr))


        # Generate the testing data
        us_npy_path = rabbit_data_path + "ultrasound/"
        
        us_files = sorted([f for f in os.listdir(us_npy_path)])
        sg_files = sorted([f for f in os.listdir(segmentations_test_folder)])
        # print(len(us_files), len(sg_files))
        named_files = check_same_corresponding_names_in_two_folders(us_npy_path, segmentations_test_folder, return_names=True,
                                                                    verbose=True)
        # print("NAMED FILES", len(named_files))
        # training_files = get_files_in_folder(labelsTr)
        # print(labelsTr)
        unique_training_dates = get_unique_dates_in_folder(labelsTr, pattern=r'\d{4}-\d{2}-\d{2}', suffix=".nii.gz")
        # print("Training files", training_files)
        print("UNIQUE TRAINING DATES", unique_training_dates)

        
        
        for us in us_files: 
            if us in named_files:
                if len(validation_dates) > 0:
                    date, recording_number, frame_number = get_date_rec_frame_for_file_name_convention(us)
                    if date in validation_dates:
                        convert(us_npy_path + us, imagesTs + f"{us}".replace(".npy", f"_0000{convert_using}"))
                        convert(segmentations_test_folder + us, labelsTs + f"{us}".replace(".npy", f"{convert_using}"))
                else:
                    convert(us_npy_path + us, imagesTs + f"{us}".replace(".npy", f"_0000{convert_using}"))
                    convert(segmentations_test_folder + us, labelsTs + f"{us}".replace(".npy", f"{convert_using}"))



        # if "segmentations trinary class" in segmentations_folder:
        #     labels = {
        #             "background": 0,
        #             "tissue": 2,
        #             "cavity": 1
        #         }
        # else:
        labels = {
                "background": 0,
                "tissue": 1,
            }
        channel_names = {
            0 : "US"
        }

        cases = os.listdir(imagesTr)

        # Generate the nnUNet dataset json file
        generate_dataset_json(
            join(nnUNet_raw, target_dataset_name),
            channel_names,
            labels,
            len(cases),
            convert_using,
            None,
            target_dataset_name,
            overwrite_image_reader_writer='nnunetv2.imageio.simpleitk_reader_writer.SimpleITKIO',
            reference='https://aortaseg24.grand-challenge.org/',
            license='see ref'
        )

        print("\nDataset information")
        print(f"Animal: {target_dataset_animal}")
        print(f"ID: {target_dataset_id}")
        print(f"Path: {base}+ nnUNet_raw/{target_dataset_name}/")
        print("items: ", os.listdir(base + f'nnUNet_raw/{target_dataset_name}/labelsTr/'))
        with open(base + f"nnUNet_raw/{target_dataset_name}/dataset.json", "r") as f:
            json_data = json.load(f)
            for key, value in json_data.items():
                print(key, value)

    if not create_dataset:
        # select_datasets = [0, 5, 10, 15, 20, 21, 26, 31, 36, 41]
        # select_datasets = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 16, 17, 18, 19, 35] # training 500 epochs
        # select_datasets = [22, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34, 36, 37, 38, 39, 40,
        #                    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62] # training 500 epochs
        # select_datasets = [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62] # continuing training 500 epochs
        # select_datasets = [0, 5, 10, 14, 15, 20, 21, 26, 31, 35, 36, 41, 101, 999, 1000] # movies
        # select_datasets = range(42, 84) # verify dataset
        # select_datasets = [0, 5, 10, 14, 15, 20, 21, 26, 31, 36, 41] # training
        # select_datasets = [0, 5, 10, 14, 15, 20,=21, 26, 31, 36, 41] # evaluating
        # select_datasets = [999, 1000, 10001, 1002, 1003, 1004, 1005, 1006, 1020, 1021] # vtk mlovies
        select_datasets = [3000]
        # select_datasets = list(range(1001, 1021, 1)) # dataset IDs that will undergo augmentation automatically by nnunet in training
        # select_datasets = [0] # Datasets that do not have any augmentation
        # for dataset_text in [f for f in sorted(os.listdir(nnUNet_raw)) if "Dataset" in f]:
        for dataset_text in get_files_in_folder_condition(nnUNet_raw, "Dataset"):
            # print(dataset_text)
            if "None" in dataset_text:
                # print("None in dataset")
                # pattern = r'Dataset(\d+)_.*_exclude-([A-Za-z]+)_(\d{5}-\d{5}))'
                pattern = r"(\d+)_RabbitHeart_exclude-([A-Za-z]+)_(\d{5}-\d{5})"
                match = re.search(pattern, dataset_text)
                if match:
                    target_dataset_id = int(match.group(1))
                    exclude = match.group(2)
                    number_range = match.group(3)
                    target_dataset_animal = f"RabbitHeart_exclude-{exclude}_{number_range}mix"
                    # print(target_dataset_id, exclude, number_range, target_dataset_animal)
                    target_dataset_name = f'Dataset{target_dataset_id:03.0f}_{target_dataset_animal}'
                else:
                    raise ValueError("No match")
            else:
                # print(True)
                # pattern = r"(\d+)_RabbitHeart_exclude-(\d{4}-\d{2}-\d{2})_(\d{5}-\d{5})"
                # pattern = r"Dataset(\d+)_RabbitHeart_exclude-([^-_]+)_(\d+-\d+)mix"
                # pattern = r"Dataset(\d+)_RabbitHeart_exclude-\(([\d_-]+)\)_(\d+-\d+)mix"
                # pattern = r"Dataset(\d+)_RabbitHeart_exclude-\((\d{4}-\d{2}-\d{2}(?:_\d{4}-\d{2}-\d{2})*)\)_(\d+-\d+)mix"
                pattern = re.compile(
                    r"Dataset(\d+)_RabbitHeart_exclude-"
                    r"(?:\((\d{4}-\d{2}-\d{2}(?:_\d{4}-\d{2}-\d{2})+)\)|(\d{4}-\d{2}-\d{2}))"
                    r"_(\d+-\d+)mix"
                    # r"Dataset(\d+)_RabbitHeart_exclude-"
                    # r"(\(\d{4}-\d{2}-\d{2}-(?:_\d{4}-\d{2}-\d{2})+\)|\d{4}-\d{2}-\d{2})"
                    # r"_(\d+-\d+)mix"

                )


                match = re.search(pattern, dataset_text.split("/")[-1])
                # print(match)
                if match:
                    target_dataset_id = int(match.group(1))
                    exclude = match.group(2) if match.group(2) else match.group(3)
                    # print("exclude", exclude)
                    number_range = match.group(4)

                    if len(exclude) > 10:
                        target_dataset_animal = f"RabbitHeart_exclude-({exclude})_{number_range}mix"
                    else:
                        target_dataset_animal = f"RabbitHeart_exclude-{exclude}_{number_range}mix"
                    # print(target_dataset_animal)
                    target_dataset_name = f'Dataset{target_dataset_id:03.0f}_{target_dataset_animal}'
                else:
                    # print(pattern)
                    # print(dataset_text.split("/")[-1])
                    raise ValueError("No match")
            

            if target_dataset_id in select_datasets:

                print(f"Dataset ID: e: {target_dataset_id}")
                print(f"Type data percentage {number_range}")
                print(f"Excluded: {exclude}")
                # raise

                with open(nnUNet_raw + target_dataset_name + "/dataset.json", "r") as f:
                    data = json.load(f)

                    new_data_labels_dictionary = {}
                    data_labels = data["labels"]
                    for key, value in data_labels.items():
                        new_data_labels_dictionary[key.lower()] = value
                    del data["labels"]
                    data["labels"] = new_data_labels_dictionary

                    # print(data)
                with open(nnUNet_raw + target_dataset_name + "/dataset.json", "w") as output_file:
                    json.dump(data, output_file)


                # VERIFY DATASET      
                if validate:
                    #     from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
                    subprocess.call(["sh", "/mnt/data1/joe/code/07_nnunet_verify_dataset.sh", f"{target_dataset_id:03.0f}", nnUNet_raw, nnUNet_results, nnUNet_preprocessed])
                    #     dfe = DatasetFingerprintExtractor(dataset_name_or_id=9999, num_processes=1, verbose=False)

                    # from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints

                    # from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity
                    # from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import extract_fingerprints

                    
                    # extract_fingerprints(select_datasets, check_dataset_integrity=True)

                    # verify_dataset_integrity(nnUNet_raw + target_dataset_name)


                    # dfe.run(overwrite_existing=True)

                    # extract_fingerprints(dataset_ids=[9999],
                    #         # fingerprint_extractor_class_name='DatasetFingerprintExtractor',
                    #         check_dataset_integrity=True)

                    # sys.exit()

                # Call if you do not want to run preprocessing part, only plan again
                if plan_experiment_only:
                    planner = "nnUNetPlannerResEncL"
                    subprocess.call(["sh", "/mnt/data1/joe/code/07_nnunet_plan_experiment.sh", f"{target_dataset_id:03.0f}", nnUNet_raw, nnUNet_results, nnUNet_preprocessed, planner])

                    # sys.exit()

                # TRAIN
                if train:  

                    # Set up the training
                    nnUNet_trainer = "nnUNetTrainerDA5_10000epochs"
                    nnUNet_model_configuration = "3d_fullres"
                    fold = "all"
                    # fold = 5
                    planner = "nnUNetPlans"
                    # planner = 'nnUNetResEncUNetLPlans'

                    # Replace with actual paths and configuration names as per your setup 
                    # plans_path = f"{nnUNet_tests}/{target_dataset_name}/{nnUNet_trainer}__nnUNetPlans__{nnUNet_model_configuration}_run{run:02.0f}/plans.json" 
                    # dataset_json_path = f"{nnUNet_tests}/{target_dataset_name}/{nnUNet_trainer}__nnUNetPlans__{nnUNet_model_configuration}_run{run:02.0f}/dataset.json" 
                    # configuration = '3d_fullres'  # Example configuration, replace with your actual configuration fold = 0  # Example fold
                    # fold = 3

                    plans_path = f"{nnUNet_preprocessed}/{target_dataset_name}/{planner}.json" 
                    dataset_json_path = f"{nnUNet_preprocessed}/{target_dataset_name}/dataset.json" 

                    # Load plans and dataset_json 
                    plans = load_json(plans_path) 
                    dataset_json = load_json(dataset_json_path)

                    # Initialize the PlansManager and ConfigurationManager 
                    plans_manager = PlansManager(plans) 
                    configuration_manager = plans_manager.get_configuration(nnUNet_model_configuration)

                    

                    # Initialize the trainer
                    trainer = nnUNetTrainer(plans, nnUNet_model_configuration, fold, dataset_json)
                    # trainer._do_i_compile() = False
                    trainer.initialize()
                    # sys.exit()
                    # trainer.plot_network_architecture()

                    # import netron
                    # start_onnx = True
                    # if start_onnx:
                    #     netron.start("/mnt/data1/joe/code/network_architecture.onnx")
                    #     sys.exit()
                    
                    # print(trainer._do_i_compile())
                    
                    # print(trainer.network)
                    print(type(trainer), type(trainer.network))

                    # sys.exit()
                    
                    # raise

                    if nnUNet_preprocessed is None or nnUNet_results is None: 
                        raise ValueError("nnUNet_preprocessed and nnUNet_results must be set to valid paths.")
                    # Initialize the network 
                    

                    # Function to count parameters 
                    def count_parameters(model): 
                        total_params = sum(p.numel() for p in model.parameters()) 
                        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
                        non_trainable_params = total_params - trainable_params
                        return total_params, trainable_params, non_trainable_params

                    # Get the number of parameters 
                    total, trainable, non_trainable = count_parameters(trainer.network)
                    print(f"Total: {total}")
                    print(f"Trainable: {trainable}")
                    print(f"Non-trainable: {non_trainable}")

                    verbose=False
                    if verbose:



                        dummy_input = torch.randn(1, 1, 128, 128, 128)  # adjust size

                        model = trainer.network
                        print("Type of model", type(model))
                        assert hasattr(model, 'forward'), "Model is missing forward()"
                        print("Model type:", type(model))
                        print("Does model have forward?", hasattr(model, 'forward'))
                        print("Is forward callable?", callable(getattr(model, 'forward', None)))
                        # print("Forward method:", getattr(model, 'forward', None))

                        torch.save(model, f"{nnUNet_model_configuration}.pth")
                        model = torch.load(f"{nnUNet_model_configuration}.pth", weights_only=False)
                        model = torch.compile(model)

                        
                        original_model = model._orig_mod
                        # print(original_model)
                        with open("model.txt", "w") as f:
                            f.write(f"{original_model}")
                        print("ORIGINAL MODEL type", type(original_model))
                        original_model = original_model.cpu().eval()
                        traced = symbolic_trace(original_model)
                        print(f"TRACED TYPE: {type(traced)}")


                    
                        print("Model Summary Info:\n")

                        # Number of parameters (total and trainable)
                        total_params = sum(p.numel() for p in traced.parameters())
                        trainable_params = sum(p.numel() for p in traced.parameters() if p.requires_grad)
                        print(f"Total parameters: {total_params:,}")
                        print(f"Trainable parameters: {trainable_params:,}")

                        # Number of buffers
                        buffers = list(traced.buffers())
                        print(f"Number of buffers: {len(buffers)}")

                        # Number of modules and named modules
                        modules = list(traced.modules())
                        named_modules = list(traced.named_modules())
                        print(f"Number of modules: {len(modules)}")
                        print(f"Number of named modules: {len(named_modules)}")

                        # Number of named parameters
                        named_params = list(traced.named_parameters())
                        print(f"Number of named parameters: {len(named_params)}")

                        # Devices used by parameters (if consistent)
                        devices = set(p.device for p in traced.parameters())
                        print(f"Devices used by parameters: {[str(d) for d in devices]}")

                        # Check if model is in train or eval mode
                        print(f"Model training mode: {traced.training}")

                        # Size of graph nodes
                        if hasattr(traced, 'graph'):
                            nodes = list(traced.graph.nodes)
                            print(f"Number of nodes in FX graph: {len(nodes)}")

                        # Show first few modules (for human overview)
                        # print("\nnamed modules:")
                        # total = 0
                        # for num, (name, mod) in enumerate(named_modules[:]):
                        #     if type(mod) != torch.nn.modules.module.Module:
                        #         print(f"{total:02.0f} - {name}: {type(mod)}")#, {type(str(type(mod)))}")
                        #         total += 1
                        #     else:
                        #         print(f"   - {name}: {type(mod)}")


                        # Show first few named parameters
                        # print("\nnamed parameters:")
                        # total = 0
                        # for name, param in named_params[:]:
                        #     print(f"- {name}: shape {tuple(param.shape)}")
                        #     total += 1

                        # total_params = 0
                        # print("\nNamed parameters and their shapes:")
                        # for name, param in named_params:
                        #     shape = tuple(param.shape)
                        #     num_params = 1
                        #     for dim in shape:
                        #         num_params *= dim
                        #     total_params += num_params
                        #     print(f" - {name}: shape {shape}, params: {num_params}")

                        # print(f"\nTotal number of parameters: {total_params:,}")


                        # Show the forward method info
                        # if hasattr(traced, 'forward'):
                        #     print("\nModel has 'forward' method:", callable(traced.forward))

                        # Show device of example parameter
                        # if named_params:
                        #     print(f"Example parameter device: {named_params[0][1].device}")




                    
                        # print(original_model.)
                        with torch.no_grad():
                            output = original_model(dummy_input)
                            output = output[0]
                        print("Output shape:", output.shape)
                        traced = symbolic_trace(original_model)
                        draw_graph(original_model, input_data=dummy_input, save_graph=True, filename="model_graph", expand_nested=True)

                        summ = summary(original_model, input_size=(1, 1, 128, 128, 128))
                        print(f"\nTotal parameters: {summ.total_params:,}")
                        print(f"Trainable parameters: {summ.trainable_params:,}")
                        print(f"Input size (MB): {summ.total_input:.2f}")
                        # print(f"Output size (MB): {summ.total_output_bytes:.2f}")
                        # print(f"Parameter size (MB): {summ.total_param_bytes:.2f}")
                        # print(f"Total estimated size (MB): {summ.total_input + summ.total_output_bytes + summ.total_param_bytes:.2f}")
                        # print(f"Total mult-adds (GB): {summ.total_mult_adds / 1e9:.2f}\n")

                        # Optionally: print readable summary lines
                        if verbose:
                            for layer in summ.summary_list[:10]:  # Modify the slice for more layers
                                if len(output) != 0:
                                    print(f"{layer.class_name:20} | output: {layer.output_size} | params: {layer.num_params}")

                        # explanation = explain(model._orig_mod.cpu(), dummy_input)

                        print("PyTorch version:", torch.__version__)
                        print("Compiled with CUDA:", torch.version.cuda)
                        print("CUDA available:", torch.cuda.is_available())




                    # sys.exit()
                    try:
                        subprocess.call(["sh", "/mnt/data1/joe/code/07_nnunet_train_model.sh", f"{target_dataset_id:03.0f}", nnUNet_model_configuration, nnUNet_trainer, nnUNet_raw, nnUNet_results, nnUNet_preprocessed, str(fold), planner])
                    
                        validation_folder = nnUNet_results + target_dataset_name + "/" + f'{nnUNet_trainer}__nnUNetPlans__{nnUNet_model_configuration}' + "/" + f"fold_{fold}/validation/"
                        # os.makedirs(validation_folder, exist_ok=True)
                        validation_masks = [f for f in os.listdir(validation_folder) if f.endswith(".nii.gz")]
                        

                        for validation_mask in validation_masks:
                            print(validation_mask)
                            d, rn, fn = get_date_rec_frame_for_file_name_convention(validation_mask, pattern=r'(\d{4}-\d{2}-\d{2})-rec(\d{2})-frame(\d{3})\.nii.gz')
                            
                            print(d, rn, fn)

                            c = compute_centroid(convert2npy(validation_folder + validation_mask))

                            us = crop_array(min_max_normalize_array(convert2npy(nnUNet_videos + f"NIIGZ/{d}_US{rn}/{d}-rec{rn}-frame{fn}_0000.nii.gz")), c, 32)
                            validation_sg = crop_array(min_max_normalize_array(convert2npy(validation_folder + validation_mask)), c, 32)
                            validation_gt = crop_array(min_max_normalize_array(convert2npy(nnUNet_raw + f"{target_dataset_name}/labelsTr/{d}-rec{rn}-frame{fn}.nii.gz")), c, 32)

                            view_us_gt_pred_plot(us, validation_gt, validation_sg, center=compute_centroid(validation_gt), row2_title="Val Pred", row3_title="Val Gt", default_title_score="F-score", title=f"{d}-rec{rn}-frame{fn}.nii.gz", save_path=validation_folder + f"{d}-rec{rn}-frame{fn}.png")

                    except Exception as e:
                        print(e)
                        print("training stopped.")

                if fine_tune:

                    weights_dataset_name = "Dataset1020_RabbitHeart_exclude-2023-11-21_00005-00000mix"
                    weights_trainer = "nnUNetTrainerDA5_1000epochs"
                    weights_model_config = "nnUNetPlans__3d_fullres"
                    weights_run = 0
                    weights_fold = "all"
                    weights = nnUNet_tests + f"{weights_dataset_name}/{weights_trainer}__nnUNetPlans__{weights_model_config}_run{weights_run:02.0f}/fold_{weights_fold}/checkpoint_best.pth"
                    
                    subprocess.call(["sh", "/mnt/data1/joe/code/07_nnunet_fine_tune.sh", f"{target_dataset_id:03.0f}", nnUNet_model_configuration, nnUNet_trainer, nnUNet_raw, nnUNet_results, nnUNet_preprocessed, weights])


                # LOOP THROUGH THE TRAINERS AND THE MODEL ARCHITECTURES HERE:
                valid_nnUNet_full_model_plans_name = []
                valid_nnUNet_folds = []
                valid_nnUNet_trainer = []
                valid_nnUNet_model_confiruation = []
                valid_nnUNet_plans = []

                all_model_configurations = ['3d_fullres', '2d', "custom", "3d_ResidualUNet", "3d_ResidualEncoderUNet", ]
                all_folds = ["all", "5", "4", "3", "2", "1"]
                all_trainers = ["nnUNetTrainer_250epochs", "nnUNetTrainer_500epochs", "nnUNetTrainer_750epochs", "nnUNetTrainer_1000epochs", "nnUNetTrainerDA5_1000epochs", "nnUNetTrainerDA5_1500epochs", "nnUNetTrainerDA5_2000epochs", "nnUNetTrainerDA5_10epochs", "nnUNetTrainer_5epochs",
                                "nnUNetTrainerDA5_Custom"]
                all_plans = ["nnUNetPlans", "nnUNetResEncUNetLPlans"]

                for nnUNet_plan in all_plans:
                    for nnUNet_trainer in all_trainers:
                        for nnUNet_model_configuration in all_model_configurations:
                            for fold in all_folds:        
                                if fold == "all":
                                    nnUNet_full_model_plans_name = f'{nnUNet_trainer}__{nnUNet_plan}__{nnUNet_model_configuration}_run{run:02.0f}'
                                else:
                                    nnUNet_full_model_plans_name = f'{nnUNet_trainer}__{nnUNet_plan}__{nnUNet_model_configuration}__fold{fold}_run{run:02.0f}'

                                valid_nnUNet_full_model_plans_name.append(nnUNet_full_model_plans_name)
                                valid_nnUNet_folds.append(fold)
                                valid_nnUNet_trainer.append(nnUNet_trainer)
                                valid_nnUNet_model_confiruation.append(nnUNet_model_configuration)
                                valid_nnUNet_plans.append(nnUNet_plan)
                
                for nnUNet_full_model_plans_name, nnUNet_fold, nnUNet_trainer, nnUNet_model_configuration, nnUNet_plan in zip(valid_nnUNet_full_model_plans_name,
                valid_nnUNet_folds,
                valid_nnUNet_trainer,
                valid_nnUNet_model_confiruation,
                valid_nnUNet_plans
                ):

                    prediction_dir = nnUNet_tests + f"{target_dataset_name}/{nnUNet_full_model_plans_name}/"
                    videos_predictions_dir = nnUNet_videos_predictions + f"{target_dataset_name}/{nnUNet_full_model_plans_name}/"

                    # PREDICT
                    if predict:

                        try:
                            model_weights_path = nnUNet_results + f"{target_dataset_name}/{nnUNet_trainer}__{nnUNet_plan}__{nnUNet_model_configuration}/fold_{nnUNet_fold}/checkpoint_final.pth"
                            model_weights_path_best = nnUNet_results + f"{target_dataset_name}/{nnUNet_trainer}__{nnUNet_plan}__{nnUNet_model_configuration}/fold_{nnUNet_fold}/checkpoint_best.pth"
                            

                            # TODO: toggle between final weights and best weights for predictions
                            # if os.path.isfile(model_weights_path_best):
                            #     print(True, model_weights_path_best)
                                # use_best = True
                            # else:
                            print(os.path.isfile(model_weights_path), model_weights_path, nnUNet_fold)
                            # sys.exit()
                            if os.path.isfile(model_weights_path) or os.path.isfile(model_weights_path_best):
                                
                                
                                

                                try:                        
                                    prediction_files = [prediction_dir + f for f in os.listdir(prediction_dir)]
                                    for f in prediction_files:
                                        os.remove(f)
                                except FileNotFoundError:
                                    pass

                                # Show the prediction files
                                prediction_files = sorted(os.listdir(nnUNet_raw + f"{target_dataset_name}/imagesTs/"))
                                for file in prediction_files:
                                    array_test = convert2npy(nnUNet_raw + f"{target_dataset_name}/imagesTs/" + file, save_path=False)
                                    print(file,  ".shape ", array_test.shape, len(np.unique(array_test)))
                                time.sleep(3)

                                # predict frames
                                # TODO: rework the nnUNet_predict shell script
                                # need: nnUNet_raw, nnUNet_preprocessed, nnUNet_results, input dir, output dir, dataset id, model_config, trainer, checkpoint
                                # if nnUNet_fold == "all":
                                print(target_dataset_name)
                                print("uununet fold", nnUNet_fold)
                                subprocess.call(["sh", 
                                                    "/mnt/data1/joe/code/07_nnunet_predict.sh", 
                                                    nnUNet_raw, # 1.
                                                    nnUNet_preprocessed, #2.
                                                    nnUNet_results, #3.

                                                    # 4. input dir
                                                    f"/mnt/data2/joe/data/data experimental/ex-vivo-rabbit/nnunet/nnUNet_raw/{target_dataset_name}/imagesTs", # input dir
                                                    
                                                    # 5. dataset id
                                                    f"{target_dataset_id:03.0f}",

                                                    # 6. output dir
                                                    f"/mnt/data2/joe/data/data experimental/ex-vivo-rabbit/nnunet/nnUNet_tests/{target_dataset_name}/{nnUNet_trainer}__{nnUNet_plan}__${nnUNet_model_configuration}_run{run:02.0f}/",
                                                    
                                                    # 7. Model configuration
                                                    nnUNet_model_configuration, 

                                                    # 8. trainer
                                                    nnUNet_trainer,

                                                    # 9. target dataset name
                                                    target_dataset_name, 

                                                    # 10. checkpoint
                                                    f"{nnUNet_results}{target_dataset_name}/{nnUNet_trainer}__{nnUNet_plan}__{nnUNet_model_configuration}/fold_all/checkpoint_best.pth",
                                                    # "checkpoint.pth",

                                                    # 11. fold
                                                    f"{nnUNet_fold}",
                                                    ])
                                # else:
                                #     subprocess.call(["sh", "/mnt/data1/joe/code/07_nnunet_predict_fold.sh", f"{target_dataset_id:03.0f}", nnUNet_model_configuration, nnUNet_trainer, nnUNet_raw, nnUNet_results, nnUNet_preprocessed, target_dataset_name, f"{run:02.0f}", str(fold)])
                                # sys.exit()
                                prediction_files = sorted([f for f in os.listdir(prediction_dir) if f.endswith(".nii.gz")])
                                for file in prediction_files:

                                    niigz_npy_pred = convert2npy(prediction_dir + file, "", save_path=False)
                                    print(f"Prediction shape {file}: {niigz_npy_pred.shape}")
                                    os.rename(prediction_dir + file, prediction_dir + file.replace(".nii.gz", "_0000.nii.gz"))

                                    

                            else:
                                # print(f"Model did not train with this confiruation: {nnUNet_full_model_plans_name}")
                                pass

                        
                        except Exception as e:
                            print(e)

                    
                    # # EVALUATION
                    if (evaluate and os.path.exists(prediction_dir)):
                        # print(nnUNet_fold)
                        # print(os.path.exists(prediction_dir), prediction_dir)
                        create_nonexisting_directory(prediction_dir)
                        # convert_using = ".nii.gz" # stkio uses: .nii.gz, .nrrd, .mha

                        metric0 = np.array([])
                        metric1 = np.array([])
                        metric2 = np.array([])
                        metric3 = np.array([])
                        metric4 = np.array([])
                        metric5 = np.array([])

                        

                        prediction_files = sorted(os.listdir(nnUNet_raw + f"{target_dataset_name}/imagesTs/"))
                        prediction_labels = sorted(os.listdir(nnUNet_raw + f"{target_dataset_name}/labelsTs/"))


                        unique_dates = []

                        for file_name, file_name_test_label in zip(prediction_files, prediction_labels):
                            match = re.search(r'(\d{4}-\d{2}-\d{2})-rec(\d{2})-frame(\d{3})_0000.nii.gz', file_name)

                            # Extracted values
                            date = match.group(1) if match else None
                            recording_number = match.group(2) if match else None
                            frame_number = int(match.group(3) if match else None)
                            unique_dates.append(date)

                            filename_sg = f"/mnt/data2/joe/data/data experimental/ex-vivo-rabbit/segmentations npy (original with optical aligned)/{date}-rec{recording_number}-frame{frame_number:03.0f}.npy"
                            sg_test = np.load(filename_sg)
                            if not os.path.exists(filename_sg):
                                filename_sg = nnUNet_raw + f"{target_dataset_name}/labelsTs/" + file_name_test_label
                                sg_test = convert2npy(filename_sg, verbose=True)
                            filename_us = f"/mnt/data2/joe/data/data experimental/ex-vivo-rabbit/ultrasound videos/{date}_US{recording_number}.npy"

                            print(filename_sg)
                            
                            # print(np.unique(sg_test), sg_test.dtype, sg_test.shape)
                            us_test = np.load(filename_us)
                            us_test.astype(np.float32) # convert to float32
                            sg_test.astype(np.uint8)
                            
                            us_test = min_max_normalize_array(us_test) # normalize
                            sg_test = min_max_normalize_array(sg_test)
                            us_video_length = len(us_test)

                            us_test_original = us_test.copy()
                            sg_test_original = sg_test.copy()
                            us_pyramid_original = np.zeros(us_test_original.shape)

                            prediction_sg = prediction_dir + f"/{date}-rec{recording_number}-frame{frame_number:03.0f}_0000{convert_using}"
                            prediction = convert2npy(prediction_sg, prediction_sg.replace(convert_using, ".npy"))
                            create_nonexisting_directory(nnUNet_tests_npy + f"segmentations/{target_dataset_name}/{nnUNet_full_model_plans_name}/")
                            create_nonexisting_directory(nnUNet_tests_npy + f"ultrasound/{target_dataset_name}/{nnUNet_full_model_plans_name}/")                         
                            np.save(nnUNet_tests_npy + f"segmentations/{target_dataset_name}/{nnUNet_full_model_plans_name}/{date}-rec{recording_number}-frame{frame_number:03.0f}.npy", prediction)
                            np.save(nnUNet_tests_npy + f"ultrasound/{target_dataset_name}/{nnUNet_full_model_plans_name}/{date}-rec{recording_number}-frame{frame_number:03.0f}.npy", us_test[int(frame_number)])
                            np.save(nnUNet_tests_npy + f"segmentations (all)/{target_dataset_name}_{nnUNet_full_model_plans_name}_{date}-rec{recording_number}-frame{frame_number:03.0f}.npy", prediction)
                            np.save(nnUNet_tests_npy + f"ultrasound (all)/{target_dataset_name}_{nnUNet_full_model_plans_name}_{date}-rec{recording_number}-frame{frame_number:03.0f}.npy", us_test[int(frame_number)])
                            prediction_original = prediction.copy()
                            print("prediction shape", prediction_original.shape, prediction_original.dtype)
                            print("sg_test shape  ", sg_test_original.shape, sg_test_original.dtype)
                            # print(prediction_sg, filename_sg)

                            

                            
                            """
                            Trim the grounth truth segmentation based on
                            the pyramid boundary
                            """
                            us_pyramid_mask = binary_thresholding_greater_than(us_test[frame_number, :, :, :], 0.0)
                            sg_test = apply_mask(sg_test, us_pyramid_mask)

                            """
                            Test cropped image based on the centroid of the white heart pixels of the target
                            """
                            centroid = compute_centroid(sg_test)
                            # print(sg_test.shape, centroid)

                            

                            crop = True

                            # TODO: THIS IS SUPREMELY IMPORTANT FOR JAN!
                            # crop the segmentation video, and insert it into the black pyramid of the ultrasound video
                            if crop:
                                shift = 32 # num pixels deviate from the centroid (image size = shift*2 x shift*2 x shift*2)
                                # shift = 0 # crop entire enumpy array

                                if shift < 0:
                                    raise ValueError("Choose a positive number for shift")
                                elif shift != 0:
                                    try:
                                        sg_test    = crop_array(sg_test,    centroid, shift)
                                        print(sg_test.shape)
                                        prediction = crop_array(prediction, centroid, shift)
                                        us_test    = crop_array(us_test,    centroid, shift)
                                    
                                    except IndexError:
                                        raise IndexError("Decrease or increase the shift integer")
                                elif shift == 0:
                                    print("USING ENTIRE NUMPY ARRAY")
                                else:
                                    raise ValueError("What did you input for shift?")

                            # print(us_test.shape, sg_test.shape, prediction.shape)

                            # print("True")
                            if predict_video:
                                print("predicting video!")

                                # TODO: make model perform on all the ultrasound videos that it did not test on.
                                # or at least all the ultrasound videos in the arsenal.

                                pattern = r"(\d{4}-\d{2}-\d{2}_US\d{2})"
                                match = re.search(pattern, filename_us)
                                if match:

                                    us_recording = match.group(0)

                                    niigz_video_pred_frames_path = videos_predictions_dir + us_recording + "/NIIGZ cropped video frames pred/"
                                    npy_video_pred_frames_path = videos_predictions_dir + us_recording + "/NPY cropped video frames pred/"
                                    niigz_cropped_video_path = videos_predictions_dir + us_recording + "/NIIGZ cropped video frames raw/"
                                    npy_video_pred_4D_path = videos_predictions_dir + us_recording + "/" #"/NPY cropped video 4D pred/"
                                    npy_video_4D_path = videos_predictions_dir + us_recording + "/" #NPY cropped video 4D/"

                                    create_nonexisting_directory(npy_video_pred_4D_path)
                                    create_nonexisting_directory(niigz_video_pred_frames_path)
                                    create_nonexisting_directory(npy_video_pred_frames_path)
                                    create_nonexisting_directory(niigz_cropped_video_path)
                                    create_nonexisting_directory(npy_video_4D_path)

                                    view_us_gt_mask_plot(us_test_original[int(frame_number), :, :, :], prediction_original, (int(us_test_original.shape[1]/2), int(us_test_original.shape[2]/2), int(us_test_original.shape[3]/2)), 
                                                            title=f"{date} {recording_number} {frame_number} pred", save_path=videos_predictions_dir + f'{date}_US{recording_number}' + "/" + f"{date}-rec{recording_number}-frame{frame_number}_pred.png")

                                    ####################
                                    # TODO: check to see if the niigz prediction frames are already there
                                    # if not os.path.exists(npy_video_pred_4D_path + us_recording + "_pred.npy"):
                                    if not os.path.exists(npy_video_pred_frames_path) or (len(os.listdir(npy_video_pred_frames_path)) == 0):                                        

                                        for us_frame in range(us_video_length):
                                            """
                                            This is an interesting thing for nnUNet
                                            it is finicky about the type of input it gets
                                            change video from us_test_original (not cropped pyramid) to
                                            us_test (cropped pyramid) and you will see that the network
                                            has difficulty with the cropped shape.
                                            make sure to change the container shape as well.
                                            """
                                            # print(us_frame)
                                            if not os.path.exists(niigz_cropped_video_path + us_recording.replace("_US", "-rec") + f"-frame{us_frame:03.0f}" + "_0000.nii.gz"):
                                                us = us_test_original[us_frame]
                                                print("converting", us_recording, "frame", us_frame, "us_frame shape", us.shape, us.dtype)
                                                image = sitk.GetImageFromArray(us) # Convert the NumPy array to a SimpleITK image
                                                sitk.WriteImage(image, niigz_cropped_video_path + us_recording.replace("_US", "-rec") + f"-frame{us_frame:03.0f}" + "_0000.nii.gz")
                                            

                                        print("str fold", str(nnUNet_fold))
                                        # sys.exit()
                                        subprocess.call(["sh", "/mnt/data1/joe/code/07_nnunet_predict_video.sh", nnUNet_raw, nnUNet_results, nnUNet_preprocessed, f"{target_dataset_id:03.0f}", nnUNet_full_model_plans_name, target_dataset_name, us_recording, nnUNet_trainer, nnUNet_model_configuration, str(nnUNet_fold)])


                                        niigz_video_pred_frames_list = [f for f in sorted(os.listdir(niigz_video_pred_frames_path)) if f.endswith(".nii.gz")]
                                        # print(niigz_video_pred_frames_path)

                                        for niigz_pred_frame in niigz_video_pred_frames_list:
                                            convert2npy(niigz_video_pred_frames_path + niigz_pred_frame, npy_video_pred_frames_path + niigz_pred_frame.replace(".nii.gz", ".npy"), save_path=True)

                                        for npy_file in sorted([f for f in os.listdir(npy_video_pred_frames_path) if f.endswith(".npy")]):
                                            print(npy_file, np.load(npy_video_pred_frames_path + npy_file).shape)

                                    else:
                                        print(f'Model already performed on ultrasound video and created prediction: {npy_video_pred_4D_path + us_recording + "_pred.npy"}') 
                                        # sys.exit()
                                        print("Not converting over again")
                                        # time.sleep(3)
                                        # for npy_file in sorted([f for f in os.listdir(npy_video_pred_frames_path) if f.endswith(".npy")]):
                                        #     print(npy_file, np.load(npy_video_pred_frames_path + npy_file).shape)
                                        # x = input("Was this the shape you wanted?")
                                        # if x == "n":
                                        #     sys.exit()

                                    
                                    ###################

                                    # Container to store the video predictions
                                    pred_container = np.zeros(us_test_original.shape)
                                    npy_video_pred_frames_list = sorted(os.listdir(npy_video_pred_frames_path))
                                    for i, npy_file in enumerate(npy_video_pred_frames_list):
                                        npy_frame = np.load(npy_video_pred_frames_path + npy_file)
                                        # """
                                        # THE CENTROID CALCULATION WAS USED HERE TO PREDICT THE US VIDEO!
                                        # """
                                        # print(i, centroid[0], centroid[1], centroid[2])
                                        pred_container[i] = npy_frame#, centroid[0]-shift:centroid[0]+shift,centroid[1]-shift:centroid[1]+shift,centroid[2]-shift:centroid[2]+shift] = npy_frame 
                                    np.save(npy_video_pred_4D_path + us_recording + "_pred.npy", pred_container.astype(np.uint8))
                                    # np.save(npy_video_pred_4D_path + us_recording + "_pred.npy", pred_container.astype(np.uint8))
                                    print(npy_video_pred_4D_path + us_recording + "_pred.npy")
                                    # print("Unique pixel values in prediction container", np.unique(pred_container))

                                    # Container to store the video ultrasound
                                    us_container = us_test_original
                                    niigz_cropped_video_frames_list = sorted(os.listdir(niigz_cropped_video_path))
                                    for i, niigz_file in enumerate(niigz_cropped_video_frames_list):
                                        image = sitk.ReadImage(niigz_cropped_video_path + niigz_file)
                                        img_array = sitk.GetArrayFromImage(image)
                                        us_container[i] = img_array #, centroid[0]-shift:centroid[0]+shift,centroid[1]-shift:centroid[1]+shift,centroid[2]-shift:centroid[2]+shift] = img_array

                                    """
                                    This is why the array US video is more gray and stuff
                                    """
                                    us_container = (us_container*255).astype(np.uint8)
                                    # us_container = us_container
                                    np.save(npy_video_4D_path + us_recording + ".npy", us_container)
                                    print(npy_video_4D_path + us_recording + ".npy")

                                    # Saving the movie predictions into one master folder
                                    np.save(nnUNet_tests_npy + f"segmentations movies (all)/{target_dataset_name}_{nnUNet_full_model_plans_name}_{date}-rec{recording_number}-frame{frame_number:03.0f}.npy", pred_container.astype(np.uint8))
                                    np.save(nnUNet_tests_npy + f"ultrasound movies (all)/{target_dataset_name}_{nnUNet_full_model_plans_name}_{date}-rec{recording_number}-frame{frame_number:03.0f}.npy", us_container.astype(np.uint8))


                                    # sys.exit()
                                    # Make monochrome movies
                                    if monochrome_videos:
                                        # if not os.path.exists(npy_video_pred_frames_path) and (not len(os.listdir(npy_video_pred_frames_path)) == 0):
                                        view_in_monochrome(npy_video_4D_path + us_recording + ".npy",
                                                    npy_video_pred_4D_path + us_recording + "_pred.npy",
                                                    centroid,
                                                    save_file_path=nnUNet_mp4s + target_dataset_name + "_" + nnUNet_full_model_plans_name + "_" + us_recording + "_",
                                                    fps=34)                               
                                    
                                    if vtk_movies:
                                        # if not os.path.exists("/mnt/data2/joe/data/data experimental/ex-vivo-rabbit/nnunet/nnUNet_gifs_grid/" + f"{nnUNet_trainer}__nnUNetPlans__{nnUNet_model_configuration}_pred_run{run:02.0f}" + ".gif"):
                                        print("Making vtk movie")
                                        # print(True)
                                        # For the vtk script call
                                        gif_name = f"{target_dataset_name}_{nnUNet_full_model_plans_name}_{us_recording}_pred_{int(run):02.0f}.gif"
                                        print(gif_name)
                                        print(gif_name in os.listdir(nnUNet_gif_grid))
                                        # sys.exit()
                                        # raise
                                            
                                        if gif_name not in os.listdir(nnUNet_gif_grid):
                                            subprocess.call(["sh", "/mnt/data1/joe/code/07_nnunet_vtk_shell.sh", str(frames), str(run), us_recording, nnUNet_full_model_plans_name, target_dataset_name, str(bool(calibrate_vtk_camera))])

                                        # sys.exit()

                                    if (vtk_movies and calibrate_vtk_camera):
                                        proceed = input("Finished calibration? [y/n]")
                                        if proceed == "y" or proceed == "Y":
                                            print("Alert: Terminating code. ")
                                            print("Proposal: Restart with calibration toggle turned off")
                            # sys.exit()
                            
                            us_test = us_test[:,:,:,:]
                            # us_test = us_test / np.max(us_test)
                            synth_us_3D = us_test[frame_number,:,:,:] # synthetic ultrasound data. 
                            synth_sg_3D = sg_test[:,:,:] # synhetic segmentation / mask (ground-truth)
                            pred_sg_3D = prediction

                            # For the binary predictions
                            if len(np.unique(pred_sg_3D)) == 2:

                                pred_sg_3D[pred_sg_3D > 0.0] = 1.0 # HAS TO BE BINARY
                                pred_sg_3D[pred_sg_3D <= 0.0] = 0.0 
                                synth_sg_3D[synth_sg_3D > 0.0] = 1.0
                                synth_sg_3D[synth_sg_3D <= 0.0] = 0.0

                                # precision, recall, accuracy, error, f-score, total population
                                # m0, m1, m2, m3, m4, m5 = calculate_metrics(synth_sg_3D, pred_sg_3D, verbose=False)
                                metrics_dict = calculate_metrics(synth_sg_3D, pred_sg_3D, verbose=False)
                                m0 = metrics_dict["recall"] 
                                m1 = metrics_dict["precision"]
                                m2 = metrics_dict["fall_out"]
                                m3 = metrics_dict["miss_rate"]
                                m4 = metrics_dict["f_score"]
                                m5 = metrics_dict["accuracy"]

                                new_metrics_dict = nnUNet_metrics_new + f"run_{run:02.0f}/" + f'excluded_{exclude}/' + nnUNet_full_model_plans_name + "/" + target_dataset_name + "/"
                                create_nonexisting_directory(new_metrics_dict)
                                np.save(new_metrics_dict + f"{date}-rec{recording_number}-frame{frame_number:03.0f}_fscore.npy", m4)


                            metric0 = np.concatenate((metric0, np.array([m0])))
                            metric1 = np.concatenate((metric1, np.array([m1])))
                            metric2 = np.concatenate((metric2, np.array([m2])))
                            metric3 = np.concatenate((metric3, np.array([m3])))
                            metric4 = np.concatenate((metric4, np.array([m4])))
                            metric5 = np.concatenate((metric5, np.array([m5])))


                            synth_sg_3D = synth_sg_3D / np.max(synth_sg_3D)
                            pred_sg_3D = pred_sg_3D / np.max(pred_sg_3D)

                            metrics_folder = nnUNet_cross_sections + f"{target_dataset_name}/{nnUNet_trainer}__nnUNetPlans__{nnUNet_model_configuration}_run{run:02.0f}/"
                            create_nonexisting_directory(metrics_folder)
                            folder = f"{metrics_folder}{date}-rec{recording_number}-frame{frame_number:03.0f}.png"

                            # View cross section plot
                            if save_plot:
                                view_us_gt_pred_plot(synth_us_3D, 
                                                synth_sg_3D, 
                                                pred_sg_3D, 
                                                title="Test {}".format(f"{date} {recording_number} {frame_number}F-SCORE: {m4:.3f}"),
                                                figsize=(15, 15),
                                                save_path=f"{metrics_folder}{date}-rec{recording_number}-frame{frame_number:03.0f}.png",
                                                verbose=False)

                                print(f"f score: {m4}")

                            # Save the array
                            # np.save(f"{npy_video_4D_path}{date}-rec{recording_number}-frame{frame_number:03.0f}.npy", pred_sg_3D)
                            # print(f"{npy_video_4D_path}{date}-rec{recording_number}-frame{frame_number:03.0f}.npy")
                            # sys.exit()
                                
                        pattern = r"\d{5}"
                        matches = re.findall(pattern, target_dataset_name)
                        if len(matches) >= 2:
                            num_exp = matches[-2]
                            num_syn = matches[-1]

                        # For nnUNet plot metrics old
                        # if len(str(target_dataset_id)) != 4:
                        #     save_metrics_file = nnUNet_evaluation_metrics_old + \
                        #     f"64x64x64/excluded_{exclude}/{nnUNet_trainer}__nnUNetPlans__{nnUNet_model_configuration}/run_{run:02.0f}/model{num_exp}-{num_syn}mix/"
                        # else:
                        #     save_metrics_file = nnUNet_evaluation_metrics_old + \
                        #     f"64x64x64/excluded_{exclude}_warped/{nnUNet_trainer}__nnUNetPlans__{nnUNet_model_configuration}/run_{run:02.0f}/model{num_exp}-{num_syn}mix/"
                        # if not os.path.exists(save_metrics_file):
                        #     os.makedirs(save_metrics_file)
                        # if save_metrics:
                        #     create_nonexisting_directory(save_metrics_file)
                        #     np.save(save_metrics_file + f"metric0.npy", np.mean(metric0))
                        #     np.save(save_metrics_file + f"metric1.npy", np.mean(metric1))
                        #     np.save(save_metrics_file + f"metric2.npy", np.mean(metric2))
                        #     np.save(save_metrics_file + f"metric3.npy", np.mean(metric3))
                        #     np.save(save_metrics_file + f"metric4.npy", np.mean(metric4))
                        #     np.save(save_metrics_file + f"metric5.npy", np.mean(metric5))
                        #     print("M4", metric4)

                        # For the nnUNet new metrics plot
                        if len(str(target_dataset_id)) != 6:
                            save_metrics_file = nnUNet_evaluation_metrics + \
                            f"64x64x64/run_{run:02.0f}/excluded_{exclude}/{nnUNet_trainer}__nnUNetPlans__{nnUNet_model_configuration}/model{num_exp}-{num_syn}mix/"
                        else:
                            save_metrics_file = nnUNet_evaluation_metrics + \
                            f"64x64x64/run_{run:02.0f}/excluded_{exclude}_warped/{nnUNet_trainer}__nnUNetPlans__{nnUNet_model_configuration}/model{num_exp}-{num_syn}mix/"

                        if save_metrics:
                            create_nonexisting_directory(save_metrics_file)
                            np.save(save_metrics_file + f"metric0.npy", np.mean(metric0))
                            np.save(save_metrics_file + f"metric1.npy", np.mean(metric1))
                            np.save(save_metrics_file + f"metric2.npy", np.mean(metric2))
                            np.save(save_metrics_file + f"metric3.npy", np.mean(metric3))
                            np.save(save_metrics_file + f"metric4.npy", np.mean(metric4))
                            np.save(save_metrics_file + f"metric5.npy", np.mean(metric5))
                            print("M4", metric4)

                    # else:
                    #     print(os.path.exists(prediction_dir), prediction_dir)
                    # view_training_runs_plot("/mnt/data2/joe/data/data experimental/ex-vivo-rabbit/nnunet/nnUNet_metrics/64x64x64/",
                    #                         save_path=f"/mnt/data2/joe/data/data experimental/ex-vivo-rabbit/nnunet/nnUNet_metrics_figures/run{run:02.0f}.png")
                
    print("FINISHED!")
    sys.exit()

    run = 0
    from support import view_training_runs_plot_version2
    view_training_runs_plot_version2("//mnt/data2/joe/data/data experimental/ex-vivo-rabbit/nnunet/nnUNet_metrics_new/",
                                        save_path=f"/mnt/data2/joe/data/data experimental/ex-vivo-rabbit/nnunet/nnUNet_metrics_figures/run{run:02.0f}_new.png")




# %%

if __name__ == "__main__":
    main()
else:
    print(f"Not executing main file: {__name__}")
