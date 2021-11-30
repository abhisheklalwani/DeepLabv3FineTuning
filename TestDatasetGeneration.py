import os
import shutil

test_data_path = "D:/small_obs_dataset/Small_Obstacle_Dataset/test/stadium_3/"
target_path = "C:/Users/Abhishek/Desktop/Work/UMass Fall 2021/682/DeepLabV3FineTuningLatest/DeepLabv3FineTuning/SmallObjectDataset/Test_Images/"
count = 0
for image in os.listdir(test_data_path+'labels'):
	if count%2 == 0:
		source_image = test_data_path+'image/'+image
		destination_image = target_path+'Images'
		source_mask = test_data_path+'labels/'+image
		destination_mask = target_path+'Masks/'+image
		shutil.copy(source_image,destination_image)
		shutil.copy(source_mask,destination_mask)
	count+=1