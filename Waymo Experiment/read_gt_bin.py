# from waymo_open_dataset.protos import submission_pb2

# f = open('/home/local/ASURITE/ataparia/snap/snapd-desktop-integration/current/Desktop/mmdet3d/gt.bin', mode='rb') 
# submission = submission_pb2.Submission() 
# if submission.ParseFromString(f.read()): 
#     print(submission) 
# else: 
#     print('fail')
    
from waymo_open_dataset.protos import metrics_pb2

# # Open the binary file in read mode
# with open('/home/local/ASURITE/ataparia/snap/snapd-desktop-integration/current/Desktop/mmdet3d/groundtruth.bin', mode='rb') as f:
#     # Initialize the Objects container
#     objs = metrics_pb2.Objects()
#     # Parse the binary content into the objs container
#     objs.ParseFromString(f.read())
# Open the binary file in read mode
with open('/mnt/data/ataparia/waymo_perception_dataset_v1_4_3/gt_validation_subset_1.bin', mode='rb') as f:
    # Initialize the Objects container
    objs = metrics_pb2.Objects()
    # Parse the binary content into the objs container
    objs.ParseFromString(f.read())

# Open the text file in write mode
with open('gt_validation_subset_1.txt', mode='w') as f:
    # Write the string representation of objs to the file
    f.write(str(objs)[:10000])
