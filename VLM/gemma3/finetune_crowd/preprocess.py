import scipy.io as sio
mat = sio.loadmat("/purestorage/AILAB/AI_4/byko/VLM/dataset/ShanghaiTech_Crowd_Counting_Dataset/part_B_final/test_data/ground_truth/GT_IMG_1.mat")
print(mat.keys())

image_info = mat["image_info"]

points = image_info[0][0][0][0][0] 

print("사람 수:", len(points))
print("좌표 예시:", points[:1])