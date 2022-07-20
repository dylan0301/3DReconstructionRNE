from _2_data import *

filepath = '/Users/jeewon/Library/CloudStorage/OneDrive-대구광역시교육청/지원/한과영/RnE/3DReconstructionRNE/pointclouddata/'
filename = '50000points_2plane.ply'

AllPoints, hyperparameter = importPly(filepath+filename)
#실제 데이터했으면 여기서 수동으로 hyperparameter 약간 수정 필요




newLabel += [numOfCluster] * len(BoundaryPoints)
plotAll = CenterPoints + BoundaryPoints

# print("클러스터 분류 결과:", newLabel)
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(Allvectors[:,0],Allvectors[:,1], Allvectors[:,2], c=ac.labels_, marker='o', s=15, cmap='rainbow')
# ap = np.array([[p.x, p.y, p.z] for p in plotAll])
# ax.scatter(ap[:, 0], ap[:, 1], ap[:, 2], c=newLabel, marker='o', s=15, cmap='rainbow')
# plt.show()