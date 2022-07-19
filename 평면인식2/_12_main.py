import _2_data

filepath = '/Users/jeewon/Library/CloudStorage/OneDrive-대구광역시교육청/지원/한과영/RnE/3DReconstructionRNE/pointclouddata/'
filename = '50000points_2plane.ply'

AllPoints, hyperparameter = _2_data.importPly(filepath+filename)
#실제 데이터했으면 여기서 수동으로 hyperparameter 약간 수정 필요