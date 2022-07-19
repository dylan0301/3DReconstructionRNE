import os
import numpy as np
from plyfile import PlyData

#ply파일에 내장된 법선벡터 쓸때
def read_ply_xyznormalrgb(filename):
    """ read XYZ normals RGB point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 9], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['nx']
        vertices[:,4] = plydata['vertex'].data['ny']
        vertices[:,5] = plydata['vertex'].data['nz']
        vertices[:,6] = plydata['vertex'].data['red']
        vertices[:,7] = plydata['vertex'].data['green']
        vertices[:,8] = plydata['vertex'].data['blue']
    return vertices 

#ply파일에서 법선벡터 버리고 xyz rgb정보만 가져옴
def read_ply_xyzrgb(filename):
    """ read XYZ RGB point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
    return vertices 


if __name__ == "__main__":
    #좌표값이 같은 점이 있는지 찾는 함수
    def isThereSameXYZ(pointcloud):
        def distanceImportFile(p1,p2):
            def sqsbt(a,b):
                return (a-b)**2
            return ((sqsbt(p1[0], p2[0])+sqsbt(p1[1], p2[1])+sqsbt(p1[2], p2[2])))**(0.5)
        for i in range(len(pointcloud)):
            p1 = pointcloud[i][:3]
            for j in range(i+1, len(pointcloud)):
                p2 = pointcloud[j][:3]
                d = distanceImportFile(p1,p2)
                if d < 0.0001:
                    print(d, i, j)
        print('asdf')
    filepath = '/Users/jeewon/Library/CloudStorage/OneDrive-대구광역시교육청/지원/한과영/RnE/3DReconstructionRNE/pointclouddata/'
    actualfile = 'medium_33091.ply'
    #actualfile = '5000points_2plane.ply'
    #actualfile = 'million_bestsayang.ply'
    #actualfile = '25MAN.ply'
    pointcloud = read_ply_xyzrgb(filepath+actualfile)
    #print(pointcloud)
    print(len(pointcloud))
    isThereSameXYZ(pointcloud)

