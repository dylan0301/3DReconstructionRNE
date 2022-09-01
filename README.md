# 3DReconstructionRNE

평면인식2/main.py 실행 파일에



```python

23 filepath = '/Users/jeewon/Library/CloudStorage/OneDrive-대구광역시교육청/지원/한과영/RnE/3DReconstructionRNE/pointclouddata/'
24 filename = '3boxes.ply'
25 
26 AllPoints, hyperparameter, name = importPly(filepath, filename)
27 #AllPoints, hyperparameter, name = OpenPlane()

```

코드가 있다.


ply파일로부터의 pointcloud data를 사용하고 싶을 때: 
```
filepath를 ply파일이 들어있는 폴더의 절대 경로로 설정한다. 이때, pointcloud 폴더에 여러 ply파일들이 있다.
filename을 해당 ply파일의 이름으로 설정한다.
그리고 26번째 줄을 작동시킨다.
```

가상으로 만든 데이터를 사용하고 싶을 때:
```
26번 줄에 주석을 달고 27번 줄의 주석을 제거한다.
평면인식2/A2_data.py 에 가면 가상으로 제작해놓은 데이터들이 있다.
현재 거기서 사용할 수 있는 데이터의 목록은 다음과 같다.

bang_muchsimple()
bang_verysimple()
cube_sameDensity()
VertexDense()
NonUniformCube()
FourCleanBoxes()
OpenPlane()
FloorWall()


```
