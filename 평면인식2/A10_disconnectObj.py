from collections import defaultdict
from A1_classes import *

#planeSet는 업데이트 안됨
def processGraph(planeSet):
    for planeA in planeSet:
        if len(planeA.containedObj) > 1:
            for obj in planeA.containedObj:
                planeB = Plane(planeA.label, None)
                planeB.containedObj = {obj}
                planeB.equation = planeA.equation
                polygonEdges = set()
                for connectedPlane in obj.planes: # !!!!!!! 여기서 문제 됨 !!!!!!!!!!!!!! 이거 한 edge당 꼭짓점이 하나두개가 아닌것같은데 어떡하지
                    if connectedPlane in planeA.planeEdgeDict.keys():
                        planeB.planeEdgeDict[connectedPlane] = planeA.planeEdgeDict[connectedPlane]
                        connectedPlane.planeEdgeDict[planeB] = connectedPlane.planeEdgeDict[planeA]
                        polygonEdges.add(planeA.planeEdgeDict[connectedPlane])
                        del connectedPlane.planeEdgeDict[planeA]
                holeFill_1(planeB, polygonEdges)
                obj.planes.remove(planeA)
                obj.planes.add(planeB)

#art gallery problem 방식
#planeB.interiorPoints들을 채워줌
def holeFill_1(planeB, polygonEdges):
    pass