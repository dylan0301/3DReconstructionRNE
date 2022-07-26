#boundarypoint들을 위치를 기준으로 DBSCAN함.
#같은 클러스터로 분류된 boundarypoint들을 같은 물체로 분류

#같은 물체로 분류된 boundarypoint 들에 대해서
    #각 boundarypoint와 거리 r 이내에 있는 centerpoint들에 대해서
        #그 centerpoint의 평면 클러스터를 확인하고
        #만약 지금까지 이 boundarypoint를 기준으로 측정한 centerpoint 중 그 평면 클러스터가 없었다면
            #리스트에 그 평면 클러스터를 추가한다
    #만약 리스트의 길이가 2라면 (한 boundarypoint를 기준으로 측정한 cluster가 2개라면)
        #그 2개의 평면 클러스터 쌍 연결 가능성을 1 증가시킨다.
#클러스터 쌍에 대해서 평면 클러스터 쌍 연결 가능성이 일정 수 이상이면 그 2개를 연결시킨다.
#또한, 이 boundarypoint 클러스터 안에서 찾은 각각의 평면 클러스터들을 D_i 리스트에 추가한다. 
#하나의 D_i는 하나의 boundarypoints DBSCAN 덩어리에서 찾은 평면들의 집합임

#각각의 평면 클러스터를 버텍스로 하고, 그 연결 관계를 edge로 하는 그래프를 그린다
#만약 버텍스(v)가 여러 D_i에 속한다면 v를 제거한다. 얘네가 바닥이나 벽, 책상
#v를 원소로 가지는 모든 D_i들에 대해서
    #D_i의 원소들 중 v와 adjacent했던 클러스터들에 대해서
        #구멍을 메꾸는 평면 cluster vertex u를 생성하고 u와 각각의 클러스터들을 이어 준다.
        #구멍 메꾸는법:
            #v제거할때 v와 연결된 edge들이 구멍을 이루는 boundarypoints다. 
            #한 edge에 속한 boundarypoints들은 직선 하나를 이룬다. 이거 RANSAC으로 찾음.
            #v 평면에 그 직선을 projection 시킴
            #이렇게 모두 projection시키면 v평면에 내부 구역이 생긴다. 그게 구멍임.
#이러면 이제 component들이 object가 될 예정


#문제점: 물체가 하나밖에 없이 바닥 위에 놓여있다면 제거가 안된다.
#ㄴ근데 그럴일은 없음. 물체 여러개다.

#문제점: 물체끼리 겹쳐 있는 경우 그 물체끼리 구분하기 아직 어려움. -답이 없음-
