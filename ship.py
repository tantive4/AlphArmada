Victory_2_dict = {
    'hull' : 8,
    'size' : 'medium', # 61 x 102
    'command' : 3,
    'squadron' : 3,
    'engineering' : 4,
    'defense_token' : ['brace', 'redirect', 'redirect'],
    'anti_squad' : [0, 1, 0],
    'shield' : [3, 3, 1],
    'battery' : [[0, 3, 3], [0, 1, 2], [0, 0, 2]],
    'navchart' : {1 : [1], 2 : [0, 1]},
    'point' : 80,
    'front_arc_center' : 48, # 전방 발포 호 선과 중앙선이 만나는 점과 함선 맨 앞 사이 거리(mm)
    'front_arc_end' : 24, # 전방 발포 호 선이 측면 끝과 만나는 점과 함선 맨 앞 앞 사이 거리 (이 둘만 있으면 발포 호 선 구현 가능)
    'rear_arc_center' : 48,
    'rear_arc_end' : 79
    }

CR90A_dict = {
    'hull' : 4,
    'size' : 'small', # 43 x 71
    'command' : 1,
    'squadron' : 1,
    'engineering' : 2,
    'defense_token' : ['evade', 'evade', 'redirect'],
    'anti_squad' : [0, 1, 0],
    'shield' : [2, 2, 1],
    'battery' : [[0, 1, 2], [0, 1, 1], [0, 0, 1]],
    'navchart' : {1 : [2], 2 : [1, 2], 3 : [0, 1, 2], 4 : [0, 1, 1, 2]},
    'point' : 44,
    'front_arc_center' : 43,
    'front_arc_end' : 23,
    'rear_arc_center' : 43,
    'rear_arc_end' : 62
    }

Neb_escort_dict = {
    'hull' : 5,
    'size' : 'small',
    'command' : 2,
    'squadron' : 2,
    'engineering' : 3,
    'defense_token' : ['evade', 'brace', 'brace'],
    'anti_squad' : [0, 2, 0],
    'shield' : [3, 1, 2],
    'battery' : [[0, 0, 3], [0, 1, 1], [0, 0, 2]],
    'navchart' : {1 : [1], 2 : [1, 1], 3 : [0, 1, 2]},
    'point' : 47,
    'front_arc_center' : 35,
    'front_arc_end' : 0,
    'rear_arc_center' : 35,
    'rear_arc_end' : 0
    }