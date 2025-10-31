from enum_class import *
import numpy as np
my_dict = {AttackRange.CLOSE: "close", AttackRange.MEDIUM: "medium", AttackRange.LONG: "long"}
nplist = np.ndarray([1,2,3], dtype=np.int8)
a = nplist[0]
print(type(a))
print(my_dict[a])