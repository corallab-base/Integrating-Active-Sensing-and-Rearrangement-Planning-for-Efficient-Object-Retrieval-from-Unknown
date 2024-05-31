from scipy.spatial.transform import Rotation as R
import numpy as np


r = R.from_euler('xyz', [180, 90, 90], degrees=True)
vector = np.array([1,2,3])
print (r.apply(vector))
