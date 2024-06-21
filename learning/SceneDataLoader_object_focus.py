import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SceneDataset_object_focus(Dataset):
    def __init__(self, path = None):
        if not path:
            sys.exit(1)
        else:
            self.path_ = path
            self.names_ = set()
            print (path)
            for root, dirs, files in os.walk(path):
                for f in files:
                    if 'scene' in f:
                        prefix = f[:f.index('scene')-1]
                        self.names_.add(prefix)
                    elif 'camera' in files:
                        prefix = f[:f.index('camera')-1]
                        self.names_.add(prefix)
            self.names_ = list(self.names_)
            self.min_pose_ = sys.maxsize
            
    def __getitem__(self, idx):
        prefix = self.path_ + self.names_[idx]
        prefix_scene = prefix + '_scene.npy'
        prefix_camera = prefix + '_camera.npy'
        scene_graph = np.load(prefix_scene)
        #detect min_z
        min_z, max_z = None, None
        dx, dy, dz, _ = scene_graph.shape
        
        for j in range(dy):
          for k in range(dz):
            if scene_graph[0][j][k] != -1:
                if min_z == None:
                    min_z = k
                max_z = k
          if min_z != None: break
         
        #detect min_y, max_y
        min_y, max_y = None, None
        for j in range(dy):
          if scene_graph[0][j][min_z] != -1:
              if min_y == None:
                  min_y = j
              max_y = j
        
        #detect max_x
        mid_y = (min_y + max_y) // 2
        min_x, max_x = None, None
        for i in range(dx):
            if scene_graph[i][mid_y][min_z] != -1:
                if min_x == None:
                    min_x = i
                max_x = i

        hold = max_z
        max_z = min_z + 28

        score_count = 0

        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                for k in range(min_z, max_z + 1):
                    if scene_graph[i][j][k] != 0:
                        score_count += 1

        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                for k in range(max_z + 1, hold):
                    scene_graph[i][j][k] = 1

        new_score = score_count * 1.0 / ((max_x - min_x + 1) * (max_y - min_y + 1) * (max_z - min_z + 1))


        scene_graph = np.where(scene_graph == -1, 255, scene_graph)
        #print (scene_graph[1][2][3][0])
        scene_graph = np.transpose(scene_graph, (3, 0, 1, 2))
        #print (scene_graph[0][1][2][3])
        camera_info = np.load(prefix_camera)
        camera_pose = camera_info[:7]
        score = np.array([camera_info[7]])
        self.min_pose_ = min(self.min_pose_, abs(camera_pose[3]))
        return scene_graph, camera_pose, score


    def __len__(self):
        return len(self.names_)

if __name__ == '__main__':
    train_dataset = SceneDataset('./scene_train/')
    dataloader = DataLoader(train_dataset, batch_size = 4, shuffle = True)

    print (train_dataset.min_pose_)
    data_iter = iter(dataloader)
    for i, (scene, camera_pose, score) in enumerate(dataloader):
        pass
    print (train_dataset.min_pose_)

