import imageio.v2 as imageio
import os
import glob

def make_animation(dir_path):
    '''
    Reference:
    https://zhuanlan.zhihu.com/p/634614676
    '''
    image_list = []
    file_num = len(os.listdir(dir_path))
    
    files = glob.glob(dir_path+'/*')
    files.sort()
    for file in files:
        im = imageio.imread(file)
        image_list.append(im)
    imageio.mimsave('animation.gif',image_list, 'GIF', duration=50)
    print('Done!')

make_animation('./png')