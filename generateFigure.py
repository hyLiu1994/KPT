#coding=utf-8

import imageio
import os
import os.path

def create_gif(gif_name, image_list, duration = 0.3):
    '''
    生成gif文件，原始图片仅支持png格式
    gif_name ： 字符串，所生成的 gif 文件名，带 .gif 后缀
    image_list :      需要合成为 gif 的图片所在路径
    duration :  gif 图像时间间隔
    '''
    frames = []
    for image_name in image_list:
        # 读取 png 图像文件
        frames.append(imageio.imread(image_name))
    # 保存为 gif 
    imageio.mimsave(gif_name, frames, 'GIF', duration = duration)
    return
