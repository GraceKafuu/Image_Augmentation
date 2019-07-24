## GraceKafuu
## 2019.7.24

import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils
import sys
import time
import matplotlib.pyplot as plt
import os
import argparse
import shutil
import xml.dom.minidom
import xml.etree.ElementTree
from PIL import Image

# d2l.set_figsize()
# img = image.imread('data/44.jpg')
# # d2l.plt.imshow(img.asnumpy())
#
# # def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
# #     Y = [aug(img) for _ in range(num_rows * num_cols)]
# #     d2l.show_images(Y, num_rows, num_cols, scale)
# #
# # apply(img, gdata.vision.transforms.RandomFlipLeftRight())
# # plt.show()
#
# fimg = gdata.vision.transforms.RandomFlipLeftRight()(img)
# # print(fimg)
# # print(type(img))
#
# d2l.plt.imshow(fimg.asnumpy())
# d2l.plt.imsave("data/qswqsqws.jpg",fimg.asnumpy())
# plt.show()
#

parser = argparse.ArgumentParser()
parser.add_argument("--img_path")
parser.add_argument("--xml_path")
#parser.add_argument("--aug_img_path")
#parser.add_argument("--aug_xml_path")

def get_img_list(path):
    img_list = []
    files = os.listdir(path)
    for f in files:
        if os.path.splitext(f)[1] == ".jpg" or os.path.splitext(f)[1] == ".png":
            img_list.append(os.path.join(path,f))
    return sorted(img_list)

def get_xml_list(path):
    xml_list = []
    files = os.listdir(path)
    for f in files:
        if os.path.splitext(f)[1] == ".xml":
            xml_list.append(os.path.join(path,f))
    return sorted(xml_list)

def main(args):
    img_path = args.img_path
    xml_path = args.xml_path
    #aug_img_path = args.aug_img_path
    #aug_xml_path = args.aug_xml_path


    if not os.path.exists(os.path.join(os.path.abspath(os.path.join(img_path,'..')),'MXNet_Aug_JPEGImages-mid')):
        os.mkdir(os.path.join(os.path.abspath(os.path.join(img_path,'..')),'MXNet_Aug_JPEGImages-mid'))
    if not os.path.exists(os.path.join(os.path.abspath(os.path.join(xml_path,'..')),'MXNet_Aug_Annotations-mid')):
        os.mkdir(os.path.join(os.path.abspath(os.path.join(xml_path,'..')),'MXNet_Aug_Annotations-mid'))
    

    img_list = get_img_list(img_path)
    start_num = int(img_list[0].split('/')[-1].split('.')[0])    

    for n,i in enumerate(img_list):
        #flip_left_right_img = gdata.vision.transforms.RandomFlipLeftRight()(image.imread(i))
        #d2l.plt.imsave("image_aug/{}_flip_left_right_img.jpg".format(n), flip_left_right_img.asnumpy())

        #flip_top_bottom_img = gdata.vision.transforms.RandomFlipTopBottom()(image.imread(i))
        #d2l.plt.imsave("image_aug/{}_flip_top_bottom_img.jpg".format(n), flip_top_bottom_img.asnumpy())

        brightness_img = gdata.vision.transforms.RandomBrightness(1)(image.imread(i))
        d2l.plt.imsave(os.path.join(os.path.abspath(os.path.join(img_path,'..')),'MXNet_Aug_JPEGImages-mid/{}_brightness_img.jpg'.format(n+start_num)), brightness_img.asnumpy())

        hue_img = gdata.vision.transforms.RandomHue(1)(image.imread(i))
        d2l.plt.imsave(os.path.join(os.path.abspath(os.path.join(img_path,'..')),'MXNet_Aug_JPEGImages-mid/{}_hue_img.jpg'.format(n+start_num)), hue_img.asnumpy())

        jitter_img = gdata.vision.transforms.RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(image.imread(i))
        d2l.plt.imsave(os.path.join(os.path.abspath(os.path.join(img_path,'..')),'MXNet_Aug_JPEGImages-mid/{}_jitter_img.jpg'.format(n+start_num)), jitter_img.asnumpy())

        saturation_img = gdata.vision.transforms.RandomSaturation(1)(image.imread(i))
        d2l.plt.imsave(os.path.join(os.path.abspath(os.path.join(img_path,'..')),'MXNet_Aug_JPEGImages-mid/{}_saturation_img.jpg'.format(n+start_num)), saturation_img.asnumpy())
        
        #shape_aug = gdata.vision.transforms.RandomResizedCrop((512, 512), scale=(0.1, 1), ratio=(0.5, 2))(i)
        #d2l.plt.imsave("image_aug/{}_shape_aug_img.jpg".format(n), shape_aug.asnumpy())

    ################处理xml######################
    xml_files = get_xml_list(xml_path)
    for n,xml in enumerate(xml_files):
        shutil.copyfile(xml, os.path.join(os.path.abspath(os.path.join(img_path,'..')),'MXNet_Aug_Annotations-mid/{}_brightness_img.xml'.format(n+start_num)))
        shutil.copyfile(xml, os.path.join(os.path.abspath(os.path.join(img_path,'..')),'MXNet_Aug_Annotations-mid/{}_hue_img.xml'.format(n+start_num)))
        shutil.copyfile(xml, os.path.join(os.path.abspath(os.path.join(img_path,'..')),'MXNet_Aug_Annotations-mid/{}_jitter_img.xml'.format(n+start_num)))
        shutil.copyfile(xml, os.path.join(os.path.abspath(os.path.join(img_path,'..')),'MXNet_Aug_Annotations-mid/{}_saturation_img.xml'.format(n+start_num)))

##修改filename,例如<filename>000001.jpg</filename> --> <filename>1_brightness_img.jpg</filename>
def aug_xml_to_final(args):
    img_path = args.img_path
    #xml_path = args.xml_path

    xml_path = os.path.join(os.path.abspath(os.path.join(img_path,'..')),'MXNet_Aug_Annotations-mid')
    aug_xml_path = os.path.join(os.path.abspath(os.path.join(img_path,'..')),'MXNet_Aug_Annotations')

    if not os.path.exists(aug_xml_path):
        os.mkdir(aug_xml_path)

    for xmlfile in sorted(os.listdir(xml_path)):
        xmlname = os.path.splitext(xmlfile)[0]
        # 读取 xml 文件
        dom = xml.dom.minidom.parse(os.path.join(xml_path, xmlfile))
        # print(os.path.join(xmldir,xmlfile))
        root = dom.documentElement
        # print("ddddddddddd:",len(root.getElementsByTagName('name')))
        # 获取标签对的名字，并为其赋一个新值
        root.getElementsByTagName('filename')[0].firstChild.data = '{}.jpg'.format(xmlname)
        #for i in range(len(root.getElementsByTagName('name'))):
        #    root.getElementsByTagName('name')[i].firstChild.data = 'defect'

        # image_aug_xml_final = os.path.join(os.path.abspath(os.path.join(img_path,'..')),'image_aug_xml_final')
        # 修改并保存文件
        #xml_specific = image_aug_xml_final + xmlfile
        with open(os.path.join(aug_xml_path,xmlfile), 'w') as fh:
            dom.writexml(fh)

    shutil.rmtree(xml_path)

    #MXNet_Aug_JPEGImages-mid_path = os.path.join(os.path.abspath(os.path.join(img_path,'..')),'MXNet_Aug_JPEGImages-mid')

    if not os.path.exists(os.path.join(os.path.abspath(os.path.join(img_path,'..')),'MXNet_Aug_JPEGImages')):
        os.mkdir(os.path.join(os.path.abspath(os.path.join(img_path,'..')),'MXNet_Aug_JPEGImages'))

    for img in get_img_list(os.path.join(os.path.abspath(os.path.join(img_path,'..')),'MXNet_Aug_JPEGImages-mid')):
        im = Image.open(img)
        rgb_im = im.convert('RGB')
        file_name = img.split('/')[-1].split('.')[0]+'.jpg'
        rgb_im.save(os.path.join(os.path.abspath(os.path.join(img_path,'..')),'MXNet_Aug_JPEGImages',file_name))

    shutil.rmtree(os.path.join(os.path.abspath(os.path.join(img_path,'..')),'MXNet_Aug_JPEGImages-mid'))



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    aug_xml_to_final(args)

