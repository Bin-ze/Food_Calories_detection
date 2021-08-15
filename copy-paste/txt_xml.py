# ! /usr/bin/python
# -*- coding:UTF-8 -*-
import os, sys
import glob
from PIL import Image

# VEDAI 图像存储位置
src_img_dir = "./fake_image"
# VEDAI 图像的 ground truth 的 txt 文件存放位置
src_txt_dir = "./lable.txt"
class_name={
'1':'pingguo',
'2':'xiangjiao',
'3':'fanqie',
'4':'huanggua',
'5':'xigua',
'6':'li',
'7':'juzi',
'8':'caomei',
'9':'putao',
'10':'mihoutao'}
#img_Lists = glob.glob(src_img_dir + '/*.jpg')
#
#img_basenames = []  # e.g. 100.jpg
#for item in img_Lists:
#    img_basenames.append(os.path.basename(item))
#
#img_names = []  # e.g. 100
#for item in img_basenames:
#    temp1, temp2 = os.path.splitext(item)
#    img_names.append(temp1)
with open(src_txt_dir) as f:
    lines = f.readlines()
for line in lines:

    im = Image.open(line.split(' ')[0])
    width, height = im.size

    # open the crospronding txt file
    #gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
    # gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()

    # write in xml file
    # os.mknod(src_xml_dir + '/' + img + '.xml')
    xml_file = open((os.path.splitext(line.split(' ')[0])[0] + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>Food2021</folder>\n')
    xml_file.write('    <filename>' + line.split(' ')[0].split('/')[-1] + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

    # write the region of image on xml file
    for img_each_label in line.split('\n')[0].split(' ')[1:]:
        spt = img_each_label.split(',')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
        xml_file.write('    <object>\n')
        if spt[4]=='':
            print(line.split(' ')[0].split('/')[-1])
        xml_file.write('        <name>' + str(class_name[spt[4]]) + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(spt[0]) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(spt[1]) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(spt[2]) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(spt[3]) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')

    xml_file.write('</annotation>')
