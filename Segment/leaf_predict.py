"""
Created on Sun Apr 14 11:32:59 2019

@author: ywx
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
#import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

import matplotlib.patches as mpatches
from skimage import measure
  
from PIL import ImageFont
from PIL import ImageDraw

def get_delbg(img_original):
    #使用分水岭和GrabCut算法进行物体分割

    img1 = cv2.resize(img_original,(450,300),interpolation=cv2.INTER_LINEAR)
    img = img1
    mask = np.zeros(img.shape[:2],np.uint8)

    # 背景色bgdModel，前景色fgdModel
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    # 感兴趣区域ROI的x，y，宽度，高度
    rect = (30,30,400,200)

    # 获得返回值mask、bgdModel、fgdModel。
    # 目标图像、掩码、感兴趣区域，背景、前景、算法迭代次数、操作模式
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    # subplot(121)创建1行2列，当前位置为1
    # plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title("grabcut"), plt.xticks([]), plt.yticks([])
    #
    # # subplot(122)当前位置为2
    # plt.subplot(122), plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    # plt.title("original"), plt.xticks([]), plt.yticks([])
    # plt.show()

    out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #这是使用分水岭和GrabCut算法分割出缩小的图片的图
    img = Image.fromarray(out)

    # 模式L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
    Img = img.convert('L')
    #Img1 = np.array(Img)

    # 自定义灰度界限，大于这个值为黑色，小于这个值为白色
    threshold = 10
    table = []
    for i in range(256):

        if i < threshold:
            table.append(0)
        else:
            table.append(1)

    # 图片二值化
    photo = Img.point(table)
    #没什么用，就是为了查看值
    photo_array = np.array(photo)

    #保存二值图，全黑，只有01
    #photo.save(".\img\caiji2\caiji2_scaling_bw.png")

    #显示对比
    # plt.subplot(131), plt.imshow(img),plt.title('original image')
    # plt.axis("off")#去除坐标轴
    # plt.subplot(132), plt.imshow(Img,cmap='gray'),plt.title('gray image')
    # plt.axis("off")#去除坐标轴
    # plt.subplot(133), plt.imshow(photo,cmap='gray'),plt.title('bw image')
    # plt.axis("off")#去除坐标轴
    # plt.show()

    ##########################二值图像放大得到掩模####################################
    [m,n] = img_original.shape[:2]
    #线性插值放大
    resized = cv2.resize(photo_array,(n,m),interpolation=cv2.INTER_LINEAR)
    #
    # plt.imshow(resized,cmap='gray')
    # plt.show()

    ##############################掩模去除叶柄和空洞#########################################

    #定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30, 50))

    #开运算
    opened = cv2.morphologyEx(resized, cv2.MORPH_OPEN, kernel)

    #显示腐蚀后的图像
    # plt.imshow(opened,cmap='gray')

    #闭运算
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    # #显示腐蚀后的图像
    # plt.imshow(closed,cmap='gray')



    ##########################获得原图去背景########################################
    (B,G,R) = cv2.split(img_original)
    result1 = B*closed
    result2 = G*closed
    result3 = R*closed

    merged = cv2.merge([result1,result2,result3])

    #BGR倒置一下，显示为RGB
    # plt.imshow(merged[..., -1::-1],cmap='gray')

    #################################去边缘########################################

    #腐蚀图像
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20, 40))
    eroded = cv2.erode(closed,kernel1)

    #显示腐蚀后的图像
    # plt.imshow(eroded,cmap='gray')

    #膨胀图像
    dilated = cv2.dilate(closed,kernel1)

    #原二值图减去腐蚀之后的图得到了边缘
    bianyuan = dilated-eroded
    # plt.imshow(bianyuan,cmap='gray')

    #取反
    bianyuan_reverse = bianyuan
    bianyuan1 = bianyuan - 1
    bianyuan_reverse = bianyuan1//255
    return merged,bianyuan_reverse, closed, bianyuan

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

 # 预测模型路径
from os import getcwd
path = getcwd()
model_root = path+'/leaf_train_model2_cpu.pth'


def predict(img_root1):
    ## 加载网络参数
    #model为只采集了病斑和健康叶片训练出来的模型，model2为加了背景训练出来的模型
    #model = torch.load(model_root)
    model = Net(3, 10, 2)
    model.load_state_dict(torch.load(model_root))

    #读入原始采集图片
    global image 
    global img_root
    img_root = img_root1
    image = cv2.imread(img_root)

    #取get_delbg返回值的第一个,第一个返回值为去掉叶柄的掩模图像，第二个返回值为边缘
    global imgimg
    imgimg = get_delbg(image)
    img = imgimg[0]

    h, w = img.shape[:2] #打印图片的行和列，第三个img.shape[3]是通道数
    # print(h,w)

    '''
    opencv的图片是以BGR存储，因此想要正常显示，必须将BGR格式转换成RGB格式
    '''

    # #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.axis("off")#去除坐标轴
    # plt.show()


    '''注意：如果上面没有把BGR图片专成RGB，这里就是：(B,G,R) = cv2.split(img)'''

    #制作训练集
    (R,G,B) = cv2.split(img)#提取R、G、B分量
    i_r=np.ravel(R)
    i_g=np.ravel(G)
    i_b=np.ravel(B)

    c = np.vstack((i_r,i_g,i_b))
    cc = np.transpose(c)
    xx1 = torch.from_numpy(cc)
    xx1 = Variable(xx1)
    xx1 = xx1.type(torch.FloatTensor) # 转Float

    out = model(xx1)

    prediction = torch.max(out, 1)[1]
    pred_y = prediction.data.numpy()

    result = pred_y.reshape(h,w)
    #
    # plt.imshow(result,cmap='gray')
    # plt.axis("off")#去除坐标轴
    # plt.show()

    #因为opencv中255代表白色，而不是1，如果直接保存的话，全黑
    result1 = result*255

    ###########################将获得的图像去边缘###########################
    #获取imgimg的第二个返回值，边缘

    bianyuan_reverse = imgimg[1]
    final_result = result1*bianyuan_reverse
    # plt.imshow(final_result, cmap='gray')
    # plt.show()

    return final_result

# 测试用例
#img_root = 'F:\ywx\python\BP\caiji\caiji1.jpg'
#test = predict(img_root)

def join(png1, png2):
    """
    :param png1: path
    :param png2: path
    :param flag: horizontal or vertical
    :return:
    """
    img1, img2 = png1, Image.open(png2)
    size1, size2 = img1.size, img2.size

    joint = Image.new('RGB', (size1[0], size1[1]+size2[1]))
    loc1, loc2 = (0, 0), (0, size1[1])
    joint.paste(img1, loc1)
    joint.paste(img2, loc2)
    #joint.save('vertical.png')
    #pic = Image.open('vertical.png')
    return joint  
    
def word2pic(im1):

    
    #设置字体，如果没有，也可以不设置
    font = ImageFont.truetype("C:\\Users\\ywx\\Desktop\\simhei.ttf",40)
    
    #打开底版图片
    

    # 在图片上添加信息
    draw = ImageDraw.Draw(im1)
    global info1, info2, info3, info4, info5
    draw.text((120,600),"参数统计如下：",(0,0,0), font=font)
    draw.text((200,670),"病斑个数:{} 个".format(info1),(0,0,0), font=font)
    draw.text((200,740),"病斑面积:{} 平方厘米".format(info2),(0,0,0), font=font)
    draw.text((200,810),"叶片面积:{} 平方厘米".format(info3),(0,0,0), font=font)
    draw.text((200,880),"叶片周长:{} 厘米".format(info4),(0,0,0), font=font)
    draw.text((200,950),"病害等级:{} 级".format(info5),(0,0,0), font=font)
    draw = ImageDraw.Draw(im1)

    return im1


def count(bw):

    
    #bw = cv2.imread('result.png',0)
    cleared = bw.copy()  #复制
    #segmentation.clear_border(cleared)  #清除与边界相连的目标物
     
    
    #connectivity表示判定连通的模式：1表示四连通，2表示8连通
    HH =measure.label(cleared, return_num = 1,connectivity = 2)  #连通区域标记
    label_image = HH[0]
    
    #连通区域个数,这个固定不变的，得出的是图像中所有的连通区域个数
    numALL = HH[1]
    
    borders = np.logical_xor(bw, cleared) #异或
    label_image[borders] = -1
    fig,(ax0,ax1)= plt.subplots(1,2, figsize=(8, 6))
    #ax0.imshow(cleared,plt.cm.gray)
    ax0.imshow(image)
    ax0.axis('off')
    #ax1.imshow(cleared,cmap='gray')
    ax1.imshow(image)
    ax1.axis('off')
    #ts = Image.new('RGB', (200,200), (255,255,255))
    num=0 #忽略小区域之后的连通区域数
    AREA = 0 #计算病斑面积
    for region in measure.regionprops(label_image): #循环得到每一个连通区域属性集    
        #忽略小区域
        if region.area < 15:
            continue
        num+=1
        AREA = AREA + region.area
        #绘制外包矩形
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=1)
        ax1.add_patch(rect)
        
    fig.tight_layout()
    hh = 'Count:' + str(num)
    plt.title(hh)
    #plt.figure(figsize = (30, 20))
    path = os.getcwd()
    print(path)
    plt.savefig(path + '/Segment/static/count.png',dpi = 200, bbox_inches = 'tight')
    #plt.show()
    #print('hhh')
  
    im2=Image.open(path + '/Segment/static/count.png');


  
    return im2,num,AREA




#测试
def count2():
   
    global img_root
    test = predict(img_root)
    result = count(test)
    
    png1 = result[0]
    png2 = path + '/Segment/static/base.png'
    

    
    hh = join(png1, png2)


    #面积
    #dis_pix = np.sum(test/255); #病斑像素点个数
    dis_pix = result[2]; #病斑像素点个数
   
    
    
    
    height = len(image)
    width = len(image[0])
    canzhao = image[0:int(height/4),0:int(width/4)]
    gray = Image.fromarray(canzhao) #灰度化
    gray_img = gray.convert('L')
    #二值化  
    threshold = 125  
    table = []
    for ii in range(256):
        
        if ii < threshold:
            table.append(0)
        else:
            table.append(1)     
    
    # 图片二值化
    bw_canzhao = gray_img.point(table)  

    #plt.imshow(bw_canzhao,cmap='gray')
    
    canzhao_array = np.array(bw_canzhao)
    
    canzhao_pix = np.sum(canzhao_array) #参照物像素个数
    
    area = dis_pix/canzhao_pix
    
    #叶片面积
    global imgimg
    bw_leaf = imgimg[2]
    leaf_pix = np.sum(bw_leaf)
    leaf_area = round(leaf_pix/canzhao_pix, 2)
    
    
    global info1, info2, info3, info4, info5
    info1 = result[1] #病斑数
    info2 = round(area,2)          #病斑面积,保留两位小数
    info3 = leaf_area       #叶片面积
    #除以38是因为边缘大概有38圈1的像素点
    info4 = round((np.sum(imgimg[3])/ np.sqrt(canzhao_pix))/38, 2)        #叶片周长
    
    if info1 == 0:
           dengji=0;
    elif info1>0 & info1<10:
           info5 = 1;
    elif info1>=10 & info1<50:
           info5 = 3;
    elif info1>=50 & info1<120:
           info5 = 5;
    elif info1>=120 & info1<200:
           info5 = 7;
    else: 
           info5 = 9;

    info5 = info5
    hhh = word2pic(hh)
    hhh.save(path + '/Segment/static/target.png')
    return hhh