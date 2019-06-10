from django.shortcuts import render
from Segment.models import forecastol_img
from django.http import HttpResponse
import MySQLdb
import collections
import json
from PIL import Image
from Segment.leaf_predict import predict, count, count2
import os
import base64
import cv2


def get_data(sql):
    conn = MySQLdb.connect(
        host='localhost',
        port=3306,
        user='root',
        passwd='123456',
        db='lybb',
        charset='utf8',
    )
    cur = conn.cursor()
    cur.execute(sql)
    results = cur.fetchall()
    cur.close()
    conn.commit()
    conn.close()
    return results


def forecastol_update_leaf(request):
    if request.method == 'POST':
        imgbase64 = request.POST.get('imgbase64', '')
        #print(imgbase64)
        if "data:image/png;base64,"in imgbase64:
            imgbase64 = imgbase64.replace("data:image/png;base64,", "")
        elif "data:image/jpeg;base64," in imgbase64:
            imgbase64 = imgbase64.replace("data:image/jpeg;base64,", "")
        #print(imgbase64)
        imgdata = base64.b64decode(imgbase64)
        path = os.getcwd()
        newpath = path+"/forecast_img/"
        sql = "select max(id) " + "from Segment_forecastol_img"
        data = get_data(sql)
        #print(data)
        pic_num = int(data[0][0])+1
        pic_root = newpath+'img_'+str(pic_num)+'.png'
        picname = 'img_'+str(pic_num)+'.png'
        with open(pic_root, "wb+")as f:
            f.write(imgdata)
        #print(picname)
        info = forecastol_img()
        info.id = pic_num
        info.imgroot = pic_root
        info.save()

        oblist = []
        oblist.append(picname)
        data = json.dumps(oblist)

        result = predict(pic_root)
        result2 = count(result)
        result3 = count2()


        #print(result)
        #print(path)
        # 保存病斑分割出来的二值图像
        #cv2.imwrite(path + '/Segment/static/resultimg/' + "img_" + str(pic_num) + ".jpg",  result)
        result3.save(path + '/Segment/static/resultimg/' + "img_" + str(pic_num) + ".png")

        #print("ssss:"+data)
        return HttpResponse(data)


def mobile_forecast_leaf(request):
    return render(request, "mobileforecastleaf.html")





