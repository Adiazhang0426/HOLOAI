"""
@Description: 
@Author: Adiazhang
@Date: 2024-05-29 09:57:35
@LastEditTime: 2024-05-30 10:27:02
@LastEditors: Adiazhang
"""

import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from skimage import measure

from ultralytics.models.yolov10.train import YOLOv10DetectionTrainer
from ultralytics.models.yolov10.predict import YOLOv10DetectionPredictor
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.predict import DetectionPredictor


"""
@Description: generate the list of position ids
@param {*} W_multiple cut times in the dimension of W
@param {*} H_multiple cut times in the dimension of H
@return {*} a list of position ids
"""


def generate_positionlist(W_multiple, H_multiple):

    positionlist = []
    for i in range(W_multiple):
        for j in range(H_multiple):
            positionlist.append("{}_{}".format(i, j))
    return positionlist


"""
@Description: generate the list of methods
@param {*} detectionmethods list of detection methods
@param {*} segmentationmethods list of segmentation methods
@return {*} a list of methods
"""


def generate_methodlist(detectionmethods, segmentationmethods):
    methodlist = []
    for det in detectionmethods:
        for segmethod in segmentationmethods:
            methodlist.append(det + "_gradvar_" + segmethod)
    return methodlist


def normalize(a):
    a = np.array(a)
    return (a - np.min(a)) / (np.max(a) - np.min(a))


"""
@Description: Crop the image 
@param {*} imgpath path for single image
@param {*} savepath root for save cropping image
@param {*} W_multiple cut times in the dimension of W
@param {*} H_multiple cut times in the dimension of H
@return {*}
"""


def cutimg(imgpath, savepath, W_multiple=4, H_multiple=4):

    img = cv2.imread(imgpath, 0)
    name = imgpath.split("\\")[-1][:-4]
    H, W = img.shape[0], img.shape[1]
    sw, sh = W / W_multiple, H / H_multiple
    for i in range(H_multiple):
        for j in range(W_multiple):
            cutimg = img[
                int(i * sh) : int((i + 1) * sh), int(j * sw) : int((j + 1) * sw)
            ]
            cv2.imwrite(
                os.path.join(
                    savepath, "{}_{}".format(i, j), name + "{}_{}.jpg".format(i, j)
                ),
                cutimg,
            )


"""
@Description: extract the boxes in the format (x1,y1,x2,y2) as well as the corresponding image ids
@param {*} txtpath root for txt files of detection boxes
@param {*} imgpath root for cropped image
@param {*} patch position id of cropped image in the format of i_j (i: H dimension; j: W dimension )
@return {*}  detected boxes and the corresponding image ids for certain patch
"""


def read_txt(txtpath, imgpath, patch):

    imglist = os.listdir(os.path.join(imgpath, patch))
    txtlist = os.listdir(txtpath)

    allbox = []
    boximgindex = []
    for i in txtlist:
        with open(os.path.join(txtpath, i), "r") as f:
            for line in f.readlines():
                [xc, yc, w, h] = [float(j) for j in line.strip().split(" ")[1:]]
                x1, y1, x2, y2 = xc - 0.5 * w, yc - 0.5 * h, xc + 0.5 * w, yc + 0.5 * h
                allbox.append([x1, y1, x2, y2])
                boximgindex.append(imglist.index(i[:-4] + ".jpg"))

    return allbox, boximgindex


"""
@Description: calculate the iou of two sets of boxes which can be either same or not
@param {*} boxes0
@param {*} boxes1
@return {*} iou matrix for all mutual boxes in these two box sets
"""


def iou(boxes0, boxes1):

    A = boxes0.shape[0]
    B = boxes1.shape[0]

    xy_max = np.minimum(
        boxes0[:, np.newaxis, 2:].repeat(B, axis=1),
        np.broadcast_to(boxes1[:, 2:], (A, B, 2)),
    )
    xy_min = np.maximum(
        boxes0[:, np.newaxis, :2].repeat(B, axis=1),
        np.broadcast_to(boxes1[:, :2], (A, B, 2)),
    )

    inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)
    inter = inter[:, :, 0] * inter[:, :, 1]

    area_0 = ((boxes0[:, 2] - boxes0[:, 0]) * (boxes0[:, 3] - boxes0[:, 1]))[
        :, np.newaxis
    ].repeat(B, axis=1)
    area_1 = ((boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]))[
        np.newaxis, :
    ].repeat(A, axis=0)

    return inter / (area_0 + area_1 - inter)


"""
@Description: merge the boxes that belong to the same particle
@param {*} ioumatrix iou_matrix from func iou()
@param {*} boxes all detected boxes from func read_txt()
@param {*} boximgindex corresponding image ids from func read_txt()
@param {*} iouthre predetermined threshold to determined whether these boxes are belong to the same particle
@param {*} W width for cropped image
@param {*} H height for cropped image
@return {*} boxes and corresponding image ids after merging, one box => more than one ids (denotes the candidate in-focus position)
"""


def merge_box(ioumatrix, boxes, boximgindex, iouthre=0.4, W=360, H=270):

    ioumatrix = np.triu(ioumatrix)
    ioumatrix[ioumatrix > iouthre] = 1
    ioumatrix[ioumatrix < iouthre] = 0

    allindex = [i for i in range(ioumatrix.shape[0])]
    ioulist = ioumatrix.tolist()
    totalmerge = []
    totalmergeindex = []

    while len(allindex) > 0:
        singlelist = ioulist[allindex[0]]
        singlemerge = []
        singlemergeindex = []
        singlemergeindex.append(boximgindex[allindex[0]])
        singlemerge.append(boxes[allindex[0]])
        allindex.remove(allindex[0])
        for j in range(len(singlelist)):
            if singlelist[j] == 1 and j in allindex:
                singlemerge.append(boxes[j])
                singlemergeindex.append(boximgindex[j])
                allindex.remove(j)
        singlemerge = np.array(singlemerge)
        mx1, my1, mx2, my2 = (
            np.average(singlemerge[:, 0]),
            np.average(singlemerge[:, 1]),
            np.average(singlemerge[:, 2]),
            np.average(singlemerge[:, 3]),
        )
        totalmerge.append([int(mx1 * W), int(my1 * H), int(mx2 * W), int(my2 * H)])
        totalmergeindex.append(singlemergeindex)

    return totalmerge, totalmergeindex


"""
@Description: determine the z position of the particle by gradient variance of grayscale
@param {*} rawimgpath image root for cropped image
@param {*} box single box after merging
@param {*} index image ids belong to the box
@param {*} patch position id of cropped image
@return {*} if the index only contains one id, then return the z position and the in-focus ROI
            else return the z position, the in-focus ROI, candidate image list and the index of in-focus ROI in candidate image list
"""


def z_location_gradient_var(rawimgpath, box, index, patch):

    x1, y1, x2, y2 = box
    imglist = os.listdir(os.path.join(rawimgpath, patch))
    gv = []

    if len(index) != 1:
        newrange = [
            i
            for i in range(max(0, min(index) - 10), min(len(imglist), max(index) + 10))
        ]
        targetimglist = [
            cv2.imread(os.path.join(rawimgpath, patch, imglist[i]), 0)[y1:y2, x1:x2]
            for i in newrange
        ]
        for i in targetimglist:
            sobelx = cv2.Sobel(i, cv2.CV_64F, 1, 0)
            sobely = cv2.Sobel(i, cv2.CV_64F, 0, 1)
            sobelx = cv2.convertScaleAbs(sobelx)
            sobely = cv2.convertScaleAbs(sobely)
            sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
            gv.append(np.var(sobelxy))
        zid = gv.index(max(gv))

        return newrange[zid], targetimglist[zid], targetimglist, zid

    else:
        img = cv2.imread(os.path.join(rawimgpath, patch, imglist[index[0]]), 0)

        return index[0], img, -1, -1


"""
@Description: 
@param {*} seg binary mask of particle ROI
@return {*} x,y (based on the ROI) and diameter of the in-focus particle
"""


def connectarea_analysis(seg):

    labels = measure.label(seg, connectivity=2)
    properties = measure.regionprops(labels)
    x = []
    y = []
    d = []
    for prop in properties:
        y.append(prop.centroid[0])
        x.append(prop.centroid[1])
        d.append(prop.equivalent_diameter)
    # delete the wrong particle
    while max(d) > 25 and len(d) > 1:
        d.remove(max(d))
    maxdid = d.index(max(d))
    return x[maxdid], y[maxdid], d[maxdid]


"""
@Description: give the correct class id to the original mask
@param {*} img in-focus particle ROI
@param {*} seg mask
@param {*} cluster
@return {*} mask with correct class id
"""


def preprocess_segmask(img, seg, cluster):

    judge = []
    for i in range(cluster):
        test = np.zeros_like(seg)
        test[seg == i] = 1
        judge.append(np.average(test * img))
    trueid = judge.index(min(judge))
    seg[seg == trueid] = 255
    seg[seg != 255] = 0
    return seg


"""
@Description: 
@param {*} resizedimg in-focus particle ROI
@param {*} cluster
@return {*} x,y,d of particle and the binary mask
"""


def segment2(resizedimg, cluster=2):

    flatenimg = resizedimg.reshape(-1, 1)
    kmeans = KMeans(n_clusters=cluster)
    kmeans.fit(flatenimg)
    seg = kmeans.labels_.reshape(resizedimg.shape[0], resizedimg.shape[1])
    seg = preprocess_segmask(resizedimg, seg, cluster)
    x, y, d = connectarea_analysis(seg)
    return x, y, d, seg


"""
@Description: iterate the threshold of grayscale and stop when the added pixels just occupy the max area of boundary
@param {*} img in-focus particle ROI
@param {*} boundary_thre hyperparameter to determine the edge pixels
@return {*} binary mask
"""


def max_boundary_gradient(img, boundary_thre=0.7):
    raw = img.copy()
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    sobelxy = normalize(sobelxy)
    thre1 = np.max(sobelxy) * boundary_thre
    sobelxy[sobelxy > thre1] = 1
    sobelxy[sobelxy <= thre1] = 0
    contrast = []
    for i in range(np.min(img), np.max(img)):
        img1 = img.copy()
        img2 = img.copy()
        img3 = np.zeros_like(img)
        img4 = np.zeros_like(img)
        img1[img1 < i] = 0
        img1[img1 >= i] = 1
        img2[img2 < i + 1] = 0
        img2[img2 >= i + 1] = 1
        img3[img1 != img2] = 1
        img4[sobelxy + img3 == 2] = 1
        contrast.append(np.sum(img4))
    thre = np.min(img) + contrast.index(max(contrast))
    raw[raw >= thre] = 255
    raw[raw < thre] = 0
    return 255 - raw


def segment_max_boundary(resizedimg):

    seg = max_boundary_gradient(resizedimg)
    x, y, d = connectarea_analysis(seg)
    return x, y, d, seg


def cluster_segment_2vector(img, fuse_vec, cluster=2):

    flatenimg = fuse_vec.reshape(-1, 2)
    kmeans = KMeans(n_clusters=cluster)
    kmeans.fit(flatenimg)
    seg = kmeans.labels_.reshape(fuse_vec.shape[0], fuse_vec.shape[1])
    seg = preprocess_segmask(img, seg, cluster)
    return seg


"""
@Description: 
@param {*} imglist candidate particle ROI list
@param {*} zid index of in-focus plane
@param {*} focusimg in-focus particle ROI
@param {*} inter number of slices are considered to calculate the variance on each side
@return {*} x,y,d of particle and the binary mask
"""


def fuse_slice_var_segment(imglist, zid, focusimg, inter=20):

    slicelist = imglist[max(0, zid - inter) : min(len(imglist), zid + inter)]
    slicelist = np.array(slicelist)
    varmatrix = np.var(slicelist, axis=0)
    varmatrix = normalize(varmatrix)
    focusimg = normalize(focusimg)
    vector = np.concatenate(
        [np.expand_dims(focusimg, axis=-1), np.expand_dims(varmatrix, axis=-1)], axis=-1
    )
    cluster_vec_seg = cluster_segment_2vector(focusimg, vector, cluster=2)
    x, y, d = connectarea_analysis(cluster_vec_seg)
    return x, y, d, cluster_vec_seg


"""
@Description: delect the repeat particle through the distance of xy or z
@param {*} partixyz already obtained particle information
@param {*} newxyz new particle information waiting for check
@return {*} flag to decide whether to append the new particle
"""


def delete_repeat(partixyz, newxyz):

    np_partixyz = np.array(partixyz)
    np_newxyz = np.array(newxyz)
    contrastxyz = np_partixyz - np_newxyz
    contrastxy = np_partixyz[:, :-1] - np_newxyz[:-1]
    distancexyz = np.min(np.linalg.norm(contrastxyz, axis=-1))
    distancexy = np.min(np.linalg.norm(contrastxy, axis=-1))
    if distancexy == 0:
        return 0
    else:
        if distancexyz < 3.5:
            return 0
        else:
            return 1


def saveresult(resultpath, patch, par):

    with open(os.path.join(resultpath, patch + ".txt"), "w") as f:
        for i in par:
            x, y, z, d, bx1, by1, bx2, by2 = (
                int(i[0]),
                int(i[1]),
                int(i[2]),
                float(i[3]),
                int(i[4]),
                int(i[5]),
                int(i[6]),
                float(i[7]),
            )
            f.write(
                "{} {} {} {} {} {} {} {}".format(x, y, z, d, bx1, by1, bx2, by2) + "\n"
            )


def location_segmentation_main(
    saveimgpath,
    savetxtpath,
    patch,
    resultpath,
    methodname,
    seginter,
    mergeiou,
    cropped_imgshape,
):
    """

    @param saveimgpath: root for cropped image
    @param savetxtpath: root for detection results
    @param patch: position id
    @param resultpath: root for saving result
    @param methodname: methodname in the format of detectionmethod_gradvar_segmentationmethod
    @param seginter: the number of images for calculating the grayscale variance
    @param mergeiou: threshold to determine the boxes belong to same particle
    @param cropped_imgshape:
    @return:
    """
    segnname = methodname.split("_")[2]
    rawboxes, rawboximgindex = read_txt(savetxtpath, saveimgpath, patch)
    np_rawboxes = np.array(rawboxes)
    ioumatrix = iou(np_rawboxes, np_rawboxes)
    mergeboxes, mergeindexs = merge_box(
        ioumatrix,
        rawboxes,
        rawboximgindex,
        mergeiou,
        cropped_imgshape[1],
        cropped_imgshape[0],
    )
    particleinfor = []
    partixy = []
    count = 0
    for mergebox, mergeindex in zip(mergeboxes, mergeindexs):
        bx, by, bx1, by1 = mergebox
        z, resizedimg, targetimglist, zid = z_location_gradient_var(
            saveimgpath, mergebox, mergeindex, patch
        )
        if z != -1 and resizedimg.shape[0] < 100:
            if segnname == "2graycluster":
                x, y, d, mask = segment2(resizedimg, 2)
            elif segnname == "maxbound":
                x, y, d, mask = segment_max_boundary(resizedimg)
            else:
                x, y, d, mask = fuse_slice_var_segment(
                    targetimglist, zid, resizedimg, seginter
                )

            singleinfor = [x + bx, y + by, z, d, bx, by, bx1, by1]
            if partixy == []:
                partixy.append([int(x + bx), int(y + by), z])
                particleinfor.append(singleinfor)
                cv2.imwrite(
                    os.path.join(resultpath, "raw_{:0>4d}.png".format(count)),
                    resizedimg,
                )
                cv2.imwrite(
                    os.path.join(resultpath, "mask_{:0>4d}.png".format(count)), mask
                )
                count += 1
            if delete_repeat(partixy, [int(x + bx), int(y + by), z]) == 1:
                partixy.append([int(x + bx), int(y + by), z])
                particleinfor.append(singleinfor)
                cv2.imwrite(
                    os.path.join(resultpath, "raw_{:0>4d}.png".format(count)),
                    resizedimg,
                )
                cv2.imwrite(
                    os.path.join(resultpath, "mask_{:0>4d}.png".format(count)), mask
                )
                count += 1
    saveresult(resultpath, patch, particleinfor)
    print(patch + "finished")


"""
@Description: train YOLOvs model beside YOLOv7 
@param {*} model model configuration in ultralytics/cfg/models
@param {*} data yaml file in ultralytics/cfg/dataset
@param {*} epoch
@param {*} batch
@param {*} device
@return {*}
"""


def train_yolo(model, data, epoch, batch, device="0"):

    if "yolov10" in model:
        args = dict(model=model, data=data, epochs=epoch, device=device, batch=batch)
        trainer = YOLOv10DetectionTrainer(overrides=args)
        trainer.train()
    # elif 'yolov7' in model:
    #     args = dict(model=model, data=data, epochs=epoch, device=device,batch=batch)
    #     trainer = DetectionTrainer(overrides=args)
    #     trainer.train()
    else:
        args = dict(model=model, data=data, epochs=epoch, device=device, batch=batch)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()


def yolo_predict(
    weight,
    source,
    save_root,
    save_txt=True,
    device="0",
    conf=0.05,
    ifv10=False,
    save_img=False,
):
    """
    @param weight: weight path of detection network default in runs/detect/trainxx/weights/best.pt
    @param source: image(file) root for predicting
    @param save_txt: whether to save boxes as .txt
    @param device: cuda or cpu
    @param conf: output boxes with confidence higher than conf
    @param ifv10: whether to use yolov10
    @param save_root: root to save labels
    @param save_img: whether to save images drew with boxes
    @return:
    """
    if ifv10:
        args = dict(
            model=weight,
            source=source,
            device=device,
            save_txt=save_txt,
            conf=conf,
            project=save_root,
            save=save_img,
        )
        Predictor = YOLOv10DetectionPredictor(overrides=args)
        Predictor.predict_cli()
    else:
        args = dict(
            model=weight,
            source=source,
            device=device,
            save_txt=save_txt,
            conf=conf,
            project=save_root,
            save=save_img,
        )
        Predictor = DetectionPredictor(overrides=args)
        Predictor.predict_cli()


def extract_fuse_info(root, positionlist, saveroot, method, H=270, W=360, pixel=3.45):
    """
    merge information from different position ids
    @param root: root for segmentation results
    @param positionlist: list containing all position ids
    @param saveroot: root for saving the final results
    @param method: method name
    """
    totalparticle = []
    for j in positionlist:
        xid = int(j.split("_")[1])
        yid = int(j.split("_")[0])
        singletxtpath = os.path.join(root, method, j, j + ".txt")
        try:
            with open(singletxtpath, "r") as f:
                for line in f.readlines():
                    x = float(line.strip().split(" ")[0]) + W * xid
                    y = float(line.strip().split(" ")[1]) + H * yid
                    z = int(line.strip().split(" ")[2]) + 1
                    d = float(line.strip().split(" ")[3]) * pixel
                    totalparticle.append([x, y, z, d])
        except:
            pass
    with open(os.path.join(saveroot, method + "_allparticle.txt"), "w") as g:
        for h in totalparticle:
            g.write("{} {} {} {}\n".format(h[0], h[1], h[2], h[3]))
    return os.path.join(saveroot, method + "_allparticle.txt")


def get_particle_information(txtpath, d_need):
    """

    @param txtpath: results of certain method
    @param d_need: whether to contain the diameter information
    @return:
    """
    allinfor = []
    with open(txtpath, "r") as f:
        for line in f.readlines():
            if d_need:
                [x, y, z, d] = [float(j) for j in line.strip().split(" ")[:4]]
                allinfor.append([x, y, z, d])
            else:
                [x, y, z] = [float(j) for j in line.strip().split(" ")[:3]]
                allinfor.append([x, y, z])
    return allinfor


def ini_preparation(
    reconstructed_img_root,
    H_multiple,
    W_multiple,
    detectionmethods,
    segmentationmethods,
    mode,
    need_create_file,
    need_cut,
    saveroot=r"results",
):
    """

    @param reconstructed_img_root: reconstructed image root with original resolution
    @param H_multiple: cut times in H dimension
    @param W_multiple: cut times in W dimension
    @param detectionmethods:
    @param segmentationmethods:
    @param mode: exp or cali
    @param saveroot:
    @return:
    """
    positionlist = generate_positionlist(W_multiple, H_multiple)
    methodnamelist = generate_methodlist(detectionmethods, segmentationmethods)

    savecaseroot = os.path.join(saveroot, mode)
    savecutimgroot = os.path.join(savecaseroot, "images")

    if need_create_file:
        try:
            os.mkdir(savecaseroot)
        except:
            pass
        try:
            os.mkdir(savecutimgroot)
        except:
            pass
        for patch in positionlist:
            os.mkdir(os.path.join(savecutimgroot, patch))
            for method in methodnamelist:
                try:
                    os.mkdir(os.path.join(savecaseroot, method))
                except:
                    pass
                os.mkdir(os.path.join(savecaseroot, method, patch))
            for det in detectionmethods:
                try:
                    os.mkdir(os.path.join(savecaseroot, det + "_results"))
                except:
                    pass
                os.mkdir(os.path.join(savecaseroot, det + "_results", patch))
    if need_cut:
        ori_imglist = os.listdir(reconstructed_img_root)
        for i in ori_imglist:
            imgpath = os.path.join(reconstructed_img_root, i)
            cutimg(imgpath, savecutimgroot)


def cal_average_std(error):
    return np.average(error * 100, axis=0), np.std(error * 100, axis=0)


def cal_error(
    root,
    positionlist,
    saveroot,
    method,
    truepath,
    d_need,
    mode,
    H=270,
    W=360,
    pixel=3.45,
):
    """

    @param predpath: particle information path
    @param truepath: true particle information path
    @param d_need: whether to calculate the error of diameter
    @return:
    """
    predpath = extract_fuse_info(root, positionlist, saveroot, method, H, W, pixel)
    pred = get_particle_information(predpath, d_need)
    true = get_particle_information(truepath, d_need)
    pred = np.array(pred)
    true = np.array(true)

    allcost = []
    error_100 = []
    error_50 = []
    error_40 = []
    error_30 = []
    error_20 = []
    lost_100 = 0
    lost_50 = 0
    lost_40 = 0
    lost_30 = 0
    lost_20 = 0
    alllost = 0
    for i in true:
        if d_need:
            pred_contrast = pred[:, :-1] - i[:-1]
            pred_contrastxy = pred[:, :-2] - i[:-2]
            pred_contrastz = np.abs(pred[:, -2] - i[-2])
        else:
            pred_contrast = pred - i
            pred_contrastxy = pred[:, :-1] - i[:-1]
            pred_contrastz = np.abs(pred[:, -1] - i[-1])
        cost = np.linalg.norm(pred_contrast, axis=-1)
        xydistance = np.linalg.norm(pred_contrastxy, axis=-1)
        m = np.min(cost)
        if m <= 10:
            id = np.where(cost == m)[0][0]
            zcost = pred_contrastz[id]
            xycost = xydistance[id]
            if d_need:
                dcost = abs((pred[id, :][-1] - i[-1]))
                allcost.append([xycost, zcost, dcost])
                if i[-1] == 100:
                    error_100.append([xycost, zcost, dcost / i[-1]])
                elif i[-1] == 50:
                    error_50.append([xycost, zcost, dcost / i[-1]])
                elif i[-1] == 40:
                    error_40.append([xycost, zcost, dcost / i[-1]])
                elif i[-1] == 30:
                    error_30.append([xycost, zcost, dcost / i[-1]])
                else:
                    error_20.append([xycost, zcost, dcost / i[-1]])

            else:
                allcost.append([xycost, zcost])
        else:
            if d_need:
                if i[-1] == 100:
                    lost_100 += 1
                elif i[-1] == 50:
                    lost_50 += 1
                elif i[-1] == 40:
                    lost_40 += 1
                elif i[-1] == 30:
                    lost_30 += 1
                else:
                    lost_20 += 1
            else:
                alllost += 1

    if d_need:
        error_20 = np.array(error_20)
        error_30 = np.array(error_30)
        error_40 = np.array(error_40)
        error_50 = np.array(error_50)
        error_100 = np.array(error_100)
        average_20, std_20 = cal_average_std(error_20)
        average_30, std_30 = cal_average_std(error_30)
        average_40, std_40 = cal_average_std(error_40)
        average_50, std_50 = cal_average_std(error_50)
        average_100, std_100 = cal_average_std(error_100)
        with open(os.path.join(saveroot, method + "_evaluate.txt"), "w") as f:
            f.write("lost_100:{}".format(lost_100) + "\n")
            f.write("lost_50:{}".format(lost_50) + "\n")
            f.write("lost_40:{}".format(lost_40) + "\n")
            f.write("lost_30:{}".format(lost_30) + "\n")
            f.write("lost_20:{}".format(lost_20) + "\n")
            f.write("average_errorz_20:{}".format(average_20[1] / 100) + "\n")
            f.write("average_errord_20:{}".format(average_20[-1]) + "\n")
            f.write("average_errorz_30:{}".format(average_30[1] / 100) + "\n")
            f.write("average_errord_30:{}".format(average_30[-1]) + "\n")
            f.write("average_errorz_40:{}".format(average_40[1] / 100) + "\n")
            f.write("average_errord_40:{}".format(average_40[-1]) + "\n")
            f.write("average_errorz_50:{}".format(average_50[1] / 100) + "\n")
            f.write("average_errord_50:{}".format(average_50[-1]) + "\n")
            f.write("average_errorz_100:{}".format(average_100[1] / 100) + "\n")
            f.write("average_errord_100:{}".format(average_100[-1]) + "\n")
            f.write("std_errorz_20:{}".format(std_20[1] / 100) + "\n")
            f.write("std_errord_20:{}".format(std_20[-1]) + "\n")
            f.write("std_errorz_30:{}".format(std_30[1] / 100) + "\n")
            f.write("std_errord_30:{}".format(std_30[-1]) + "\n")
            f.write("std_errorz_40:{}".format(std_40[1] / 100) + "\n")
            f.write("std_errord_40:{}".format(std_40[-1]) + "\n")
            f.write("std_errorz_50:{}".format(std_50[1] / 100) + "\n")
            f.write("std_errord_50:{}".format(std_50[-1]) + "\n")
            f.write("std_errorz_100:{}".format(std_100[1] / 100) + "\n")
            f.write("std_errord_100:{}".format(std_100[-1]) + "\n")
    else:
        allcost = np.array(allcost)
        allaverage, allstd = cal_average_std(allcost)
        with open(
            os.path.join(saveroot, mode + "_" + method + "_evaluate.txt"), "w"
        ) as f:
            f.write("alllost:{}".format(alllost) + "\n")
            f.write("average_errorz:{}".format(allaverage[1] / 100) + "\n")
            f.write("std_errorz:{}".format(allaverage[-1] / 100) + "\n")
