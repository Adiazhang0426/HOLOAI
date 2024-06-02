"""
@Description: main function for three tasks: training the detection network, extracting the information of the holographic particle field, evaluating with the ground truth
@Author: Adiazhang
@Date: 2024-05-29 13:45:51
@LastEditTime: 2024-05-30 10:49:23
@LastEditors: Adiazhang
"""

import os
import argparse
import multiprocessing as mp
from main_utils import (
    train_yolo,
    yolo_predict,
    location_segmentation_main,
    cal_error,
    ini_preparation,
    generate_methodlist,
    generate_positionlist,
)
from YOLOv7.detect_first_initiatemodel import detect_yolo, initiate_model


def main(opt):
    task = opt.task
    (
        detection_methods,
        segmentation_methods,
        H_multiple,
        W_multiple,
        save_run_root,
        mode,
        cropped_imgshape,
    ) = (
        opt.detection_methods,
        opt.segmentation_methods,
        opt.H_multiple,
        opt.W_multiple,
        opt.save_run_root,
        opt.mode,
        opt.cropped_imgshape,
    )
    positionlist, methodlist = generate_positionlist(
        W_multiple, H_multiple
    ), generate_methodlist(detection_methods, segmentation_methods)

    if task == "train":
        model, dataset, epoch, device, batch = (
            opt.model,
            opt.dataset,
            opt.epoch,
            opt.device,
            opt.batch,
        )

        train_yolo(model, dataset, epoch, device, batch)

    elif task == "run":
        (
            weights,
            detection_conf,
            save_detect_img,
            reconstructed_img_root,
            need_create_file,
            need_cut,
            mergeiou,
            seg_interval,
            ifv10s,
            device,
        ) = (
            opt.weights,
            opt.detection_conf,
            opt.save_detect_img,
            opt.reconstructed_img_root,
            opt.need_create_file,
            opt.need_cut,
            opt.mergeiou,
            opt.cal_interval,
            opt.ifv10s,
            opt.device,
        )

        ini_preparation(
            reconstructed_img_root,
            H_multiple,
            W_multiple,
            detection_methods,
            segmentation_methods,
            mode,
            need_create_file,
            need_cut,
            save_run_root,
        )
        print("----------------ini_preparation finished!----------------")
        save_run_root = os.path.join(save_run_root, mode)
        cropped_img_root = os.path.join(save_run_root, "images")
        for weight, detection_method, ifv10 in zip(weights, detection_methods, ifv10s):
            if "yolov7" not in detection_method:
                for patch in positionlist:
                    source = os.path.join(cropped_img_root, patch)
                    save_txt_root = os.path.join(
                        save_run_root, detection_method + "_results", patch
                    )
                    yolo_predict(
                        weight,
                        source,
                        save_txt_root,
                        True,
                        device,
                        detection_conf,
                        ifv10,
                        save_detect_img,
                    )
            else:
                model, stride = initiate_model(
                    weight, True, not save_detect_img, device
                )
                for patch in positionlist:
                    detect_yolo(
                        model=model,
                        stride=stride,
                        sources=os.path.join(cropped_img_root, patch),
                        project=os.path.join(
                            save_run_root, detection_method + "_results", patch
                        ),
                        save_img=save_detect_img,
                        thre=detection_conf,
                    )
        print("----------------Detection finished!----------------")
        for method in methodlist:
            p_list = []
            detection_method = method.split("_")[0]

            for patch in positionlist:
                save_method_root = os.path.join(save_run_root, method, patch)
                if "yolov7" in method:
                    detect_txt_root = os.path.join(
                        save_run_root,
                        detection_method + "_results",
                        patch,
                        "exp",
                        "labels",
                    )
                else:
                    detect_txt_root = os.path.join(
                        save_run_root,
                        detection_method + "_results",
                        patch,
                        "train",
                        "labels",
                    )
                p = mp.Process(
                    target=location_segmentation_main,
                    args=(
                        cropped_img_root,
                        detect_txt_root,
                        patch,
                        save_method_root,
                        method,
                        seg_interval,
                        mergeiou,
                        cropped_imgshape,
                    ),
                )
                p_list.append(p)
            for p in p_list:
                p.start()
            for p in p_list:
                p.join()
        print("----------------Segmentation finished!----------------")
    else:
        save_run_root = os.path.join(save_run_root, mode)
        save_evaluate_root, pixel, ground_truth_path = (
            opt.save_evaluate_root,
            opt.pixel,
            opt.ground_truth_txtpath,
        )
        save_evaluate_root = os.path.join(save_evaluate_root, mode)
        try:
            os.mkdir(save_evaluate_root)
        except:
            pass
        if "exp" in mode:
            d_need = False
        else:
            d_need = True
        p_list = []
        for method in methodlist:
            p = mp.Process(
                target=cal_error,
                args=(
                    save_run_root,
                    positionlist,
                    save_evaluate_root,
                    method,
                    ground_truth_path,
                    d_need,
                    mode,
                    cropped_imgshape[1],
                    cropped_imgshape[0],
                    pixel,
                ),
            )
            p_list.append(p)
        for p in p_list:
            p.start()
        for p in p_list:
            p.join()
        print("----------------Evaluation finished!----------------")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "./YOLOv7")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        nargs="+",
        type=str,
        default="evaluate",
        help="train or run or evaluate",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        type=str,
        default="",
        help="model configuration for training",
    )
    parser.add_argument(
        "--dataset", nargs="+", type=str, default="", help="training dataset"
    )
    parser.add_argument("--epoch", nargs="+", type=int, default=500)
    parser.add_argument("--batch", nargs="+", type=int, default=16)
    parser.add_argument(
        "--weights",
        nargs="+",
        type=list,
        default=[r"weights\yolov10x.pt"],
        help="path for best.pt,should match the order of detection methods",
    )
    parser.add_argument(
        "--detection_conf",
        nargs="+",
        type=float,
        default=0.05,
        help="determine the lowest confidence of output boxes",
    )
    parser.add_argument("--save_run_root", nargs="+", type=str, default=r"results")
    parser.add_argument("--save_evaluate_root", nargs="+", type=str, default=r"tests")
    parser.add_argument(
        "--save_detect_img",
        nargs="+",
        type=bool,
        default=False,
        help="whether to save image drew with boxes",
    )
    parser.add_argument(
        "--ifv10s", nargs="+", type=list, default=[True], help="whether to use yolov10"
    )
    parser.add_argument("--device", nargs="+", type=str, default="0")
    parser.add_argument("--pixel", nargs="+", type=float, default=3.45)
    parser.add_argument(
        "--reconstructed_img_root",
        nargs="+",
        type=str,
        default=r"H:\ice_particle\2-case2-pw180-pa60-0.56\reconstruct\Image_20230524182723_w1440_h1080_10",
        help="reconstructed holographic image root with original resolution",
    )
    parser.add_argument(
        "--H_multiple", nargs="+", type=int, default=4, help="cut times in H dimension"
    )
    parser.add_argument(
        "--W_multiple", nargs="+", type=int, default=4, help="cut times in W dimension"
    )
    parser.add_argument(
        "--detection_methods",
        nargs="+",
        type=list,
        default=["yolov10x"],
        help="yolovs detection model",
    )
    parser.add_argument(
        "--segmentation_methods",
        nargs="+",
        type=list,
        default=["grayvar"],
        help="segmentation method",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        type=str,
        default="exp_ice",
        help="cali or exp, the casename",
    )
    parser.add_argument(
        "--need_create_file",
        nargs="+",
        type=bool,
        default=True,
        help="whether to create files for later process",
    )
    parser.add_argument(
        "--need_cut", nargs="+", type=bool, default=True, help="whether to crop image"
    )
    parser.add_argument(
        "--mergeiou",
        nargs="+",
        type=float,
        default=0.4,
        help="threshold to determine the boxes belong to same particle",
    )
    parser.add_argument(
        "--cropped_imgshape", nargs="+", type=list, default=[270, 360], help="[H,W]"
    )
    parser.add_argument(
        "--cal_interval",
        nargs="+",
        type=int,
        default=20,
        help="number of slices used for calculating the extra feature map on each side",
    )
    parser.add_argument(
        "--ground_truth_txtpath",
        nargs="+",
        type=str,
        default="",
    )
    opt = parser.parse_args()
    main(opt)
